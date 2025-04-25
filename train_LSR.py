import sys
sys.path.append("../")
import os
import argparse
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import ImageFolder
from losses import LSR_Loss
import logging
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from util import DWT,IWT,setup_logger
from models.LRH_fr import LRH_f, LRH_r
from models.LSR import Model as LSR
from models.denoisenet import Network as DN
from tqdm import tqdm
import lpips
import pdb






def downsample(hr,scale):
    lr = F.interpolate(hr, scale_factor=1.0/scale, mode='bicubic')
    lr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
    return lr

def guass_blur(hr,k_sz,sigma):
    transform = transforms.GaussianBlur(kernel_size=k_sz,sigma=sigma)
    return transform(hr)
    
def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())


def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    return optimizer



def train_one_epoch(LRH_f, LRH_r, LSR, DN, criterion, train_dataloader, optimizer_f, optimizer_r, optimizer_lsr, optimizer_DN, sigma_current_max, epoch, logger_train, tb_logger, args):
    if args.finetune:
        LRH_f.train()
        LRH_r.train()
    else:
        for param in LRH_f.parameters():
            param.requires_grad=False
        for param in LRH_r.parameters():
            param.requires_grad=False
    for param in DN.parameters():
        param.requires_grad=False
    LSR.train()
 
    device = next(LRH_f.parameters()).device
    dwt = DWT()
    iwt = IWT()

    sigma_min, _ = args.attack_level
    total_loss = 0.0
    num_batches = len(train_dataloader)

    for i, d in enumerate(train_dataloader):

        batch_size = d.shape[0]
        d = d.to(device)  #[16,3,224,224]
        cover_img = d[d.shape[0] // 2:, :, :, :]  #[8,3,224,224]
        secret_img = d[:d.shape[0] // 2, :, :, :]
        #mix
        noiselvl = np.random.uniform(0,55,size=1) #random noise level
        noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=noiselvl[0] / 255.) 
        noise_secret = secret_img + noise 
        blur_secret = guass_blur(noise_secret,2*random.randint(0,11)+3,random.uniform(0.1,2))
        scalelvl = random.choice([2,4])
        mix_secret = downsample(blur_secret,scalelvl)  
       
        input_cover = dwt(cover_img)
        input_secret = dwt(mix_secret)

        
        if args.finetune:
            optimizer_f.zero_grad()
            optimizer_r.zero_grad()
        optimizer_lsr.zero_grad()
        optimizer_DN.zero_grad()
        #################
        # hide#
        #################

        output_steg, output_z = LRH_f(input_cover, input_secret)
        steg_ori = iwt(output_steg)

        # 攻击阶段
        sigma = np.random.uniform(sigma_min, sigma_current_max, size=1)
        noise = torch.cuda.FloatTensor(steg_ori.size()).normal_(mean=0, std=sigma[0])

        noised_steg = steg_ori + noise

        # 去噪阶段
        noise_level_est, denoised_steg = DN(noised_steg)

        #################
        #denoise#
        #################
        steg_img = LSR(denoised_steg)
        output_clean = dwt(steg_img)

        #################
        #reveal#
        #################
        output_z_guass = gauss_noise(output_z.shape)
        cover_rev, secret_rev = LRH_r(output_clean, output_z_guass, rev=True)
        rec_img = iwt(secret_rev)
        
        #loss
        out_criterion = criterion(secret_img,cover_img,steg_ori,steg_img,rec_img,args.sweight,args.cweight,args.pweight,args.finetune)
        loss = out_criterion['loss']
        loss.backward()
        optimizer_lsr.step()
        if args.finetune:
            optimizer_f.step()
            optimizer_r.step()
    
        total_loss += loss.item()

        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"[attack level: {sigma[0]:.6f}/{sigma_current_max:.6f}]"
                f'\tLoss: {loss.item():.3f} |'
        
            )
    avg_loss = total_loss / num_batches
    logger_train.info(f"Train epoch {epoch}: Average loss: {avg_loss:.3f}, sigma: {sigma_current_max:.3f}")
    tb_logger.add_scalar('[train]: loss', avg_loss, epoch)
    return avg_loss


def test_epoch(args,epoch, test_dataloader, LRH_f, LRH_r, LSR, DN, logger_val,criterion,lpips_fn,degrate_type):
    dwt = DWT()
    iwt = IWT()
    LRH_f.eval()
    LRH_r.eval()
    DN.eval()
    LSR.eval() #在测试时禁用BN，避免batchsize为1时的问题
    device = next(LRH_f.parameters()).device
    psnrc = AverageMeter()
    ssimc = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()
    psnrcori = AverageMeter()
    ssimcori = AverageMeter()
    lpipsc =  AverageMeter()
    lpipss =  AverageMeter()
    lpipscori =  AverageMeter()
    loss = AverageMeter()
    i=0
    sigma_total = 0.0
    with torch.no_grad():
        for idx,d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            d = d.to(device)
            cover_img = d[d.shape[0] // 2:, :, :, :]  #[1,3,224,224]
            secret_img = d[:d.shape[0] // 2, :, :, :]
            if degrate_type == 1:
                noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=25 / 255.)  
                noise_secret_img = secret_img + noise 
                input_secret_img = noise_secret_img
               
            elif degrate_type == 2:
                blur_secret_img = guass_blur(secret_img,15,1.6)
                input_secret_img = blur_secret_img    

            elif degrate_type == 3:
                lr_secret_img = downsample(secret_img,4)
                input_secret_img = lr_secret_img       
                
            else:
                noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=25 / 255.)  
                noise_secret_img = secret_img + noise 
                blur_secret_img = guass_blur(noise_secret_img,9,1)
                lr_secret_img = downsample(blur_secret_img,2)
                input_secret_img = lr_secret_img

            input_cover = dwt(cover_img)
            input_secret = dwt(input_secret_img)
            # hide
            output_steg, output_z = LRH_f(input_cover,input_secret)
            steg_ori = iwt(output_steg)
            
            # 攻击阶段
            sigma_min, sigma_max = args.attack_level
            sigma_current_max = sigma_min + (sigma_max - sigma_min) * (epoch / args.epochs)
            sigma = np.random.uniform(sigma_min, sigma_current_max, size=1)
            sigma_total += sigma[0]
            noise = torch.cuda.FloatTensor(steg_ori.size()).normal_(mean=0, std=sigma[0])
            noised_steg = steg_ori + noise

            # 去噪阶段
            noise_level_est, denoised_steg = DN(noised_steg)
            #denoise
            steg_clean = LSR(denoised_steg)
            output_clean = dwt(steg_clean)

            
            #reveal
            output_z_guass = gauss_noise(output_z.shape)
            cover_rev, secret_rev= LRH_r(output_clean, output_z_guass,rev=True)
            rec_img = iwt(secret_rev)

            
            out_criterion = criterion(secret_img,cover_img,steg_clean,steg_ori,rec_img,args.sweight,args.cweight,args.pweight,args.finetune)
            loss.update(out_criterion["loss"])
            
            #comute lpips tensor
            lc = lpips_fn.forward(cover_img,steg_clean)
            ls = lpips_fn.forward(secret_img,rec_img)
            lori = lpips_fn.forward(cover_img,steg_ori)

            lpipsc.update(lc.mean().item())
            lpipss.update(ls.mean().item())
            lpipscori.update(lori.mean().item())

            #compute psnr and save image
            save_dir = os.path.join('experiments', args.experiment,'images')
            
            secret_img = torch2img(secret_img)
            cover_img = torch2img(cover_img)
            degrade_secret_img = torch2img(input_secret_img)

            secret_img_rec = torch2img(rec_img)
            steg_img = torch2img(steg_clean)
            steg_img_ori = torch2img(steg_ori)
            denoised_steg_img = torch2img(denoised_steg)

            p1, m1 = compute_metrics(secret_img_rec, secret_img)
            psnrs.update(p1)
            ssims.update(m1)
            p2, m2 = compute_metrics(steg_img, cover_img)
            psnrc.update(p2)
            ssimc.update(m2)
            p3, m3 = compute_metrics(steg_img_ori, cover_img)
            psnrcori.update(p3)
            ssimcori.update(m3)
                
            if args.save_img:
                denoised_steg_dir = os.path.join(save_dir, 'denoised_steg', str(degrate_type))
                if not os.path.exists(denoised_steg_dir):
                    os.makedirs(denoised_steg_dir)
                denoised_steg_img.save(os.path.join(denoised_steg_dir, '%03d.png' % i))

                rec_dir = os.path.join(save_dir,'rec',str(degrate_type))
                if not os.path.exists(rec_dir):
                    os.makedirs(rec_dir)

                secret_dir = os.path.join(save_dir,'secret',str(degrate_type))
                if not os.path.exists(secret_dir):
                    os.makedirs(secret_dir)

                cover_dir = os.path.join(save_dir,'cover')
                if not os.path.exists(cover_dir):
                    os.makedirs(cover_dir)
                
                stego_dir = os.path.join(save_dir,'stego',str(degrate_type))
                if not os.path.exists(stego_dir):
                    os.makedirs(stego_dir)

                ori_stego_dir = os.path.join(save_dir,'ori_stego',str(degrate_type))
                if not os.path.exists(ori_stego_dir):
                    os.makedirs(ori_stego_dir)
                
                steg_img.save(os.path.join(stego_dir,'%03d.png' % i))
                secret_img_rec.save(os.path.join(rec_dir,'%03d.png' % i))

                degrade_secret_img.save(os.path.join(secret_dir,'%03d.png' % i))
                cover_img.save(os.path.join(cover_dir,'%03d.png' % i))

                steg_img_ori.save(os.path.join(ori_stego_dir,'%03d.png' % i))

                i=i+1


    logger_val.info(
        f"Test epoch {epoch} - Degrate Type {degrate_type}: Average losses:"
        f"\tPSNRC: {psnrc.avg:.6f} |"
        f"\tSSIMC: {ssimc.avg:.6f} |"
        f"\tLPIPSC: {lpipsc.avg:.6f} |"
        f"\tPSNRS: {psnrs.avg:.6f} |"
        f"\tSSIMS: {ssims.avg:.6f} |"
        f"\tLPIPSS: {lpipss.avg:.6f} |"
        f"\tPSNRCORI: {psnrcori.avg:.6f} |"
        f"\tSSIMCORI: {ssimcori.avg:.6f} |"
        f"\tLPIPSORI: {lpipscori.avg:.6f} |\n"
        f"Attack level: {(sigma_total/i):.6f}/{sigma_current_max:.6f}\n"
    )

              



def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        base_name = os.path.basename(filename)  # 提取文件名
        dir_name = os.path.dirname(filename)   # 提取目录路径
        best_filename = os.path.join(dir_name, f"best_{base_name}")
        shutil.copyfile(filename, best_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-d_test", "--test_dataset", type=str, required=True, help="Testing dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=3000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(224,224),
        help="Size of the training patches to be cropped (default: %(default)s)",
    ),
    parser.add_argument(
        "--test-patch-size",
        type=int,
        nargs=2,
        default=(1024, 1024),
        help="Size of the testing patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--checkpoint_f", type=str, help="Path to forward checkpoint"),
    parser.add_argument("--checkpoint_r", type=str, help="Path to reverse checkpoint"),
    parser.add_argument("--checkpoint_DN", type=str, help="Path to DNNet checkpoint"),
    parser.add_argument("--checkpoint_lsr", type=str, help="Path to a LSR checkpoint"),
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    ),
    parser.add_argument(
        "--val_freq", type=int,  default=30, help="how often should an evaluation be performed"
    ),
    parser.add_argument(
        "--channels_in", type=int,  default=12, help="channels into punet"
    ),
    parser.add_argument("--num-steps_f", type=int, help="steps of LRH forward"),
    parser.add_argument("--num-steps_r", type=int, help="steps of LRH reverse"),
    parser.add_argument(
        "--klvl", type=int,  default=3, help="num of scales in LSR"
    ),
    parser.add_argument(
        "--mid", type=int,  default=2,help="middle_blk_num in SRM"
    ),
    parser.add_argument(
        "--enc", default = [2,2,4], nargs='+', type=int, help="enc_blk_num in SRM"
    ),
    parser.add_argument(
        "--dec", default = [2,2,2], nargs='+', type=int, help="dec_blk_num in SRM"
    ),
    parser.add_argument(
        "--save_img", action="store_true", default=False, help="Save model to disk"
    )
    parser.add_argument("--spn", default = 2,type=int, help="the ratio of noisy samples"),
    parser.add_argument("--spb", default = 2,type=int, help="the ratio of noisy samples"),
    parser.add_argument("--spl", default = 2,type=int, help="the ratio of noisy samples"),
    parser.add_argument("--spm", default = 2,type=int, help="the ratio of noisy samples"),
    parser.add_argument(
        "--test", action="store_true", default=False, help="test"
    ),
    parser.add_argument(
        "--finetune", action="store_true", default=False, help="train LRH and LSR in an endtoend manner"
    ),
    parser.add_argument(
        "--std", type=float,  default=1.6, help="Standard deviation"
    ),
    parser.add_argument(
        "--sweight", type=float,  default=2, help="weight of restoration loss"
    ),
    parser.add_argument(
        "--cweight", type=float,  default=1, help="weight of security loss"
    ),
    parser.add_argument(
        "--pweight", type=float,  default=0.01,help="weight of perceptual loss"
    ),
    parser.add_argument(
        "--lfrestore", type=bool, nargs='?', const=True, default=True, help="Save model to disk"
    ),
    parser.add_argument(
        "--steps", type=int,  default=4, help="num of wlblocks in each scale of LSR"
    ),
    parser.add_argument("--nafwidth", default = 32,type=int, help="the ratio of noisy samples"),
    parser.add_argument("--attack_level", type=float, nargs=2, default=[0.0, 0.1], help="Lower and upper bounds for sigma range in Gaussian noise attack")
    parser.add_argument(
        "--patience",
        default=150,
        type=int,
        help="Patience for early stopping (default: %(default)s)",
    ),
    parser.add_argument(
        "--test_degrade_type",
        type=int,
        default=4,
        help="test degrade type (default: %(default)s)",
    ),
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))

    setup_logger('train', os.path.join('experiments', args.experiment), 'train_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)
    setup_logger('val', os.path.join('experiments', args.experiment), 'val_' + args.experiment,
                      level=logging.INFO,
                      screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('experiments', args.experiment, 'checkpoints'))


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.test_patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="", transform=train_transforms)
    test_dataset = ImageFolder(args.test_dataset, split="", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    lrh_f = LRH_f(args.num_steps_f).to(device)
    lrh_r = LRH_r(args.num_steps_r).to(device)
    DN = DN().to(device)
  
    

    lsr = LSR(steps = args.steps,klvl=args.klvl,mid=args.mid,enc=args.enc,dec=args.dec,lfrestore=args.lfrestore,width=args.nafwidth)
    lsr = lsr.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        lsr = CustomDataParallel(lsr)
    logger_train.info(args)

    optimizer_f = configure_optimizers(lrh_f, args)
    optimizer_r = configure_optimizers(lrh_r, args)
    optimizer_DN = torch.optim.Adam(DN.parameters(), lr=args.learning_rate)
    optimizer_lsr = configure_optimizers(lsr, args)
    # 加载检查点
    if args.checkpoint_f:
        print("Loading lrh_f from", args.checkpoint_f)
        checkpoint_f = torch.load(args.checkpoint_f, map_location=device)
        last_epoch_f = checkpoint_f["epoch"] + 1
        lrh_f.load_state_dict(checkpoint_f["state_dict"])
        optimizer_f.load_state_dict(checkpoint_f["optimizer"])
        optimizer_f.param_groups[0]['lr'] = args.learning_rate

    if args.checkpoint_r:
        print("Loading lrh_r from", args.checkpoint_r)
        checkpoint_r = torch.load(args.checkpoint_r, map_location=device)
        last_epoch_r = checkpoint_r["epoch"] + 1
        lrh_r.load_state_dict(checkpoint_r["state_dict"])
        optimizer_r.load_state_dict(checkpoint_r["optimizer"])
        optimizer_r.param_groups[0]['lr'] = args.learning_rate

    if args.checkpoint_DN:
        print("Loading DN from", args.checkpoint_DN)
        checkpoint_DN = torch.load(args.checkpoint_DN, map_location=device)
        state_dict = checkpoint_DN["state_dict"]
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        DN.load_state_dict(new_state_dict)
        optimizer_DN.load_state_dict(checkpoint_DN["optimizer"])
        optimizer_DN.param_groups[0]['lr'] = args.learning_rate
    
    if args.checkpoint_lsr:  # load from previous checkpoint
        print("Loading lsr from", args.checkpoint_lsr)
        checkpoint_lsr= torch.load(args.checkpoint_lsr, map_location=device)
        last_epoch = checkpoint_lsr["epoch"] + 1
        best_loss = checkpoint_lsr["best_loss"]
        lsr.load_state_dict(checkpoint_lsr["state_dict"])
        optimizer_lsr.load_state_dict(checkpoint_lsr["optimizer"])
        optimizer_lsr.param_groups[0]['lr'] = args.learning_rate

    criterion = LSR_Loss()
    lpips_fn = lpips.LPIPS(net='alex',version='0.1')
    lpips_fn.cuda()
    
    last_epoch = 0
    loss = float("inf")
    best_loss = float("inf")
    
    sigma_min, sigma_max = args.attack_level
    sigma_previous_max = 1e-6
    if not args.test:
        for epoch in range(last_epoch, args.epochs):
            logger_train.info(f"Learning rate: {optimizer_lsr.param_groups[0]['lr']}")
            sigma_current_max = sigma_min + (sigma_max - sigma_min) * (epoch / args.epochs)

            if sigma_current_max > sigma_previous_max:
                best_loss = best_loss * sigma_current_max / sigma_previous_max
                sigma_previous_max = sigma_current_max
            loss = train_one_epoch(
                lrh_f,
                lrh_r,
                lsr,
                DN,
                criterion,
                train_dataloader,
                optimizer_f,
                optimizer_r,
                optimizer_lsr,
                optimizer_DN,
                sigma_current_max,
                epoch,
                logger_train,
                tb_logger,
                args
            )
            if epoch % args.val_freq == 0:
                degrate_type = 4
                test_epoch(args, epoch, test_dataloader, lrh_f, lrh_r, lsr, DN,logger_val,criterion,lpips_fn,degrate_type)

            is_best = loss < best_loss
            if is_best:
                best_loss = loss
                last_improvement_epoch = epoch
            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": lsr.state_dict(),
                        "best_loss":best_loss,
                        "optimizer": optimizer_lsr.state_dict(),
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "lsr_checkpoint.pth.tar")
                )
                if args.finetune:
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": lrh_f.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer_f.state_dict(),
                        },
                        is_best,
                        os.path.join('experiments', args.experiment, 'checkpoints', "lrh_f_checkpoint.pth.tar")
                    )
                    save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": lrh_r.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer_r.state_dict(),
                    },
                    is_best,
                    os.path.join('experiments', args.experiment, 'checkpoints', "lrh_r_checkpoint.pth.tar")
                )
                if is_best:
                    logger_train.info(f"Best checkpoints saved, best loss: {best_loss:.4f}")
            if last_improvement_epoch != -1 and epoch - last_improvement_epoch >= args.patience:
                logger_train.info(f"No improvement in loss for {args.patience} epochs, early stopping triggered.")
                break
    else:
        test_epoch(args, args.epochs, test_dataloader, lrh_f, lrh_r, lsr, DN,logger_val,criterion,lpips_fn,args.test_degrade_type)


if __name__ == "__main__":
    main(sys.argv[1:])

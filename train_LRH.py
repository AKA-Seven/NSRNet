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
from losses import LRH_Loss
import logging
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from pytorch_msssim import ms_ssim
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from models.LRH import LRH_f, LRH_r
from util import DWT,IWT,setup_logger
from tqdm import tqdm






def downsample(hr,scale):
    lr = F.interpolate(hr, scale_factor=1.0/scale, mode='bicubic')
    lr = F.interpolate(lr, scale_factor=scale, mode='bicubic')
    return lr

def gauss_blur(hr,k_sz,sigma):
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


def train_one_epoch(LRH_f, LRH_r, criterion, train_dataloader, optimizer_f, optimizer_r, epoch, logger_train, tb_logger, args):
    LRH_f.train()
    LRH_r.train()
    device = next(LRH_f.parameters()).device

    dwt = DWT()
    iwt = IWT()

    total_loss = 0

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        cover_img = d[d.shape[0] // 2:, :, :, :]
        secret_img = d[:d.shape[0] // 2, :, :, :]
        p = np.array([args.brate, args.nrate, args.lrate])
        type = np.random.choice(args.data_type, p=p.ravel())

        if type == 1:
            # blur
            blur_secret_img = gauss_blur(secret_img, 2 * random.randint(0, 11) + 3, random.uniform(0.1, 2))
            input_secret_img = blur_secret_img    
        elif type == 2:
            # add noise
            noiselvl = np.random.uniform(0, 55, size=1)  # random noise level
            noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=noiselvl[0] / 255.)  
            noise_secret_img = secret_img + noise 
            input_secret_img = noise_secret_img
        else:
            # down sample to low resolution
            scalelvl = random.choice([2, 4])
            lr_secret_img = downsample(secret_img, scalelvl)
            input_secret_img = lr_secret_img       

        input_cover = dwt(cover_img)
        input_secret = dwt(input_secret_img)

        optimizer_f.zero_grad()
        optimizer_r.zero_grad()

        #################
        # Hide
        #################
        output_steg, output_z = LRH_f(input_cover, input_secret)
        steg_img = iwt(output_steg)

        #################
        # Reveal
        #################
        steg_img_dwt = dwt(steg_img)
        output_z_guass = gauss_noise(output_z.shape)
        cover_rev, secret_rev = LRH_r(steg_img_dwt, output_z_guass, rev=True)
        secret_rev = iwt(secret_rev)

        #################
        # Loss
        #################
        steg_low = output_steg.narrow(1, 0, 3)
        cover_low = input_cover.narrow(1, 0, 3)

        out_criterian = criterion(
            input_secret_img, cover_img, steg_img, secret_rev, steg_low, cover_low,
            args.rec_weight, args.guide_weight, args.freq_weight
        )
        hide_loss = out_criterian['hide_loss']
        hide_loss.backward()

        optimizer_f.step()
        optimizer_r.step()

        total_loss += hide_loss.item()

        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\thide loss: {hide_loss.item():.3f} |'
            )

    tb_logger.add_scalar('{}'.format('[train]: hide_loss'), hide_loss.item(), epoch)

    return total_loss 



def test_epoch(args,epoch, test_dataloader, LRH_f, LRH_r, logger_val ):
    dwt = DWT()
    iwt = IWT()
    LRH_f.eval()
    LRH_r.eval()
    device = next(LRH_f.parameters()).device
    psnrc_n = AverageMeter()
    psnrs_n = AverageMeter()
    ssimc_n = AverageMeter()
    ssims_n = AverageMeter()
    psnrc_b = AverageMeter()
    psnrs_b = AverageMeter()
    ssimc_b = AverageMeter()
    ssims_b = AverageMeter()
    psnrc_l = AverageMeter()
    psnrs_l = AverageMeter()
    ssimc_l = AverageMeter()
    ssims_l = AverageMeter()

    i=0
    with torch.no_grad():
        for d in tqdm(test_dataloader):
            d = d.to(device)
            cover_img = d[d.shape[0] // 2:, :, :, :]
            secret_img = d[:d.shape[0] // 2, :, :, :]

            #blur
            blur_secret_img = gauss_blur(secret_img,15,args.sigma)

            #add noise
            noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=25 / 255.)  
            noise_secret_img = secret_img + noise 

            # down sampe to low resolution
            lr_secret_img = downsample(secret_img,4)


            input_cover = dwt(cover_img)
            noise_secret = dwt(noise_secret_img)
            blur_secret = dwt(blur_secret_img)
            lr_secret = dwt(lr_secret_img)
            #################
            # hide#
            #################

            output_stegn, output_z = LRH_f(input_cover,noise_secret)
            steg_imgn = iwt(output_stegn)
            output_stegb, output_z = LRH_f(input_cover,blur_secret)
            steg_imgb = iwt(output_stegb)
            output_stegl, output_z = LRH_f(input_cover,lr_secret)
            steg_imgl = iwt(output_stegl)
            #################
            #reveal#
            #################
            output_z_guass = gauss_noise(output_z.shape)
            cover_revn, secret_revn= LRH_r(output_stegn, output_z_guass,rev=True)
            secret_revn = iwt(secret_revn)
            cover_revb, secret_revb= LRH_r(output_stegb, output_z_guass,rev=True)
            secret_revb = iwt(secret_revb)
            cover_revl, secret_revl= LRH_r(output_stegl, output_z_guass,rev=True)
            secret_revl = iwt(secret_revl)

            save_dir = os.path.join('experiments', args.experiment,'images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            secret_img = torch2img(secret_img)
            cover_img = torch2img(cover_img)

            noise_secret_img = torch2img(noise_secret_img)
            steg_imgn = torch2img(steg_imgn)
            secret_revn = torch2img(secret_revn)
            p1, m1 = compute_metrics(secret_revn, noise_secret_img)
            psnrs_n.update(p1)
            ssims_n.update(m1)
            p2, m2 = compute_metrics(steg_imgn, cover_img)
            psnrc_n.update(p2)
            ssimc_n.update(m2)

            blur_secret_img = torch2img(blur_secret_img)
            steg_imgb = torch2img(steg_imgb)
            secret_revb = torch2img(secret_revb)
            p1, m1 = compute_metrics(secret_revb, blur_secret_img)
            psnrs_b.update(p1)
            ssims_b.update(m1)
            p2, m2 = compute_metrics(steg_imgb, cover_img)
            psnrc_b.update(p2)
            ssimc_b.update(m2)
        

            lr_secret_img = torch2img(lr_secret_img)
            steg_imgl = torch2img(steg_imgl)
            secret_revl = torch2img(secret_revl)
            p1, m1 = compute_metrics(secret_revl, lr_secret_img)
            psnrs_l.update(p1)
            ssims_l.update(m1)
            p2, m2 = compute_metrics(steg_imgl, cover_img)
            psnrc_l.update(p2)
            ssimc_l.update(m2)
           
            
            if args.save_images:

                cover_dir = os.path.join(save_dir,'cover')
                if not os.path.exists(cover_dir):
                    os.makedirs(cover_dir)
                stego_dir = os.path.join(save_dir,'stego')
                if not os.path.exists(stego_dir):
                    os.makedirs(stego_dir)
                secret_dir = os.path.join(save_dir,'secret')
                if not os.path.exists(secret_dir):
                    os.makedirs(secret_dir)
                noise_secret_dir = os.path.join(save_dir,'secret_noise')
                if not os.path.exists(noise_secret_dir):
                    os.makedirs(noise_secret_dir)
                rec_dir = os.path.join(save_dir,'recover')
                if not os.path.exists(rec_dir):
                    os.makedirs(rec_dir)
                lr_secret_dir = os.path.join(save_dir,'secret_lr')
                if not os.path.exists(lr_secret_dir):
                    os.makedirs(lr_secret_dir)
                blur_secret_dir = os.path.join(save_dir,'secret_blur')
                if not os.path.exists(blur_secret_dir):
                    os.makedirs(blur_secret_dir)

                blur_secret_img.save(os.path.join(blur_secret_dir,'%03d.png' % i))
                bstego_dir = os.path.join(stego_dir,'blur')
                brec_dir = os.path.join(rec_dir,'blur')
                if not os.path.exists(bstego_dir):
                    os.makedirs(bstego_dir)
                if not os.path.exists(brec_dir):
                    os.makedirs(brec_dir)
                steg_imgb.save(os.path.join(bstego_dir,'%03d.png' % i))
                secret_revb.save(os.path.join(brec_dir, '%03d.png' % i))

                lr_secret_img.save(os.path.join(lr_secret_dir,'%03d.png' % i))
                lstego_dir = os.path.join(stego_dir,'lr')
                lrec_dir = os.path.join(rec_dir,'lr')
                if not os.path.exists(lstego_dir):
                    os.makedirs(lstego_dir)
                if not os.path.exists(lrec_dir):
                    os.makedirs(lrec_dir)
                steg_imgl.save(os.path.join(lstego_dir,'%03d.png' % i))
                secret_revl.save(os.path.join(lrec_dir, '%03d.png' % i))

                noise_secret_img.save(os.path.join(noise_secret_dir,'%03d.png' % i))
                nstego_dir = os.path.join(stego_dir,'noise')
                nrec_dir = os.path.join(rec_dir,'noise')
                if not os.path.exists(nstego_dir):
                    os.makedirs(nstego_dir)
                if not os.path.exists(nrec_dir):
                    os.makedirs(nrec_dir)
                steg_imgn.save(os.path.join(nstego_dir,'%03d.png' % i))
                secret_revn.save(os.path.join(nrec_dir, '%03d.png' % i))


            i=i+1


    logger_val.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tPSNRC_N: {psnrc_n.avg:.6f} |"
        f"\tSSIMC_N: {ssimc_n.avg:.6f} |"
        f"\tPSNRS_N: {psnrs_n.avg:.6f} |" 
        f"\tSSIMS_N: {ssims_n.avg:.6f} |"
        f"\tPSNRC_B: {psnrc_b.avg:.6f} |"
        f"\tSSIMC_B: {ssimc_b.avg:.6f} |"
        f"\tPSNRS_B: {psnrs_b.avg:.6f} |" 
        f"\tSSIMS_B: {ssims_b.avg:.6f} |"
        f"\tPSNRC_L: {psnrc_l.avg:.6f} |"
        f"\tSSIMC_L: {ssimc_l.avg:.6f} |"
        f"\tPSNRS_L: {psnrs_l.avg:.6f} |" 
        f"\tSSIMS_L: {ssims_l.avg:.6f} |"
    )

    return 0

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        base_name = os.path.basename(filename)  
        dir_name = os.path.dirname(filename)   
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
        default=10000,
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
        default=(224, 224),
        help="Size of the training patches to be cropped (default: %(default)s)",
    )
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
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    ),
    parser.add_argument("--channel-in", type=int,default=12, help="channels into punet"),
    parser.add_argument("--num-steps_f", type=int, help="steps of LRH forward"),
    parser.add_argument("--num-steps_r", type=int, help="steps of LRH reverse"),
    parser.add_argument("--rec-weight", default = 1.0,type=float,help="weight of revealing loss"),
    parser.add_argument("--guide-weight", default = 1.0,type=float,help="weight of concealing loss")
    parser.add_argument("--freq-weight", default = 0.25,type=float,help="weight of frequency loss"),
    parser.add_argument("--data-type", default = [1,2,3], nargs='+', type=int),
    parser.add_argument("--val-freq", default = 30, type=int, help="how often should an evaluation be performed"),
    parser.add_argument("--sigma", default = 1.6,type=float),
    parser.add_argument(
        "--save-images", action="store_true", default=False, help="Save images to disk"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="test"
    )
    parser.add_argument("--nrate", default = 0.8,type=float, help="the ratio of noisy samples"),
    parser.add_argument("--lrate", default = 0.1,type=float, help="the ratio of lr samples"),
    parser.add_argument("--brate", default = 0.1,type=float, help="the ratio of blur samples"),
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
        [transforms.RandomCrop(args.patch_size),transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.test_patch_size),transforms.ToTensor()]
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

    # hide net
    lrh_f = LRH_f(args.num_steps_f)
    lrh_f = lrh_f.to(device)
    lrh_r = LRH_r(args.num_steps_r)
    lrh_r = lrh_r.to(device)

    logger_train.info(args)

    optimizer_f = configure_optimizers(lrh_f, args)
    optimizer_r = configure_optimizers(lrh_r, args)
    criterion = LRH_Loss()
    
    last_epoch = 0
    if args.checkpoint_f:  # load from previous checkpoint for lrh_f
        print("Loading", args.checkpoint_f)
        checkpoint_f = torch.load(args.checkpoint_f, map_location=device)
        last_epoch_f = checkpoint_f["epoch"] + 1
        lrh_f.load_state_dict(checkpoint_f["state_dict"])
        optimizer_f.load_state_dict(checkpoint_f["optimizer"])
        optimizer_f.param_groups[0]['lr'] = args.learning_rate

    if args.checkpoint_r:  # load from previous checkpoint for lrh_r
        print("Loading", args.checkpoint_r)
        checkpoint_r = torch.load(args.checkpoint_r, map_location=device)
        last_epoch_r = checkpoint_r["epoch"] + 1
        lrh_r.load_state_dict(checkpoint_r["state_dict"])
        optimizer_r.load_state_dict(checkpoint_r["optimizer"])
        optimizer_r.param_groups[0]['lr'] = args.learning_rate

    best_loss = float("inf")
    if not args.test:
        last_epoch = max(last_epoch_f if args.checkpoint_f else 0, last_epoch_r if args.checkpoint_r else 0)
        for epoch in range(last_epoch, args.epochs):
            logger_train.info(f"Learning rate (lrh_f): {optimizer_f.param_groups[0]['lr']}")
            logger_train.info(f"Learning rate (lrh_r): {optimizer_r.param_groups[0]['lr']}")
            loss = train_one_epoch(
                lrh_f, 
                lrh_r,
                criterion,
                train_dataloader,
                optimizer_f, 
                optimizer_r,
                epoch,
                logger_train,
                tb_logger,
                args
            )
            if epoch % args.val_freq == 0:
                test_epoch(args, epoch, test_dataloader, lrh_f, lrh_r, logger_val) 

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                # Save checkpoint for lrh_f
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
                # Save checkpoint for lrh_r
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
                    logger_val.info(f"Best checkpoints saved for both lrh_f and lrh_r in epoch {epoch}.")
    else:
        loss = test_epoch(args, 0, test_dataloader, lrh_f, lrh_r, logger_val)

if __name__ == "__main__":
    main(sys.argv[1:])

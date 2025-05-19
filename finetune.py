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
from losses import FinetuneLoss
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
from models.SPD import Network as SPD





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


def configure_optimizers(net, lr):

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    return optimizer


def train_one_epoch(LRH_f, LRH_r, SPD, criterion, train_dataloader, optimizer_f, optimizer_r, optimizer_spd, epoch, sigma_min, sigma_current_max, logger_train, tb_logger, args):
    device = next(LRH_f.parameters()).device
    dwt = DWT()
    iwt = IWT()

    if args.finetune == 1:
        LRH_f.eval()
        LRH_r.train()
        SPD.train()
    elif args.finetune == 2:
        LRH_f.train()
        LRH_r.train()
        SPD.eval()
    elif args.finetune == 3:
        LRH_f.eval()
        LRH_r.eval()
        SPD.train()
    elif args.finetune == 4:
        LRH_f.eval()
        LRH_r.train()
        SPD.eval()
    elif args.finetune == 5:
        LRH_f.train()
        LRH_r.train()
        SPD.train()
    else:
        raise ValueError("finetune must be 1, 2, 3, 4, 5")

    total_loss = 0.0
    num_batches = len(train_dataloader)

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        cover_img = d[d.shape[0] // 2:, :, :, :]
        secret_img = d[:d.shape[0] // 2, :, :, :]
        p = np.array([args.brate, args.nrate, args.lrate])
        type = np.random.choice(args.data_type, p=p.ravel())
        if type == 1:
            blur_secret_img = gauss_blur(secret_img, 2 * random.randint(0, 11) + 3, random.uniform(0.1, 2))
            input_secret_img = blur_secret_img
        elif type == 2:
            noiselvl = np.random.uniform(0, 25, size=1)
            noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=noiselvl[0] / 255.)
            noise_secret_img = secret_img + noise
            input_secret_img = noise_secret_img
        else:
            scalelvl = random.choice([2, 4])
            lr_secret_img = downsample(secret_img, scalelvl)
            input_secret_img = lr_secret_img

        input_cover = dwt(cover_img)
        input_secret = dwt(input_secret_img)

        # 清零梯度
        if optimizer_f:
            optimizer_f.zero_grad()
        if optimizer_r:
            optimizer_r.zero_grad()
        if optimizer_spd:
            optimizer_spd.zero_grad()

        # 隐藏阶段
        output_steg, output_z = LRH_f(input_cover, input_secret)
        steg_img = iwt(output_steg)

        # 攻击阶段
        if args.random:
            sigma = np.random.uniform(sigma_min, sigma_current_max, size=1)
            noise = torch.cuda.FloatTensor(steg_img.size()).normal_(mean=0, std=sigma[0]/255.0)
        else:
            noise = torch.cuda.FloatTensor(steg_img.size()).normal_(mean=0, std=sigma_current_max/255.0)
        noised_steg = steg_img + noise

        # 去噪阶段
        noise_level_est, denoised_steg = SPD(noised_steg)

        # 揭示阶段
        steg_img_dwt = dwt(denoised_steg)
        output_z_guass = gauss_noise(output_z.shape)
        cover_rev, secret_rev = LRH_r(steg_img_dwt, output_z_guass, rev=True)
        secret_rev = iwt(secret_rev)


        ps, _ = args.patch_size
        if args.random:
            gt_noise = torch.full((ps, ps), sigma[0]/255.0)
        else:
            gt_noise = torch.full((ps, ps), sigma_current_max/255.0)
        if_asym = torch.ones((3, ps, ps), device=device)
        steg_low = output_steg.narrow(1, 0, 3)
        cover_low = input_cover.narrow(1, 0, 3)
        out_criterian = criterion(
            args.loss_weights, input_secret_img, cover_img, steg_img, secret_rev, steg_low, cover_low,
            denoised_steg, noise_level_est, gt_noise, if_asym
        )
        loss = out_criterian['total_loss']
        guide_loss = out_criterian['guide_loss']
        rec_loss = out_criterian['rec_loss']
        freq_loss = out_criterian['freq_loss']
        denoise_loss = out_criterian['denoise_loss']
        wavelet_loss = out_criterian['wavelet_loss']

        # 反向传播和优化
        loss.backward()
        if args.finetune in [1, 2, 4, 5]:
            optimizer_r.step()
        if args.finetune in [2, 5]:
            optimizer_f.step()
        if args.finetune in [1, 3, 5]:
            optimizer_spd.step()

        total_loss += loss.item()

        if i % 10 == 0:
            if args.random:
                att_lvl_str = f"\tAttack level: {sigma[0]:.2f}/{sigma_current_max:.2f}"
            else:
                att_lvl_str = f"\tAttack level: {sigma_current_max:.2f}"

            logger_train.info(
                f"Train epoch {epoch} | {att_lvl_str}: "
                f"[{i * len(d)}/{len(train_dataloader.dataset)} "
                f"({100. * i / len(train_dataloader):.1f}%)]\n"
                f"\ttotal_loss: {loss.item():.2f} | "
                f"\tguide_loss: {guide_loss.item():.2f} | "
                f"\tfreq_loss: {freq_loss.item():.2f}\n"
                f"\trec_loss: {rec_loss.item():.2f} | "
                f"\tdenoise_loss: {denoise_loss.item():.2f} | "
                f"\twavelet_loss: {wavelet_loss.item():.2f}\n"
            )



    avg_loss = total_loss / num_batches
    logger_train.info(f"Train epoch {epoch}: Average loss: {avg_loss:.3f}")
    tb_logger.add_scalar('[train]: loss', avg_loss, epoch)

    return avg_loss



def test_epoch(args, epoch, test_dataloader, LRH_f, LRH_r, SPD, logger_val):
    dwt = DWT()
    iwt = IWT()
    LRH_f.eval()
    LRH_r.eval()
    SPD.eval()
    device = next(LRH_f.parameters()).device

    # 指标：秘密图像 vs 恢复图像
    psnr_n = AverageMeter()  
    ssims_n = AverageMeter()
    psnr_b = AverageMeter()  
    ssims_b = AverageMeter()
    psnr_l = AverageMeter()
    ssims_l = AverageMeter()

    # 指标：覆盖图像 vs 隐写图像
    psnr_n_stego = AverageMeter() 
    ssimc_n_stego = AverageMeter()
    psnr_b_stego = AverageMeter() 
    ssimc_b_stego = AverageMeter()
    psnr_l_stego = AverageMeter() 
    ssimc_l_stego = AverageMeter()

    # 指标：隐写图像 vs 去噪后隐写图像
    psnr_n_denoised = AverageMeter()  
    ssimc_n_denoised = AverageMeter()
    psnr_b_denoised = AverageMeter() 
    ssimc_b_denoised = AverageMeter()
    psnr_l_denoised = AverageMeter()
    ssimc_l_denoised = AverageMeter()

    sigma_total = 0
    i = 0
    with torch.no_grad():
        for d in tqdm(test_dataloader):
            d = d.to(device)
            cover_img = d[d.shape[0] // 2:, :, :, :]
            secret_img = d[:d.shape[0] // 2, :, :, :]

            # 三种破坏
            blur_secret_img = gauss_blur(secret_img, 15, args.sigma)
            noise = torch.cuda.FloatTensor(secret_img.size()).normal_(mean=0, std=15 / 255.)
            noise_secret_img = secret_img + noise
            lr_secret_img = downsample(secret_img, 4)

            # DWT 变换
            input_cover = dwt(cover_img)
            noise_secret = dwt(noise_secret_img)
            blur_secret = dwt(blur_secret_img)
            lr_secret = dwt(lr_secret_img)

            # 隐藏阶段
            output_stegn, output_z = LRH_f(input_cover, noise_secret)
            steg_imgn = iwt(output_stegn)
            output_stegb, output_z = LRH_f(input_cover, blur_secret)
            steg_imgb = iwt(output_stegb)
            output_stegl, output_z = LRH_f(input_cover, lr_secret)
            steg_imgl = iwt(output_stegl)

            # 攻击阶段
            sigma_min, sigma_max = args.attack_level
            sigma_current_max = sigma_min + (sigma_max - sigma_min) * (epoch / args.epochs)
            sigma = np.random.uniform(sigma_min, sigma_current_max, size=1)
            sigma_total += sigma[0]
            noise_n = torch.cuda.FloatTensor(steg_imgn.size()).normal_(mean=0, std=sigma[0]/255.0)
            noised_stegn = steg_imgn + noise_n
            noise_b = torch.cuda.FloatTensor(steg_imgb.size()).normal_(mean=0, std=sigma[0]/255.0)
            noised_stegb = steg_imgb + noise_b
            noise_l = torch.cuda.FloatTensor(steg_imgl.size()).normal_(mean=0, std=sigma[0]/255.0)
            noised_stegl = steg_imgl + noise_l

            # 去噪阶段
            noise_level_est_n, output_n = SPD(noised_stegn)
            denoised_stegn = output_n
            noise_level_est_b, output_b = SPD(noised_stegb)
            denoised_stegb = output_b
            noise_level_est_l, output_l = SPD(noised_stegl)
            denoised_stegl = output_l

            # 揭示阶段
            output_z_guass = gauss_noise(output_z.shape)
            denoised_stegn_dwt = dwt(denoised_stegn)
            denoised_stegb_dwt = dwt(denoised_stegb)
            denoised_stegl_dwt = dwt(denoised_stegl)
            cover_revn, secret_revn = LRH_r(denoised_stegn_dwt, output_z_guass, rev=True)
            secret_revn = iwt(secret_revn)
            cover_revb, secret_revb = LRH_r(denoised_stegb_dwt, output_z_guass, rev=True)
            secret_revb = iwt(secret_revb)
            cover_revl, secret_revl = LRH_r(denoised_stegl_dwt, output_z_guass, rev=True)
            secret_revl = iwt(secret_revl)

            # 保存图像
            save_dir = os.path.join('experiments', args.experiment, 'images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 转换为图像格式
            secret_img = torch2img(secret_img)
            cover_img = torch2img(cover_img)
            noise_secret_img = torch2img(noise_secret_img)
            blur_secret_img = torch2img(blur_secret_img)
            lr_secret_img = torch2img(lr_secret_img)
            steg_imgn = torch2img(steg_imgn)
            steg_imgb = torch2img(steg_imgb)
            steg_imgl = torch2img(steg_imgl)
            noised_steg_imgn = torch2img(noised_stegn)
            noised_steg_imgb = torch2img(noised_stegb)
            noised_steg_imgl = torch2img(noised_stegl)
            denoised_steg_imgn = torch2img(denoised_stegn)
            denoised_steg_imgb = torch2img(denoised_stegb)
            denoised_steg_imgl = torch2img(denoised_stegl)
            secret_revn = torch2img(secret_revn)
            secret_revb = torch2img(secret_revb)
            secret_revl = torch2img(secret_revl)

            # 计算指标：秘密图像 vs 恢复图像
            p1, m1 = compute_metrics(secret_revn, noise_secret_img)
            psnr_n.update(p1)
            ssims_n.update(m1)
            p2, m2 = compute_metrics(secret_revb, blur_secret_img)
            psnr_b.update(p2)
            ssims_b.update(m2)
            p3, m3 = compute_metrics(secret_revl, lr_secret_img)
            psnr_l.update(p3)
            ssims_l.update(m3)

            # 计算指标：覆盖图像 vs 隐写图像
            p4, m4 = compute_metrics(steg_imgn, cover_img)
            psnr_n_stego.update(p4)
            ssimc_n_stego.update(m4)
            p5, m5 = compute_metrics(steg_imgb, cover_img)
            psnr_b_stego.update(p5)
            ssimc_b_stego.update(m5)
            p6, m6 = compute_metrics(steg_imgl, cover_img)
            psnr_l_stego.update(p6)
            ssimc_l_stego.update(m6)

            # 计算指标：隐写图像 vs 去噪后隐写图像
            p7, m7 = compute_metrics(denoised_steg_imgn, steg_imgn)
            psnr_n_denoised.update(p7)
            ssimc_n_denoised.update(m7)
            p8, m8 = compute_metrics(denoised_steg_imgb, steg_imgb)
            psnr_b_denoised.update(p8)
            ssimc_b_denoised.update(m8)
            p9, m9 = compute_metrics(denoised_steg_imgl, steg_imgl)
            psnr_l_denoised.update(p9)
            ssimc_l_denoised.update(m9)

            # 保存图像
            if args.save_images:
                cover_dir = os.path.join(save_dir, 'cover')
                if not os.path.exists(cover_dir):
                    os.makedirs(cover_dir)
                stego_dir = os.path.join(save_dir, 'stego')
                if not os.path.exists(stego_dir):
                    os.makedirs(stego_dir)
                secret_dir = os.path.join(save_dir, 'secret')
                if not os.path.exists(secret_dir):
                    os.makedirs(secret_dir)
                noise_secret_dir = os.path.join(save_dir, 'secret_noise')
                if not os.path.exists(noise_secret_dir):
                    os.makedirs(noise_secret_dir)
                rec_dir = os.path.join(save_dir, 'recover')
                if not os.path.exists(rec_dir):
                    os.makedirs(rec_dir)
                lr_secret_dir = os.path.join(save_dir, 'secret_lr')
                if not os.path.exists(lr_secret_dir):
                    os.makedirs(lr_secret_dir)
                blur_secret_dir = os.path.join(save_dir, 'secret_blur')
                if not os.path.exists(blur_secret_dir):
                    os.makedirs(blur_secret_dir)
                attacked_dir = os.path.join(save_dir, 'attacked')
                if not os.path.exists(attacked_dir):
                    os.makedirs(attacked_dir)
                denoised_dir = os.path.join(save_dir, 'denoised')
                if not os.path.exists(denoised_dir):
                    os.makedirs(denoised_dir)

                # 保存 blur 类型图像
                blur_secret_img.save(os.path.join(blur_secret_dir, '%03d.png' % i))
                bstego_dir = os.path.join(stego_dir, 'blur')
                brec_dir = os.path.join(rec_dir, 'blur')
                battacked_dir = os.path.join(attacked_dir, 'blur')
                bdenoised_dir = os.path.join(denoised_dir, 'blur')
                if not os.path.exists(bstego_dir):
                    os.makedirs(bstego_dir)
                if not os.path.exists(brec_dir):
                    os.makedirs(brec_dir)
                if not os.path.exists(battacked_dir):
                    os.makedirs(battacked_dir)
                if not os.path.exists(bdenoised_dir):
                    os.makedirs(bdenoised_dir)
                steg_imgb.save(os.path.join(bstego_dir, '%03d.png' % i))
                secret_revb.save(os.path.join(brec_dir, '%03d.png' % i))
                noised_steg_imgb.save(os.path.join(battacked_dir, '%03d.png' % i))
                denoised_steg_imgb.save(os.path.join(bdenoised_dir, '%03d.png' % i))

                # 保存 lr 类型图像
                lr_secret_img.save(os.path.join(lr_secret_dir, '%03d.png' % i))
                lstego_dir = os.path.join(stego_dir, 'lr')
                lrec_dir = os.path.join(rec_dir, 'lr')
                lattacked_dir = os.path.join(attacked_dir, 'lr')
                ldenoised_dir = os.path.join(denoised_dir, 'lr')
                if not os.path.exists(lstego_dir):
                    os.makedirs(lstego_dir)
                if not os.path.exists(lrec_dir):
                    os.makedirs(lrec_dir)
                if not os.path.exists(lattacked_dir):
                    os.makedirs(lattacked_dir)
                if not os.path.exists(ldenoised_dir):
                    os.makedirs(ldenoised_dir)
                steg_imgl.save(os.path.join(lstego_dir, '%03d.png' % i))
                secret_revl.save(os.path.join(lrec_dir, '%03d.png' % i))
                noised_steg_imgl.save(os.path.join(lattacked_dir, '%03d.png' % i))
                denoised_steg_imgl.save(os.path.join(ldenoised_dir, '%03d.png' % i))

                # 保存 noise 类型图像
                noise_secret_img.save(os.path.join(noise_secret_dir, '%03d.png' % i))
                nstego_dir = os.path.join(stego_dir, 'noise')
                nrec_dir = os.path.join(rec_dir, 'noise')
                nattacked_dir = os.path.join(attacked_dir, 'noise')
                ndenoised_dir = os.path.join(denoised_dir, 'noise')
                if not os.path.exists(nstego_dir):
                    os.makedirs(nstego_dir)
                if not os.path.exists(nrec_dir):
                    os.makedirs(nrec_dir)
                if not os.path.exists(nattacked_dir):
                    os.makedirs(nattacked_dir)
                if not os.path.exists(ndenoised_dir):
                    os.makedirs(ndenoised_dir)
                steg_imgn.save(os.path.join(nstego_dir, '%03d.png' % i))
                secret_revn.save(os.path.join(nrec_dir, '%03d.png' % i))
                noised_steg_imgn.save(os.path.join(nattacked_dir, '%03d.png' % i))
                denoised_steg_imgn.save(os.path.join(ndenoised_dir, '%03d.png' % i))

            i += 1

    # 输出所有指标
    logger_val.info(
        f"Test epoch {epoch} | Attack level: {(sigma_total/i):.2f}/{sigma_current_max:.2f} | Average metrics:\n"
        f"Cover vs Stego:\n"
        f"\tPSNR_N: {psnr_n_stego.avg:.2f} | SSIM_N: {ssimc_n_stego.avg:.4f}\n"
        f"\tPSNR_B: {psnr_b_stego.avg:.2f} | SSIM_B: {ssimc_b_stego.avg:.4f}\n"
        f"\tPSNR_L: {psnr_l_stego.avg:.2f} | SSIM_L: {ssimc_l_stego.avg:.4f}\n"
        f"Stego vs Denoised Stego:\n"
        f"\tPSNR_N: {psnr_n_denoised.avg:.2f} | SSIM_N: {ssimc_n_denoised.avg:.4f}\n"
        f"\tPSNR_B: {psnr_b_denoised.avg:.2f} | SSIM_B: {ssimc_b_denoised.avg:.4f}\n"
        f"\tPSNR_L: {psnr_l_denoised.avg:.2f} | SSIM_L: {ssimc_l_denoised.avg:.4f}\n"
        f"Secret vs Recovered:\n"
        f"\tPSNR_N: {psnr_n.avg:.2f} | SSIM_N: {ssims_n.avg:.4f}\n"
        f"\tPSNR_B: {psnr_b.avg:.2f} | SSIM_B: {ssims_b.avg:.4f}\n"
        f"\tPSNR_L: {psnr_l.avg:.2f} | SSIM_L: {ssims_l.avg:.4f}\n"
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
        default=2000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        nargs=3,
        default=[0.0001, 0.0001, 0.0001],
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
    parser.add_argument("--checkpoint_spd", type=str, help="Path to SPD checkpoint"),
    parser.add_argument(
        "-exp", "--experiment", type=str, required=True, help="Experiment name"
    ),
    parser.add_argument("--channel-in", type=int,default=12, help="channels into punet"),
    parser.add_argument("--num-steps_f", type=int, help="steps of LRH forward"),
    parser.add_argument("--num-steps_r", type=int, help="steps of LRH reverse"),
    parser.add_argument("--data_type", default = [1,2,3], nargs='+', type=int),
    parser.add_argument("--val-freq", default = 50, type=int, help="how often should an evaluation be performed"),
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
    parser.add_argument("--loss_weights", type=float, nargs=4, default=[1, 0.25, 5, 1], help="guide_weight, freq_weight, rec_weight, denoise_weight"),
    parser.add_argument("--attack_level", type=float, nargs=2, default=[0.0, 25.0], help="Lower and upper bounds for sigma range in Gaussian noise attack")
    parser.add_argument(
        "--patience",
        default=150,
        type=int,
        help="Patience for early stopping (default: %(default)s)",
    ),
    parser.add_argument(
        "--finetune",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Finetune mode: 1 for LRH_r+SPD, 2 for LRH_f+LRH_r, 3 for SPD only, 4 for LRH_r only, 5 for all models)",
    ),
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="Randomize sigma for Gaussian noise attack",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))

    setup_logger('train', os.path.join('experiments', args.experiment), 'train_' + args.experiment,
                 level=logging.INFO, screen=True, tofile=True)
    setup_logger('val', os.path.join('experiments', args.experiment), 'val_' + args.experiment,
                 level=logging.INFO, screen=True, tofile=True)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # 初始化模型
    lrh_f = LRH_f(args.num_steps_f).to(device)
    lrh_r = LRH_r(args.num_steps_r).to(device)
    spd = SPD().to(device)

    # 根据 finetune 初始化优化器
    lr_f, lr_r, lr_spd = args.learning_rate
    if args.finetune == 1:
        optimizer_f = None
        optimizer_r = configure_optimizers(lrh_r, lr_r)
        optimizer_spd = torch.optim.Adam(spd.parameters(), lr=lr_spd)
    elif args.finetune == 2:
        optimizer_f = configure_optimizers(lrh_f, lr_f)
        optimizer_r = configure_optimizers(lrh_r, lr_r)
        optimizer_spd = None
    elif args.finetune == 3:
        optimizer_f = None
        optimizer_r = None
        optimizer_spd = torch.optim.Adam(spd.parameters(), lr=lr_spd)
    elif args.finetune == 4:
        optimizer_f = None
        optimizer_r = configure_optimizers(lrh_r, lr_r)
        optimizer_spd = None
    elif args.finetune == 5:
        optimizer_f = configure_optimizers(lrh_f, lr_f)
        optimizer_r = configure_optimizers(lrh_r, lr_r)
        optimizer_spd = torch.optim.Adam(spd.parameters(), lr=lr_spd)

    criterion = FinetuneLoss()

    logger_train.info(args)
    if args.finetune == 1:
        logger_train.info("Finetune mode 1: Training LRH_r and SPD")
    elif args.finetune == 2:
        logger_train.info("Finetune mode 2: Training LRH_f and LRH_r")
    elif args.finetune == 3:
        logger_train.info("Finetune mode 3: Training SPD only")
    elif args.finetune == 4:
        logger_train.info("Finetune mode 4: Training LRH_r only")
    elif args.finetune == 5:
        logger_train.info("Finetune mode 5: Training all models")

    best_loss = float("inf")
    last_improvement_epoch = -1

    # 加载检查点
    if args.checkpoint_f:
        print("Loading lrh_f from", args.checkpoint_f)
        checkpoint_f = torch.load(args.checkpoint_f, map_location=device)
        last_epoch_f = checkpoint_f["epoch"] + 1
        lrh_f.load_state_dict(checkpoint_f["state_dict"])
        if args.finetune == 2:
            optimizer_f.load_state_dict(checkpoint_f["optimizer"])
            optimizer_f.param_groups[0]['lr'] = lr_f

    if args.checkpoint_r:
        print("Loading lrh_r from", args.checkpoint_r)
        checkpoint_r = torch.load(args.checkpoint_r, map_location=device)
        last_epoch_r = checkpoint_r["epoch"] + 1
        lrh_r.load_state_dict(checkpoint_r["state_dict"])
        if args.finetune in [1, 2]:
            optimizer_r.load_state_dict(checkpoint_r["optimizer"])
            optimizer_r.param_groups[0]['lr'] = lr_r

    if args.checkpoint_spd:
        print("Loading SPD from", args.checkpoint_spd)
        checkpoint_spd = torch.load(args.checkpoint_spd, map_location=device)
        last_epoch_spd = checkpoint_spd["epoch"] + 1
        state_dict = checkpoint_spd["state_dict"]
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        spd.load_state_dict(new_state_dict)
        if args.finetune in [1, 3]:
            optimizer_spd.load_state_dict(checkpoint_spd["optimizer"])
            optimizer_spd.param_groups[0]['lr'] = lr_spd


    if(args.finetune == 1):
        last_epoch = max(last_epoch_r if args.checkpoint_r else 0, last_epoch_spd if args.checkpoint_spd else 0)
    elif(args.finetune == 2):
        last_epoch = max(last_epoch_f if args.checkpoint_f else 0, last_epoch_r if args.checkpoint_r else 0)
    elif(args.finetune == 3):
        last_epoch = last_epoch_spd if args.checkpoint_spd else 0
    elif(args.finetune == 4):
        last_epoch = last_epoch_r if args.checkpoint_r else 0
    elif(args.finetune == 5):
        last_epoch = max(last_epoch_f if args.checkpoint_f else 0, last_epoch_r if args.checkpoint_r else 0, last_epoch_spd if args.checkpoint_spd else 0)

    sigma_min, sigma_max = args.attack_level
    sigma_previous_max = 1e-6

    if not args.test:
        for epoch in range(last_epoch, args.epochs + 1):
            
            if optimizer_f:
                logger_train.info(f"Learning rate (lrh_f): {optimizer_f.param_groups[0]['lr']}")
            if optimizer_r:
                logger_train.info(f"Learning rate (lrh_r): {optimizer_r.param_groups[0]['lr']}")
            if optimizer_spd:
                logger_train.info(f"Learning rate (spd): {optimizer_spd.param_groups[0]['lr']}")
            
            sigma_current_max = sigma_min + (sigma_max - sigma_min) * (epoch / args.epochs)

            if sigma_current_max > sigma_previous_max:
                best_loss = best_loss * sigma_current_max / sigma_previous_max
                sigma_previous_max = sigma_current_max

            loss = train_one_epoch(
                lrh_f, lrh_r, spd, criterion, train_dataloader,
                optimizer_f, optimizer_r, optimizer_spd,
                epoch, sigma_min, sigma_current_max, logger_train, tb_logger, args
            )

            if (epoch+1) % args.val_freq == 0:
                test_epoch(args, epoch, test_dataloader, lrh_f, lrh_r, spd, logger_val)

            is_best = loss < best_loss
            if is_best:
                best_loss = loss
                last_improvement_epoch = epoch
            elif (epoch - last_improvement_epoch) % 10 == 0:
                logger_train.info(f"Patience warning: {epoch - last_improvement_epoch}/{args.patience} epochs without improvement.")
            if args.save:
                if args.finetune in [1, 2, 4, 5]:
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
                if args.finetune in [2, 5]:
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
                if args.finetune in [1, 3, 5]:
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": spd.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer_spd.state_dict(),
                        },
                        is_best,
                        os.path.join('experiments', args.experiment, 'checkpoints', "spd_checkpoint.pth.tar")
                    )
                if is_best:
                    logger_train.info(f"Best checkpoints saved, best loss: {best_loss:.4f}")

            if last_improvement_epoch != -1 and epoch - last_improvement_epoch >= args.patience:
                logger_train.info(f"No improvement in loss for {args.patience} epochs, early stopping triggered.")
                break
    else:
        test_epoch(args, args.epochs, test_dataloader, lrh_f, lrh_r, spd, logger_val)


if __name__ == "__main__":
    main(sys.argv[1:])
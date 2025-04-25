import torch
import torch.nn as nn
from util import DWT, IWT

class LRH_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
    def forward(self, secret, cover , stego ,rec, steg_low, cover_low,rec_weight,guide_weight,freq_weight):
        N, _, H, W = secret.size()
        out = {}
        guide_loss = self.mse(stego,cover)
        reconstruction_loss = self.mse(rec,secret)
        freq_loss = self.mse(steg_low,cover_low)
        hide_loss = rec_weight*reconstruction_loss  + freq_weight*freq_loss  +guide_weight*guide_loss
        out['g_loss'] = guide_loss
        out['r_loss'] = reconstruction_loss
        out['f_loss'] = freq_loss
        out['hide_loss'] = hide_loss
        return out

class LSR_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
        self.vgg_loss =  VGGLoss(3,1,True)
    def forward(self,secret_img,cover_img,steg_clean,steg_ori,rec_img,sweight,cweight,pweight,finetune):
        N, _, H, W = secret_img.size()
        out = {}
        lossc = self.mse(cover_img,steg_clean)
        lossc_ori = self.mse(cover_img,steg_ori)
        losss = self.mse(secret_img,rec_img)
        percep_lossc = self.vgg_loss(cover_img,steg_clean)
        percep_losss = self.vgg_loss(secret_img,rec_img)
        if finetune:
            loss =lossc*cweight + sweight*losss+ pweight*(percep_lossc+2*percep_losss)+lossc_ori*cweight
            out['pixel_loss'] = cweight*lossc + sweight*losss+lossc_ori*cweight
            out['percep_loss'] = pweight*(percep_lossc+2*percep_losss)
            out['loss'] = loss
        else:
            loss =lossc*cweight + sweight*losss+ pweight*(percep_lossc+2*percep_losss)
            out['pixel_loss'] = cweight*lossc + sweight*losss
            out['percep_loss'] = pweight*(percep_lossc+2*percep_losss)
            out['loss'] = loss
        return out

class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """
    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer.to('cuda:0'))
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1

        self.vgg_loss = nn.Sequential(*layers)
        self.criterion = torch.nn.MSELoss(reduce=True, size_average=False).to('cuda:0')

    def forward(self, source,target):
        return self.criterion(self.vgg_loss(source),self.vgg_loss(target))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
        self.dwt = DWT()  # 实例化小波变换
        # 确保 MSE 损失函数在正确的设备上
        self.to(device)
    
    def forward(self, loss_weights, secret, cover, stego, rec, steg_low, cover_low, denoised_steg, est_noise, gt_noise, if_asym):
        # 将所有输入张量移动到同一设备
        secret = secret.to(device)
        cover = cover.to(device)
        stego = stego.to(device)
        rec = rec.to(device)
        steg_low = steg_low.to(device)
        cover_low = cover_low.to(device)
        denoised_steg = denoised_steg.to(device)
        est_noise = est_noise.to(device)
        gt_noise = gt_noise.to(device)
        if_asym = if_asym.to(device)
        # 确保 loss_weights 中的权重是张量并移动到正确设备
        guide_weight, freq_weight, rec_weight, denoise_weight = [w.to(device) if isinstance(w, torch.Tensor) else torch.tensor(w, device=device) for w in loss_weights]

        N, _, H, W = secret.size()
        out = {}
        guide_loss = 0
        reconstruction_loss = 0
        freq_loss = 0
        denoise_loss = 0
        wavelet_loss = 0  # 新增小波高频子带损失

        # 原始损失
        guide_loss = guide_weight * self.mse(stego, cover)
        freq_loss = freq_weight * self.mse(steg_low, cover_low)
        reconstruction_loss = rec_weight * self.mse(rec, secret)

        # 去噪损失
        l2_loss = self.mse(denoised_steg, stego)
        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        # 小波高频子带损失
        # 对 stego 和 denoised_steg 进行小波分解
        stego_dwt = self.dwt(stego)  # 返回 (LL, (LH, HL, HH))
        denoised_dwt = self.dwt(denoised_steg)
        # 计算高频子带 (LH, HL, HH) 的 MSE
        for (stego_hf, denoised_hf) in zip(stego_dwt[1], denoised_dwt[1]):  # 遍历 LH, HL, HH
            wavelet_loss += self.mse(stego_hf, denoised_hf)
        wavelet_loss = wavelet_loss / 3  # 平均三个高频子带的损失

        # 调整去噪损失，移除 TV 损失（或减小权重），加入小波损失
        denoise_loss = denoise_weight * (l2_loss + 0.2 * asym_loss + wavelet_loss)  # 添加小波损失，权重可调

        # 总损失
        total_loss = reconstruction_loss + freq_loss + guide_loss + denoise_loss

        out['guide_loss'] = guide_loss
        out['rec_loss'] = reconstruction_loss
        out['freq_loss'] = freq_loss
        out['denoise_loss'] = denoise_loss
        out['wavelet_loss'] = wavelet_loss  
        out['total_loss'] = total_loss
        return out

    def _tensor_size(self, tensor):
        return tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
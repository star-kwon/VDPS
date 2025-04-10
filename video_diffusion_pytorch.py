import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import os

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

import numpy as np
from scipy import io
import random
import cv2

from torch.nn.functional import pad
from scipy.special import binom
import scipy

## Zernike forward model
def torch_fft(H):
    H = torch.fft.fftshift(torch.fft.fft2(H), dim=(-2, -1))
    return H

def torch_ifft(H):
    H = torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1)))
    return H

def torch_pad_center(H, pdim, xy: bool = False, padval=0):
    if xy:
        pxdim = pdim[0]
        pydim = pdim[1]
    else:
        pxdim = pdim[1]
        pydim = pdim[0]

    oxdim = H.shape[-1]
    oydim = H.shape[-2]

    oxcen = np.fix(oxdim / 2) + 1
    pxcen = np.fix(pxdim / 2) + 1
    lpx = pxcen - oxcen
    rpx = (pxdim - oxdim) - lpx

    oycen = np.fix(oydim / 2) + 1
    pycen = np.fix(pydim / 2) + 1
    lpy = pycen - oycen
    rpy = (pydim - oydim) - lpy

    return pad(H, pad=(int(lpx), int(rpx), int(lpy), int(rpy)), value=padval)

def torch_crop_center(H, dim, shift=(0, 0)):
    """
    Crops the center region of a tensor. Supports both square and rectangular crops.

    Args:
        H (torch.Tensor): Input tensor of shape (batch, channel, height, width).
        dim (int or tuple): Size of the crop. If int, a square crop is performed. If tuple, it specifies (height, width).
        shift (tuple): Optional shift (y, x) to apply to the center position.

    Returns:
        torch.Tensor: Cropped tensor.
    """
    batch, channel, Nh, Nw = H.size()

    if isinstance(dim, tuple):
        dim_h, dim_w = dim  # Rectangular crop
    else:
        dim_h, dim_w = dim, dim  # Square crop

    return H[:, :, (Nh - dim_h) // 2 + shift[0] : (Nh + dim_h) // 2 + shift[0], (Nw - dim_w) // 2 + shift[1] : (Nw + dim_w) // 2 + shift[1]]

def zernike_radial(n, m, r):
    """Compute the radial component of the Zernike polynomial."""
    R = torch.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        coeff = (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - abs(m)) // 2 - k)
        R += coeff * r ** (n - 2 * k)
    return R

def zernike_polynomial(n, m, rho, theta):
    """Compute the Zernike polynomial on a polar grid."""
    if m >= 0:
        return zernike_radial(n, m, rho) * torch.cos(m * theta)
    else:
        return zernike_radial(n, -m, rho) * torch.sin(-m * theta)

def generate_zernike_map(size, n_max=9):
    """Generate Zernike aberration maps up to order n_max."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    rho = torch.sqrt(X**2 + Y**2)
    theta = torch.atan2(Y, X)
    mask = rho <= 1  # Unit disk mask

    maps = {}

    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):  # m takes even values
            Z = torch.zeros_like(rho)
            Z[mask] = zernike_polynomial(n, m, rho[mask], theta[mask])
            maps[(n, m)] = Z

    return maps

def generate_amp_mask(size):
    """Generate Zernike aberration maps up to order n_max."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    rho = torch.sqrt(X**2 + Y**2) 
    mask = rho <= 1  # Unit disk mask

    return mask
 
def get_zernike_maps(zernike_maps, selected_modes):
    Z_list = []
    for n, m in selected_modes:
        Z = zernike_maps[(n, m)].numpy() 
        Z_list.append(torch.tensor(Z)) 

    Z_list = torch.stack(Z_list)  
    return Z_list

## Mapping network
class MappingNet_Zernike(nn.Module):
    def __init__(self):
        super(MappingNet_Zernike, self).__init__()

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 10)
        self.gelu = nn.ELU(alpha=0.5)

    def forward(self, input):
        return torch.tanh(self.fc2(self.gelu(self.fc1(input)))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * torch.pi

class MappingNet_Zernike_varying(nn.Module):
    def __init__(self):
        super(MappingNet_Zernike_varying, self).__init__()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 100)
        self.gelu = nn.ELU(alpha=0.5)

    def forward(self, input):
        return torch.tanh(self.fc2(self.gelu(self.fc1(input)))).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * torch.pi

class MappingNet(nn.Module):
    def __init__(self):
        super(MappingNet, self).__init__()

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 10)
        self.gelu = nn.ELU(alpha=0.5)

    def forward(self, input):
        return self.gelu(self.fc2(self.gelu(self.fc1(input))))

## Synthetic forward models
def make_gaussian_kernel(kernel_size, sigma):
    ts = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size).cuda()
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel

def fast_gaussian_blur(img: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    trailing_dims = img.shape[:-3]
    kernel_size = int(3*sigma)*2+1
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = make_gaussian_kernel(kernel_size, sigma).cuda()
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="constant", value=0)
    # Separable 2d conv
    kernel = kernel.view(*trailing_dims, 1, kernel_size, 1)
    img = F.conv2d(img, kernel)
    kernel = kernel.view(*trailing_dims, 1, 1, kernel_size)
    img = F.conv2d(img, kernel)

    return img

def generate_random_mask(shape, pixel_ratio):
    """
    Generates a random binary mask with the given pixel ratio.

    Args:
        shape (tuple): Shape of the mask (B, C, T, H, W).
        pixel_ratio (float): Ratio of pixels to be set to 1.

    Returns:
        torch.Tensor: Random binary mask.
    """
    B, C, T, H, W = shape
    num_pixels = H * W * T
    num_ones = int(num_pixels * pixel_ratio)
    
    # Generate a flat array with the appropriate ratio of ones and zeros
    flat_mask = torch.zeros(num_pixels, dtype=torch.float32)
    flat_mask[:num_ones] = 1
    
    # Shuffle to randomize the positions of ones and zeros
    flat_mask = flat_mask[torch.randperm(num_pixels)]
    
    # Reshape to the original spatial dimensions and duplicate across channels
    mask = flat_mask.view(1, T, H, W)
    mask = mask.expand(B, C, T, H, W)
    
    return mask

class dehazing_model(nn.Module):
    def __init__(self, beta=1.0, A=1.0):
        super(dehazing_model, self).__init__()
        self.beta = beta
        self.A = A

    def forward(self, I_c):
        transmission = torch.exp(torch.tensor(-self.beta))
        I_h = I_c * transmission + self.A * (1 - transmission)
        return I_h

class synthetic_forward_model(torch.nn.Module):
    def __init__(self, deg='blur', mask=None):
        super().__init__()

        self.deg = deg
        if mask is not None:
            self.mask = mask

    def forward(self, video):
        t = video.shape[2]
        video = unnormalize_img(video)
        video = video.clip(0, 1)

        list_frame = []
        for i in range(t):
            inten = video[:, :, i]
            if self.deg == 'blur':
                inten = fast_gaussian_blur(inten, 3.0)
            elif self.deg == 'inpaint':
                inten = inten * self.mask[:,:,i]
            elif self.deg == 'dehaze':
                inten = dehazing_model(beta=0.6, A=1.0)(inten)
            list_frame.append(inten.unsqueeze(0))

        forwarded_video = torch.cat(list_frame, axis=2)
        return forwarded_video

class forward_model(torch.nn.Module):
    def __init__(self, pix=8*1e-6, lamb=532e-9, dist = 10, sigma=4, size=64):
        super().__init__()

        image_length = size
        FOV = size
        pad = size // 2
        self.image_length = image_length
        self.pad=pad

        # propagation
        fx, fy = torch.meshgrid(torch.linspace(0, FOV + 2 * pad - 1, FOV + 2 * pad), torch.linspace(0, FOV + 2 * pad - 1, FOV + 2 * pad))

        fx = (fx - np.fix((FOV + 2 * pad) / 2)) / ((FOV + 2 * pad) * pix)
        fy = (fy - np.fix((FOV + 2 * pad) / 2)) / ((FOV + 2 * pad) * pix)

        quad_pha = 1 / (lamb ** 2) - fx ** 2 - fy ** 2

        self.prop = torch.exp(1j * 2 * torch.pi * dist * 1e-3 * torch.sqrt(quad_pha)).cuda()
        self.sigma=sigma

    def forward(self, video):
        b, c, t, h, w = video.size()
        video = unnormalize_img(video)
        video = video.clip(0, 1)
        pha = torch.zeros((h, w)).cuda()
        list_frame = []

        for i in range(t):
            amp = video[:, :, i]
            field_full = amp * torch.exp(1j * pha)
            field_pad = torch.nn.ReplicationPad2d(self.pad)(field_full).squeeze()
            field_fourier = torch.fft.fftshift(torch.fft.fft2(field_pad))

            ASM = field_fourier * self.prop

            diff_field = torch.fft.ifft2(torch.fft.ifftshift(ASM))
            diff_field = diff_field[self.pad:-self.pad, self.pad:-self.pad]

            inten_pad = abs(diff_field) ** 2
            m, n = inten_pad.shape

            inten = inten_pad[m // 2 - self.image_length // 2 :m // 2 + self.image_length // 2, n // 2 - self.image_length // 2: n // 2 + self.image_length // 2]

            inten = fast_gaussian_blur(inten.unsqueeze(0).unsqueeze(0), self.sigma)

            list_frame.append(inten.unsqueeze(0))

        forwarded_video = torch.cat(list_frame, axis=2)
        return forwarded_video

class forward_model_unknown(torch.nn.Module):
    def __init__(self, pix=8*1e-6, lamb=532e-9):
        super().__init__()

        image_length = 64
        FOV = 64
        pad = 32
        self.image_length = image_length
        self.pad=pad

        # propagation
        fx, fy = torch.meshgrid(torch.linspace(0, FOV + 2 * pad - 1, FOV + 2 * pad), torch.linspace(0, FOV + 2 * pad - 1, FOV + 2 * pad))

        fx = (fx - np.fix((FOV + 2 * pad) / 2)) / ((FOV + 2 * pad) * pix)
        fy = (fy - np.fix((FOV + 2 * pad) / 2)) / ((FOV + 2 * pad) * pix)

        self.quad_pha = 1 / (lamb ** 2) - fx ** 2 - fy ** 2

    def forward(self, video, dist, sigma):
       
        b, c, t, h, w = video.size()
        video = unnormalize_img(video)
        video = video.clip(0, 1)
        pha = torch.zeros((h, w)).cuda()
        list_frame = []

        for i in range(t):
            amp = video[:, :, i]
            field_full = amp * torch.exp(1j * pha)
            field_pad = torch.nn.ReplicationPad2d(self.pad)(field_full).squeeze()
            field_fourier = torch.fft.fftshift(torch.fft.fft2(field_pad))

            self.prop = torch.exp(1j * 2 * torch.pi * dist[i] * 1e-3 * torch.sqrt(self.quad_pha.cuda())).cuda()
            ASM = field_fourier * self.prop

            diff_field = torch.fft.ifft2(torch.fft.ifftshift(ASM))
            diff_field = diff_field[self.pad:-self.pad, self.pad:-self.pad]

            inten_pad = abs(diff_field) ** 2
            m, n = inten_pad.shape

            inten = inten_pad[m // 2 - self.image_length // 2 :m // 2 + self.image_length // 2, n // 2 - self.image_length // 2: n // 2 + self.image_length // 2]

            inten = fast_gaussian_blur(inten.unsqueeze(0).unsqueeze(0), sigma[i])

            list_frame.append(inten.unsqueeze(0))

        forwarded_video = torch.cat(list_frame, axis=2)

        return forwarded_video

class forward_model_unknown_real(torch.nn.Module):
    def __init__(self, pix=24*1e-6, lamb=532e-9, size=64):
        super().__init__()

        image_length = size
        FOV = size
        pad = size // 2
        self.image_length = image_length
        self.pad=pad

        # propagation
        fx, fy = torch.meshgrid(torch.linspace(0, FOV + 2 * pad - 1, FOV + 2 * pad), torch.linspace(0, FOV + 2 * pad - 1, FOV + 2 * pad))

        fx = (fx - np.fix((FOV + 2 * pad) / 2)) / ((FOV + 2 * pad) * pix)
        fy = (fy - np.fix((FOV + 2 * pad) / 2)) / ((FOV + 2 * pad) * pix)

        self.quad_pha = 1 / (lamb ** 2) - fx ** 2 - fy ** 2

    def forward(self, video, dist, sigma):
       
        b, c, t, h, w = video.size()
        video = unnormalize_img(video)
        video = video.clip(0, 1)
        pha = torch.zeros((h, w)).cuda()
        list_frame = []

        for i in range(t):
            amp = video[:, :, i]
            field_full = amp * torch.exp(1j * pha)
            field_pad = torch.nn.ReplicationPad2d(self.pad)(field_full).squeeze()
            field_fourier = torch.fft.fftshift(torch.fft.fft2(field_pad))

            self.prop = torch.exp(1j * 2 * torch.pi * dist[0] * torch.sqrt(self.quad_pha.cuda())).cuda()
            ASM = field_fourier * self.prop

            diff_field = torch.fft.ifft2(torch.fft.ifftshift(ASM))
            diff_field = diff_field[self.pad:-self.pad, self.pad:-self.pad]

            inten_pad = abs(diff_field) ** 2
            m, n = inten_pad.shape

            inten = inten_pad[m // 2 - self.image_length // 2 :m // 2 + self.image_length // 2, n // 2 - self.image_length // 2: n // 2 + self.image_length // 2]

            inten = fast_gaussian_blur(inten.unsqueeze(0).unsqueeze(0), sigma[i])

            list_frame.append(inten.unsqueeze(0))

        forwarded_video = torch.cat(list_frame, axis=2)

        return forwarded_video

class forward_model_unknown_zernike(torch.nn.Module):
    def __init__(self, size=64, device='cuda'):
        super().__init__()

        # Get zernike basis
        self.size = size
        self.zernike_maps = generate_zernike_map(size, n_max=4)
        self.device = device

        # Select a few Zernike modes to visualize  
        selected_modes = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)]
        self.zmap_basis = get_zernike_maps(self.zernike_maps, selected_modes).unsqueeze(1).to(device)

    def forward(self, video, zernike_basis):
        w_gt = zernike_basis
        pupil_phase = torch.sum(w_gt * self.zmap_basis, dim=0).unsqueeze(0)
        pupil_amp = generate_amp_mask(self.size).to(self.device) / self.size**2
        pupil = pupil_amp * torch.exp(1j * pupil_phase) 

        sim_dim = self.size * 2
        pupil_pad = torch_pad_center(pupil, (sim_dim, sim_dim)) 
        psf = torch_fft(pupil_pad).abs() ** 2 
        psf = torch_crop_center(psf, self.size)

        t = video.shape[2]
        video = unnormalize_img(video)
        video = video.clip(0, 1)
        list_frame = []

        for i in range(t):
            frame = video[:, :, i]
            blurred_frame = F.conv2d(frame, psf.to(self.device), padding="same")
            blurred_frame = torch_crop_center(blurred_frame.to(self.device), 64)

            list_frame.append(blurred_frame.unsqueeze(0))

        forwarded_video = torch.cat(list_frame, axis=2)

        return forwarded_video, psf

class forward_model_unknown_zernike_varying(torch.nn.Module):
    def __init__(self, size=64, device='cuda'):
        super().__init__()

        # Get zernike basis
        self.size = size
        self.zernike_maps = generate_zernike_map(size, n_max=4)
        self.device = device

        # Select a few Zernike modes to visualize  
        selected_modes = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)]
        self.zmap_basis = get_zernike_maps(self.zernike_maps, selected_modes).unsqueeze(1).to(device)

    def forward(self, video, zernike_basis):
        t = video.shape[2]

        video = unnormalize_img(video)
        video = video.clip(0, 1)
        list_frame = []
        psf_frame = []

        for i in range(t):
            frame = video[:, :, i]

            # Generate the PSF for each frame
            w_gt = zernike_basis[i*10:(i+1)*10]
            pupil_phase = torch.sum(w_gt * self.zmap_basis, dim=0).unsqueeze(0)
            pupil_amp = generate_amp_mask(self.size).to(self.device) / self.size**2
            pupil = pupil_amp * torch.exp(1j * pupil_phase) 

            sim_dim = self.size * 2
            pupil_pad = torch_pad_center(pupil, (sim_dim, sim_dim)) 
            psf = torch_fft(pupil_pad).abs() ** 2 
            psf = torch_crop_center(psf, self.size)

            # Apply the PSF to the frame           
            blurred_frame = F.conv2d(frame, psf.to(self.device), padding="same")
            blurred_frame = torch_crop_center(blurred_frame.to(self.device), 64)

            list_frame.append(blurred_frame.unsqueeze(0))
            psf_frame.append(psf)

        forwarded_video = torch.cat(list_frame, axis=2)
        psf = torch.cat(psf_frame, axis=1)

        return forwarded_video, psf

# helpers functions
def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias
        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model
class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        resnet_groups = 8
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        #cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        cond_dim = time_dim + int(cond_dim or 0)
       
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.2,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)
        r = x.clone()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.forward_model = forward_model()

        # register buffer helper function that casts float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale)
        x_recon = self.predict_start_from_noise(x, t=t, noise = noise)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon, noise

    #@torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon, noise = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond[0].shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond = cond, cond_scale = cond_scale)

    def synthetic_sample(self, measurement_video, deg = 'blur', mask = None, cond=None, cond_scale=1.):
        device = next(self.denoise_fn.parameters()).device
        b, c, t, h, w = measurement_video.shape
        target_video = measurement_video.detach()

        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)

        device = self.betas.device

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                      total=self.num_timesteps):
            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((1,), i, device=device, dtype=torch.long),
                                         cond=cond, cond_scale=cond_scale)
            
            if deg == 'inpaint':
                forwarded_recon = x_recon * mask
            else:
                forwarded_recon = synthetic_forward_model(deg=deg)(x_recon)

            diff = torch.linalg.norm(target_video - forwarded_recon) * math.sin(i/self.num_timesteps*(math.pi))**2

            guide = torch.autograd.grad(diff, video)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

        return unnormalize_img(out), forwarded_recon


    def physics_informed_sample(self, measurement_video, dist = 20, sigma=4, cond=None, cond_scale=1.):
        device = next(self.denoise_fn.parameters()).device
        b, c, t, h, w = measurement_video.shape
        target_video = measurement_video.detach()

        video = torch.randn([1, c, t, h, w], device=device)

        device = self.betas.device

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                      total=self.num_timesteps):
            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((1,), i, device=device, dtype=torch.long),
                                         cond=cond, cond_scale=cond_scale)

            forwarded_recon = forward_model(dist=dist, sigma=sigma, size=x_recon.shape[-1])(x_recon)

            diff = torch.linalg.norm(target_video - forwarded_recon) * math.sin(i/self.num_timesteps*(math.pi))**2

            guide = torch.autograd.grad(diff, video)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

        return unnormalize_img(out), forwarded_recon
    
    def physics_informed_sample_blind_zernike(self, measurement_video, cond=None, cond_scale=1.):
        initial_param = nn.Parameter(torch.zeros(256, requires_grad=True, device="cuda"))
        mapping_net = MappingNet_Zernike().cuda()
        optimizer = torch.optim.Adam(mapping_net.parameters(), lr=1e-2, betas=(0.9, 0.999))
        
        device = next(self.denoise_fn.parameters()).device
        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)
        
        b, c, t, h, w = video.shape
        target_video = measurement_video.detach()
        device = self.betas.device
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                    total=self.num_timesteps)
        for i in pbar:
            optimizer.zero_grad()

            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((b,), i, device=device, dtype=torch.long),
                                            cond=cond, cond_scale=cond_scale)

            zernike_basis = mapping_net(initial_param)

            forwarded_recon, psf = forward_model_unknown_zernike(size=32)(x_recon, zernike_basis)

            diff = torch.linalg.norm(target_video - forwarded_recon) / 10

            guide = torch.autograd.grad(diff, video, retain_graph=True)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

            loss = torch.mean((target_video - forwarded_recon)**2)
            loss.backward(retain_graph=True)

            optimizer.step()

        return unnormalize_img(out), zernike_basis, psf

    def physics_informed_sample_blind_zernike_varying(self, measurement_video, cond=None, cond_scale=1.):
        initial_param = nn.Parameter(torch.zeros(128, requires_grad=True, device="cuda"))
        mapping_net = MappingNet_Zernike_varying().cuda()
        optimizer = torch.optim.Adam(mapping_net.parameters(), lr=1e-2, betas=(0.9, 0.999))
        
        device = next(self.denoise_fn.parameters()).device
        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)
        
        b, c, t, h, w = video.shape
        target_video = measurement_video.detach()
        device = self.betas.device
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                    total=self.num_timesteps)
        for i in pbar:
            optimizer.zero_grad()

            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((b,), i, device=device, dtype=torch.long),
                                            cond=cond, cond_scale=cond_scale)

            zernike_basis = mapping_net(initial_param)

            forwarded_recon, psf = forward_model_unknown_zernike_varying(size=32)(x_recon, zernike_basis)

            diff = torch.linalg.norm(target_video - forwarded_recon) / 10

            guide = torch.autograd.grad(diff, video, retain_graph=True)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

            loss = torch.mean((target_video - forwarded_recon)**2)
            loss.backward(retain_graph=True)

            optimizer.step()

        return unnormalize_img(out), zernike_basis, psf

    def physics_informed_sample_unknown_sigma(self, measurement_video, dist=[], cond=None, cond_scale=1.):
        initial_param = nn.Parameter(torch.zeros(128, requires_grad=True, device="cuda"))
        mapping_net = MappingNet().cuda()
        optimizer = torch.optim.Adam(mapping_net.parameters(), lr=1e-3, betas=(0.9,0.999))
        
        device = next(self.denoise_fn.parameters()).device
        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)

        b, c, t, h, w = video.shape
        target_video = measurement_video.detach()

        device = self.betas.device

        pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                    total=self.num_timesteps)
        for i in pbar:
            optimizer.zero_grad()

            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((b,), i, device=device, dtype=torch.long),
                                            cond=cond, cond_scale=cond_scale)

            sigma = mapping_net(initial_param) + 1.0

            forwarded_recon = forward_model_unknown()(x_recon, dist, sigma)

            diff = torch.linalg.norm(target_video - forwarded_recon) / 5

            guide = torch.autograd.grad(diff, video, retain_graph=True)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

            loss = torch.mean((target_video - forwarded_recon)**2)
            loss.backward(retain_graph=True)

            optimizer.step()

            pbar.set_postfix({'1': float(sigma[0]), '2': float(sigma[1]),
                                '3': float(sigma[2]), '4': float(sigma[3]),
                                '5': float(sigma[4]), '6': float(sigma[5]),
                                '7': float(sigma[6]), '8': float(sigma[7]),
                                '9': float(sigma[8]), '10': float(sigma[9]), },refresh=False)

        return unnormalize_img(out), forwarded_recon, sigma
    
    def physics_informed_sample_unknown_dist(self, measurement_video, sigma=[], cond=None, cond_scale=1.):

        initial_param = nn.Parameter(torch.zeros(128, requires_grad=True, device="cuda"))
        mapping_net = MappingNet().cuda()
        optimizer = torch.optim.Adam(mapping_net.parameters(), lr=1e-3, betas=(0.9,0.999))

        device = next(self.denoise_fn.parameters()).device
        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)

        b, c, t, h, w = video.shape
        target_video = measurement_video.detach()

        device = self.betas.device

        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                    total=self.num_timesteps)
        for i in pbar:
            optimizer.zero_grad()

            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((b,), i, device=device, dtype=torch.long),
                                            cond=cond, cond_scale=cond_scale)

            dist = mapping_net(initial_param) + 2.5

            forwarded_recon = forward_model_unknown()(x_recon, dist, sigma)

            diff = torch.linalg.norm(target_video - forwarded_recon) / 5

            guide = torch.autograd.grad(diff, video, retain_graph=True)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

            loss = torch.mean((target_video - forwarded_recon)**2)
            loss.backward(retain_graph=True)

            optimizer.step()

            pbar.set_postfix({'1': float(dist[0]), '2': float(dist[1]),
                                '3': float(dist[2]), '4': float(dist[3]),
                                '5': float(dist[4]), '6': float(dist[5]),
                                '7': float(dist[6]), '8': float(dist[7]),
                                '9': float(dist[8]), '10': float(dist[9]), },refresh=False)

        return unnormalize_img(out), forwarded_recon, dist
    
    def physics_informed_sample_unknown_full(self, measurement_video, cond=None, cond_scale=1.):

        initial_param = nn.Parameter(torch.zeros(128, requires_grad=True, device="cuda"))

        my_nn_1 = MappingNet().cuda()
        my_nn_2 = MappingNet().cuda()

        optimizer_1 = torch.optim.Adam(my_nn_1.parameters(), lr=1e-3, betas=(0.9,0.999))
        optimizer_2 = torch.optim.Adam(my_nn_2.parameters(), lr=1e-3, betas=(0.9,0.999))

        device = next(self.denoise_fn.parameters()).device

        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)

        b, c, t, h, w = video.shape
        target_video = measurement_video.detach()

        device = self.betas.device

        pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                    total=self.num_timesteps)
        for i in pbar:
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((b,), i, device=device, dtype=torch.long),
                                            cond=cond, cond_scale=cond_scale)

            sigma = my_nn_1(initial_param) + 0.5
            dist = my_nn_2(initial_param) + 2.5

            forwarded_recon = forward_model_unknown()(x_recon, dist, sigma)

            diff = torch.linalg.norm(target_video - forwarded_recon) / 5

            guide = torch.autograd.grad(diff, video, retain_graph=True)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

            loss = torch.mean((target_video - forwarded_recon)**2)
            loss.backward(retain_graph=True)

            optimizer_1.step()
            optimizer_2.step()

            pbar.set_postfix({'1': float(sigma[0]), '2': float(sigma[1]),
                                '3': float(sigma[2]), '4': float(sigma[3]),
                                '5': float(sigma[4]), '6': float(sigma[5]),
                                '7': float(sigma[6]), '8': float(sigma[7]),
                                '9': float(sigma[8]), '10': float(sigma[9]), },refresh=False)

        return unnormalize_img(out), forwarded_recon, sigma, dist
    
    def physics_informed_sample_unknown_real(self, measurement_video, dist, cond=None, cond_scale=1.):
        initial_param = nn.Parameter(torch.zeros(128, requires_grad=True, device="cuda"))
        mapping_net = MappingNet().cuda()
        optimizer = torch.optim.Adam(mapping_net.parameters(), lr=1e-3, betas=(0.9,0.999))
        
        device = next(self.denoise_fn.parameters()).device
        video = torch.randn([1, self.channels, self.num_frames, self.image_size, self.image_size], device=device)

        b, c, t, h, w = video.shape
        target_video = measurement_video.detach()

        device = self.betas.device

        pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                    total=self.num_timesteps)
        for i in pbar:
            optimizer.zero_grad()

            video.requires_grad_(True)
            target_video.requires_grad_(True)

            out, x_recon = self.p_sample(video, torch.full((b,), i, device=device, dtype=torch.long),
                                            cond=cond, cond_scale=cond_scale)

            sigma = mapping_net(initial_param) + 1.0

            forwarded_recon = forward_model_unknown_real(size=x_recon.shape[-1])(x_recon, dist, sigma)

            diff = torch.linalg.norm(target_video - forwarded_recon) / 5

            guide = torch.autograd.grad(diff, video, retain_graph=True)[0]
            video = out - guide

            video = video.detach_()
            target_video = target_video.detach_()

            loss = torch.mean((target_video - forwarded_recon)**2)
            loss.backward(retain_graph=True)

            optimizer.step()

        return unnormalize_img(out), forwarded_recon
    
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)

# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

def custom_loader(path):
    image = io.loadmat(path)
    image = image['output']
    cube = image.astype(np.float32)

    return cube

class Dataset_UCF(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 1,
        num_frames = 16,
        frame_skip = 1,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['avi', 'mp4'],
        train = True
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels

        if train:
            self.paths = sorted([p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')])
        else:
            self.paths = sorted([p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')])

        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = mp4_to_tensor(path, self.image_size, self.channels, self.num_frames, self.frame_skip)
        folder = str(path).split('/')[-2]
        return self.cast_num_frames_fn(tensor), folder
    
class Dataset_DAVIS(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=1,
        num_frames=16,
        frame_skip=1,
        force_num_frames=True,
        train=True
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.train = train
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.paths = sorted([p for p in Path(folder).iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        seq_folder = self.paths[index]
        tensor = folder_to_tensor(seq_folder, image_size=self.image_size, num_frames=self.num_frames)
        seq_name = seq_folder.name
        return self.cast_num_frames_fn(tensor), seq_name

class Dataset_VISEM(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 1,
        num_frames = 16,
        frame_skip = 1,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['avi', 'mp4'],
        train = True
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels

        if train:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        else:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = mp4_to_tensor_centercrop(path, self.image_size, self.channels, self.num_frames, self.frame_skip)
        folder = str(path).split('/')[-1]
        folder = folder.split('.')[0]
        return self.cast_num_frames_fn(tensor), folder

##for review process and real application
class Dataset_Test_Processed(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 1,
        num_frames = 16,
        frame_skip = 1,
        force_num_frames = True,
        train = True
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        if train == True:
            self.paths = [p for p in sorted(Path(f'{folder}').glob('*/*/*'))]
        else:
            self.paths = [p for p in sorted(Path(f'{folder}').glob('*/*/*'))]
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.sigma_list = {'100um': 0.89453, '550um': 1.0196, '900um': 1.2504}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        sigma = self.sigma_list[str(path).split('/')[-3].split('_')[-1]]
        tensor, distance = mat_folder_to_tensor(path, self.image_size, self.channels, self.num_frames, self.frame_skip)
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()

        path_list = str(path).split('/')[-1]

        dc = 0.1
        tensor = tensor - dc
        tensor[tensor<0]=0
        tensor = tensor

        return tensor, distance, path_list, sigma

def mp4_to_tensor(path, image_size=64, channels=3, num_frames=10, frame_skip = 1):

    cap = cv2.VideoCapture(str(path))
    frame_list = []

    while True:
        ret, frame = cap.read()
        if (ret == True):
            resized_frame = cv2.resize(frame, (image_size, image_size))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            frame_list.append(torch.from_numpy(resized_frame).unsqueeze(0).unsqueeze(0)/255.0)
        else:
            break

    cap.release()

    while len(frame_list)<num_frames*frame_skip:
       frame_list = frame_list + frame_list[::-1][1:]

    video = torch.cat(frame_list, dim=1)

    t = video.shape[1]

    frame_start = random.randint(0, t - (num_frames-1)*frame_skip - 1)

    tensor = video[:, frame_start:frame_start+(num_frames-1)*frame_skip+1:frame_skip, :, :]

    return tensor

def mat_folder_to_tensor(path, image_size=64, channels=3, num_frames=10, frame_skip = 1):#, channels = 3, transform = T.ToTensor()):

    img_list = sorted([str(p) for p in Path(f'{path}').glob(f'*.mat')])

    frame_list = []

    for i in range(num_frames):
        frame = scipy.io.loadmat(img_list[i])
        distance = frame['z']
        frame = frame['holo_roi_crop']
        frame = frame[np.newaxis, np.newaxis, :, :]
        frame_list.append(torch.from_numpy(frame))

    video = torch.cat(frame_list, dim=1)
    tensor = video

    return tensor, distance

def folder_to_tensor(folder_path, image_size=64, num_frames=16):
    frame_list = []

    for i in range(num_frames):
        filename = os.path.join(folder_path, f"{i:05d}.jpg")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found.")

        # Load image
        img = Image.open(filename).convert('L')  # Convert to grayscale ('L' mode)

        # Center crop
        w, h = img.size
        min_side = min(w, h)
        left = (w - min_side) // 2
        top = (h - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        img = img.crop((left, top, right, bottom))

        # Resize to image_size x image_size
        img = img.resize((image_size, image_size), Image.BICUBIC)

        # To tensor (shape: 1, 1, H, W)
        frame_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).unsqueeze(0) / 255.0
        frame_list.append(frame_tensor)

    # Concatenate along time dimension -> (1, T, H, W)
    video_tensor = torch.cat(frame_list, dim=1)

    return video_tensor  # shape: (1, 16, 64, 64)

def mp4_to_tensor_centercrop(path, image_size=64, channels=3, num_frames=10, frame_skip = 1):

    cap = cv2.VideoCapture(str(path))
    frame_list = []

    while True:
        ret, frame = cap.read()
        if (ret == True):
            h, w, _ = frame.shape
            frame = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

            h, w, _ = frame.shape
            top = (h - image_size) // 2
            left = (w - image_size) // 2
            cropped_frame = frame[top:top + image_size, left:left + image_size]
            resized_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            frame_list.append(torch.from_numpy(resized_frame).unsqueeze(0).unsqueeze(0)/255.0)
        else:
            break

    cap.release()

    while len(frame_list)<num_frames*frame_skip:
       frame_list = frame_list + frame_list[::-1][1:]

    video = torch.cat(frame_list, dim=1)

    t = video.shape[1]

    frame_start = random.randint(0, t - (num_frames-1)*frame_skip - 1)

    tensor = video[:, frame_start:frame_start+(num_frames-1)*frame_skip+1:frame_skip, :, :]

    return tensor

##for real experiment
def real_folder_to_tensor(path, image_size=64, channels=3, num_frames=10, frame_skip = 1):

    img_list = sorted([str(p) for p in Path(f'{path}').glob(f'*.mat')])

    frame_list = []

    for i in range(num_frames):

        frame = io.loadmat(img_list[i])
        distance = frame['z'][0]
        frame = frame['holo_roi_crop']

        frame = frame[np.newaxis, np.newaxis, :, :]
        frame_list.append(torch.from_numpy(frame))

    video = torch.cat(frame_list, dim=1)

    tensor = video
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor[tensor<0] = 0

    return tensor, distance

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 2,
        max_grad_norm = None,
        dataset='UCF101'
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        if dataset == 'UCF101':
            self.ds = Dataset_UCF(folder, image_size, channels = channels, num_frames = num_frames, train=True)
            self.ds_test = Dataset_UCF(folder, image_size, channels = channels, num_frames = num_frames, train=False)
        elif dataset == 'DAVIS':
            self.ds = Dataset_DAVIS(folder, image_size, channels = channels, num_frames = num_frames, train=True)
            self.ds_test = Dataset_DAVIS(folder, image_size, channels = channels, num_frames = num_frames, train=False)
        elif dataset == 'VISEM':
            self.ds = Dataset_VISEM(folder, image_size, channels = channels, num_frames = num_frames, train=True)
            self.ds_test = Dataset_VISEM(folder, image_size, channels = channels, num_frames = num_frames, train=False)
        elif dataset == 'Real':
            self.ds = Dataset_Test_Processed(folder, image_size, channels = channels, num_frames = num_frames, train=True)
            self.ds_test = Dataset_Test_Processed(folder, image_size, channels = channels, num_frames = num_frames, train=False)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.dl_test = cycle(data.DataLoader(self.ds_test, batch_size = 1, shuffle=False, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            log = {'loss': loss.item()}
            print(loss.item())

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                data = next(self.dl_test).cuda()
                sample = self.ema_model.sample(batch_size=1)

                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(sample.squeeze(0), video_path)

                video_path = str(self.results_folder / str(f'{milestone}_target.gif'))
                video_tensor_to_gif(data.squeeze(0), video_path)

                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')

    def eval_synthetic(self, deg='blur'):
        for sample in range(len(self.ds_test)):
            data, folder_name = next(self.dl_test)
            target = data.cuda()

            if deg == 'inpaint':
                mask = generate_random_mask(target.shape, 0.5).cuda()
                measurement = normalize_img(target) * mask
            else:
                measurement = synthetic_forward_model(deg = deg)(normalize_img(target))

            target_list = []
            measurement_list = []
            output_list = []
            forward_output_list = []
            if deg == 'inpaint':
                output, forward_output = self.ema_model.synthetic_sample(measurement, deg = deg, mask=mask)
            else:
                output, forward_output = self.ema_model.synthetic_sample(measurement, deg = deg)

            folder_name = folder_name[0]

            output_path = self.results_folder / 'results' / deg
            os.makedirs(output_path, exist_ok=True)

            target_list.append(target.squeeze(0))
            measurement_list.append(measurement.squeeze(0))
            output_list.append(output[0])
            forward_output_list.append(forward_output.squeeze(0))

            target_list = torch.cat(target_list, dim=0)
            forward_output_list = torch.cat(forward_output_list, dim=0)
            measurement_list = torch.cat(measurement_list, dim=0)
            measurement_list = torch.clamp(measurement_list, 0, 1)
            output_list = torch.cat(output_list, dim=0)

            video_path = str(output_path / f'{folder_name}_target.gif')
            video_tensor_to_gif(target_list, video_path)

            video_path = str(output_path / f'{folder_name}_measurement.gif')
            video_tensor_to_gif(measurement_list, video_path)

            video_path = str(output_path / f'{folder_name}_output.gif')
            video_tensor_to_gif(output_list, video_path)

        print('Synthetic reconstruction completed')

    def eval_physics_informed(self, dist = 10, sigma=2):

        for sample in range(len(self.ds_test)):
            data, folder_name = next(self.dl_test)
            target = data.cuda()

            measurement = forward_model(dist = dist, sigma = sigma, size=target.shape[-1])(normalize_img(target))

            target_list = []
            measurement_list = []
            output_list = []
            forward_output_list = []
            output, forward_output = self.ema_model.physics_informed_sample(measurement, dist = dist, sigma = sigma)

            folder_name = folder_name[0]

            output_path = self.results_folder / 'results' / 'physical_forward' /  str(dist) / str(sigma)
            os.makedirs(output_path, exist_ok=True)

            target_list.append(target.squeeze(0))
            measurement_list.append(measurement.squeeze(0))
            output_list.append(output[0])
            forward_output_list.append(forward_output.squeeze(0))

            target_list = torch.cat(target_list, dim=0)
            forward_output_list = torch.cat(forward_output_list, dim=0)
            measurement_list = torch.cat(measurement_list, dim=0)
            measurement_list = torch.clamp(measurement_list, 0, 1)
            output_list = torch.cat(output_list, dim=0)

            video_path = str(output_path / f'{folder_name}_target.gif')
            video_tensor_to_gif(target_list, video_path)

            video_path = str(output_path / f'{folder_name}_measurement.gif')
            video_tensor_to_gif(measurement_list, video_path)

            video_path = str(output_path / f'{folder_name}_output.gif')
            video_tensor_to_gif(output_list, video_path)

        print('Physics informed reconstruction completed')


    def eval_physics_informed_blind(self, dist_low = 2.5, dist_high = 10, sigma_low=0.5, sigma_high=2):
        milestone = self.step // self.save_and_sample_every

        for sample in range(len(self.ds_test)):
            data, folder_name = next(self.dl_test)
            target = data.cuda()

            sigma_list = [random.uniform(sigma_low, sigma_high) for _ in range(10)]
            dist_list = [random.uniform(dist_low, dist_high) for _ in range(10)]
            measurement = forward_model_unknown()(normalize_img(target), dist = dist_list, sigma = sigma_list)
            for i in range(1):
                target_list = []
                measurement_list = []
                output_list = []
                forward_output_list = []

                if dist_low == dist_high:
                    output, forward_output, _ = self.ema_model.physics_informed_sample_unknown_sigma(measurement, dist = dist_list)
                elif sigma_low == sigma_high:
                    output, forward_output, _ = self.ema_model.physics_informed_sample_unknown_dist(measurement, sigma = sigma_list)
                else:
                    output, forward_output, _, _ = self.ema_model.physics_informed_sample_unknown_full(measurement)

            folder_name = folder_name[0]

            output_path = self.results_folder / 'results' / 'physical_forward_blind' / f'{dist_low},{dist_high}' / f'{sigma_low},{sigma_high}'
            os.makedirs(output_path, exist_ok=True)

            target_list.append(target.squeeze(0))
            measurement_list.append(measurement.squeeze(0))
            output_list.append(output[0])
            forward_output_list.append(forward_output.squeeze(0))

            target_list = torch.cat(target_list, dim=0)
            forward_output_list = torch.cat(forward_output_list, dim=0)
            measurement_list = torch.cat(measurement_list, dim=0)
            measurement_list = torch.clamp(measurement_list, 0, 1)
            output_list = torch.cat(output_list, dim=0)

            video_path = str(output_path / f'{folder_name}_target.gif')
            video_tensor_to_gif(target_list, video_path)

            video_path = str(output_path / f'{folder_name}_measurement.gif')
            video_tensor_to_gif(measurement_list, video_path)

            video_path = str(output_path / f'{folder_name}_output.gif')
            video_tensor_to_gif(output_list, video_path)

        print('Physics informed blind sampling completed')
    
    def eval_zernike_blind(self):
        for sample in range(len(self.ds_test)):
            data, folder_name = next(self.dl_test)
            target = data.cuda()

            zernike_basis = torch.randn(10, 1, 1, 1).cuda() * 2
            
            measurement, psf_gt = forward_model_unknown_zernike(size=32)(normalize_img(target), zernike_basis=zernike_basis)
            for i in range(1):
                target_list = []
                measurement_list = []
                output_list = []

                output, _, psf = self.ema_model.physics_informed_sample_blind_zernike(measurement)
            
            folder_name = folder_name[0]

            output_path = self.results_folder / 'zernike_results'
            os.makedirs(output_path, exist_ok=True)

            target_list.append(target.squeeze(0))
            measurement_list.append(measurement.squeeze(0))
            output_list.append(output[0])

            target_list = torch.cat(target_list, dim=0)
            measurement_list = torch.cat(measurement_list, dim=0)
            measurement_list = measurement_list - measurement_list.min() 
            measurement_list = measurement_list / measurement_list.max()
            output_list = torch.cat(output_list, dim=0)

            video_path = str(output_path / f'{folder_name}_target.gif')
            video_tensor_to_gif(target_list, video_path)

            video_path = str(output_path / f'{folder_name}_measurement.gif')
            video_tensor_to_gif(measurement_list, video_path)

            video_path = str(output_path / f'{folder_name}_output.gif')
            video_tensor_to_gif(output_list, video_path)

            psf_path = str(output_path / f'{folder_name}_psf_gt.png')
            psf_gt = psf_gt[0, 0, :, :].detach().cpu().numpy()
            psf_gt = (psf_gt - psf_gt.min()) / (psf_gt.max() - psf_gt.min())
            psf_gt = (psf_gt * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(psf_gt, mode='L')
            image.save(psf_path)

            psf_path = str(output_path / f'{folder_name}_psf_estimate.png')
            psf = psf[0, 0, :, :].detach().cpu().numpy()
            psf = (psf - psf.min()) / (psf.max() - psf.min())
            psf = (psf * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(psf, mode='L')
            image.save(psf_path)

        print('Zernike PSF blind reconstruction completed')

    def eval_zernike_blind_varying(self):
        for sample in range(len(self.ds_test)):
            data, folder_name = next(self.dl_test)
            target = data.cuda()

            zernike_basis = torch.randn(100, 1, 1, 1).cuda() * 2
            
            measurement, psf_gt = forward_model_unknown_zernike_varying(size=32)(normalize_img(target), zernike_basis=zernike_basis)
            for i in range(1):
                target_list = []
                measurement_list = []
                output_list = []

                output, _, psf = self.ema_model.physics_informed_sample_blind_zernike_varying(measurement)
            
            folder_name = folder_name[0]

            output_path = self.results_folder / 'zernike_varying_results'
            os.makedirs(output_path, exist_ok=True)

            target_list.append(target.squeeze(0))
            measurement_list.append(measurement.squeeze(0))
            output_list.append(output[0])

            target_list = torch.cat(target_list, dim=0)
            measurement_list = torch.cat(measurement_list, dim=0)
            measurement_list = measurement_list - measurement_list.min() 
            measurement_list = measurement_list / measurement_list.max()
            output_list = torch.cat(output_list, dim=0)

            video_path = str(output_path / f'{folder_name}_target.gif')
            video_tensor_to_gif(target_list, video_path)

            video_path = str(output_path / f'{folder_name}_measurement.gif')
            video_tensor_to_gif(measurement_list, video_path)

            video_path = str(output_path / f'{folder_name}_output.gif')
            video_tensor_to_gif(output_list, video_path)

            psf_gt = (psf_gt - psf_gt.min()) / (psf_gt.max() - psf_gt.min())
            video_path = str(output_path / f'{folder_name}_psf_gt.gif')
            video_tensor_to_gif(psf_gt, video_path)
            
            psf = (psf - psf.min()) / (psf.max() - psf.min())
            video_path = str(output_path / f'{folder_name}_psf_estimate.gif')
            video_tensor_to_gif(psf, video_path)

        print('Zernike PSF varying blind reconstruction completed')

    def eval_physics_informed_real(self):
        for sample in range(len(self.ds_test)):
            data = next(self.dl_test)

            ##for real measurement
            measurement, distance, path_list, sigma = data
            input_path = path_list[0]
            measurement = measurement.float().cuda()
            distance = distance.float().cuda()
            sigma = sigma.float().cuda() * 2 * math.sqrt(2*math.log(2))
            # target = torch.sqrt(measurement)

            for i in range(1):
                measurement_list = []
                output_list = []
                output, _ = self.ema_model.physics_informed_sample_unknown_real(measurement, dist=distance)

            folder_name = input_path

            output_path = self.results_folder / 'physical_forward_real'
            os.makedirs(output_path, exist_ok=True)

            measurement_list.append(measurement.squeeze(0))
            output_list.append(output[0])

            measurement_list = torch.cat(measurement_list, dim=0)
            measurement_list = torch.clamp(measurement_list, 0, 1)
            output_list = torch.cat(output_list, dim=0)

            video_path = str(output_path / f'{folder_name}_measurement.gif')
            video_tensor_to_gif(measurement_list, video_path)

            video_path = str(output_path / f'{folder_name}_output.gif')
            video_tensor_to_gif(output_list, video_path)

        print('Physics informed real sampling completed')


    def eval(self):
        milestone = self.step // self.save_and_sample_every

        output = self.ema_model.sample(batch_size=1)
                    
        output_path = str(self.results_folder)
        os.makedirs(output_path, exist_ok=True)

        video_path = output_path + str(f'/{milestone}_output.gif')
        video_tensor_to_gif(output.squeeze(0), video_path)

        print('Sampling completed')
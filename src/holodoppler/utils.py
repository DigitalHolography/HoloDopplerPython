"""
Utility functions for array operations
"""

import numpy as np
import scipy.fftpack as fft
from matlab_imresize import imresize

def resize_fft2_slicewise(img, new_h, new_w, xp=np, fft=np.fft):
    img = img.astype(xp.float32)
    h, w = img.shape[:2]
    rest = img.shape[2:]
    img = img.reshape(h, w, -1)
    n_slices = img.shape[-1]
    out = xp.empty((new_h, new_w, n_slices), dtype=xp.float32)
    for i in range(n_slices):
        slice_2d = img[:, :, i]
        F = fft.fftshift(
            fft.fft2(slice_2d),
        )
        F_new = xp.zeros((new_h, new_w), dtype=F.dtype)
        h_min, w_min = min(h, new_h), min(w, new_w)
        ho, wo = (h - h_min)//2, (w - w_min)//2
        hn, wn = (new_h - h_min)//2, (new_w - w_min)//2
        F_new[hn:hn+h_min, wn:wn+w_min] = F[ho:ho+h_min, wo:wo+w_min]
        resized = fft.ifft2(
            fft.ifftshift(F_new)
        ).real
        resized *= (new_h * new_w) / (h * w)
        out[:, :, i] = resized
    return out.reshape(new_h, new_w, *rest)

def resize_matlab_slicewise(img, new_h, new_w, xp=np):
    img = img.astype(xp.float32)
    h, w = img.shape[:2]
    rest = img.shape[2:]
    img = img.reshape(h, w, -1)
    n_slices = img.shape[-1]
    out = xp.empty((new_h, new_w, n_slices), dtype=xp.float32)
    for i in range(n_slices):
        slice_2d = img[:, :, i]
        resized = imresize(slice_2d, output_shape=(new_h, new_w))
        out[:, :, i] = resized
    return out.reshape(new_h, new_w, *rest)

def pad_array_centrally(arr, new_shape, xp):
    """Pad array centrally to new shape"""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    ny, nx = arr.shape[-2:]
    new_ny, new_nx = new_shape
    
    if new_ny < ny or new_nx < nx:
        raise ValueError("new_shape must be >= current shape")
    
    pad_y0 = (new_ny - ny) // 2
    pad_y1 = new_ny - ny - pad_y0
    pad_x0 = (new_nx - nx) // 2
    pad_x1 = new_nx - nx - pad_x0
    
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-2] = (pad_y0, pad_y1)
    pad_width[-1] = (pad_x0, pad_x1)
    
    return xp.pad(arr, pad_width, mode="constant")


def crop_array_centrally(arr, target_shape, xp):
    """Crop array centrally to target shape"""
    if isinstance(target_shape, int):
        target_shape = (target_shape, target_shape)
    
    ny, nx = arr.shape[-2:]
    tgt_ny, tgt_nx = target_shape
    
    if tgt_ny > ny or tgt_nx > nx:
        raise ValueError("target_shape must be <= current shape")
    
    crop_y0 = (ny - tgt_ny) // 2
    crop_y1 = crop_y0 + tgt_ny
    crop_x0 = (nx - tgt_nx) // 2
    crop_x1 = crop_x0 + tgt_nx
    
    slices = [slice(None)] * arr.ndim
    slices[-2] = slice(crop_y0, crop_y1)
    slices[-1] = slice(crop_x0, crop_x1)
    
    return arr[tuple(slices)]


def elliptical_mask(ny, nx, radius_frac, xp):
    """Create elliptical boolean mask"""
    radius_frac = max(0.0, min(1.0, float(radius_frac)))
    a = (nx / 2) * radius_frac
    b = (ny / 2) * radius_frac
    
    Y, X = xp.ogrid[:ny, :nx]
    cy, cx = ny / 2, nx / 2
    
    mask = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
    return mask


def gaussian_flatfield(A, gaussian_width, gaussian_filter_func):
    """Apply Gaussian flatfield correction"""
    return A / gaussian_filter_func(A, gaussian_width)


def subpixel_parabola(vm, v0, vp):
    """Subpixel refinement using parabola fit"""
    denom = vm - 2.0 * v0 + vp
    if abs(float(denom)) < 1e-12:
        return 0.0
    return 0.5 * float(vm - vp) / float(denom)


def signed_peak(ky, kx, ny, nx):
    """Convert peak indices to signed shifts"""
    if ky > ny // 2:
        ky -= ny
    if kx > nx // 2:
        kx -= nx
    return float(ky), float(kx)


def temporal_gaussian_filter(arr, sigma):
    """Apply 1D Gaussian filter along time axis"""
    if sigma == 0:
        return arr
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr.astype(np.float32), sigma=sigma, axis=2)


def normalize_image(arr):
    """Normalize image to 0-255 range"""
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    return arr.astype(np.uint8)

def temporal_gaussian(arr, sigma):
                if sigma == 0 :
                    return arr
                return gaussian_filter1d(arr.astype(np.float32), sigma=sigma, axis=2)
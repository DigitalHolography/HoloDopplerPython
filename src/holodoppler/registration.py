"""
Image registration: translation, rotation, and scale estimation
"""

import numpy as np


class ImageRegistration:
    """Image registration using phase correlation"""
    
    def __init__(self, backend_manager):
        self.bm = backend_manager
    
    @staticmethod
    def _xcorr2_fft(a, b, xp):
        """Cross-correlation via FFT"""
        fa = xp.fft.fft2(a)
        fb = xp.fft.fft2(b)
        return xp.fft.ifft2(fa * xp.conj(fb))
    
    def _phase_corr_subpixel(self, a, b):
        """Subpixel phase correlation"""
        xp = self.bm.xp
        ny, nx = a.shape[-2:]
        
        fa = xp.fft.fft2(a)
        fb = xp.fft.fft2(b)
        
        cps = fb * fa.conj()
        cps /= xp.abs(cps) + 1e-12
        
        corr = xp.fft.ifft2(cps)
        mag = xp.abs(corr)
        
        ky, kx = xp.unravel_index(xp.argmax(mag), mag.shape)
        ky, kx = int(ky), int(kx)
        
        from .utils import signed_peak, subpixel_parabola
        peak_y, peak_x = signed_peak(ky, kx, ny, nx)
        
        sub_y = subpixel_parabola(
            mag[(ky - 1) % ny, kx],
            mag[ky, kx],
            mag[(ky + 1) % ny, kx],
        )
        sub_x = subpixel_parabola(
            mag[ky, (kx - 1) % nx],
            mag[ky, kx],
            mag[ky, (kx + 1) % nx],
        )
        
        return -(peak_y + sub_y), -(peak_x + sub_x)
    
    def translation_only(self, fixed, moving, radius=None):
        """Estimate translation only"""
        xp = self.bm.xp
        ny, nx = fixed.shape
        
        from .utils import elliptical_mask, gaussian_flatfield
        
        mask = elliptical_mask(ny, nx, radius, xp) if radius else xp.ones((ny, nx), dtype=bool)
        
        fixed_f = fixed.astype(xp.float32)
        moving_f = moving.astype(xp.float32)
        
        # Apply Gaussian smoothing
        fixed_smooth = self.bm.gaussian_filter(fixed_f, 1.5)
        moving_smooth = self.bm.gaussian_filter(moving_f, 1.5)
        
        fixed_c = (fixed_smooth - xp.mean(fixed_smooth[mask])) * mask
        moving_c = (moving_smooth - xp.mean(moving_smooth[mask])) * mask
        
        fixed_c = fixed_c / (xp.max(xp.abs(fixed_c)) + 1e-12)
        moving_c = moving_c / (xp.max(xp.abs(moving_c)) + 1e-12)
        
        xcorr = self._xcorr2_fft(fixed_c, moving_c, xp)
        mag = xp.abs(xcorr)
        
        ky0, kx0 = xp.unravel_index(xp.argmax(mag), mag.shape)
        return (int(ky0), int(kx0))
    
    def apply_translation(self, img, shift_y, shift_x):
        """Apply translation using Fourier phase shift"""
        xp = self.bm.xp
        ny, nx = img.shape[-2:]
        
        fy = xp.fft.fftfreq(ny).reshape(ny, 1)
        fx = xp.fft.fftfreq(nx).reshape(1, nx)
        
        phase = xp.exp(-2j * xp.pi * (fy * shift_y + fx * shift_x))
        
        out = xp.fft.ifft2(
            xp.fft.fft2(img, axes=(-2, -1)) * phase,
            axes=(-2, -1),
        )
        
        if xp.isrealobj(img):
            out = out.real
        
        return out.astype(img.dtype)
    
    def apply_roll(self, img, peaks, xp):
        """Apply translation using roll (integer shifts only)"""
        peak_y, peak_x = peaks
        return xp.roll(xp.roll(img, peak_y, axis=-2), peak_x, axis=-1)
    
    def _logpolar_transform(self, img, radial_bins, angular_bins):
        """Log-polar transform for rotation/scale estimation"""
        xp = self.bm.xp
        
        if self.bm.is_gpu:
            from cupyx.scipy.ndimage import map_coordinates
        else:
            from scipy.ndimage import map_coordinates
        
        ny, nx = img.shape
        cy = (ny - 1) * 0.5
        cx = (nx - 1) * 0.5
        
        max_radius = min(cx, cy)
        log_r = xp.linspace(0.0, xp.log(max_radius), radial_bins)
        theta = xp.linspace(0.0, 2.0 * xp.pi, angular_bins, endpoint=False)
        
        rr = xp.exp(log_r).reshape(-1, 1)
        tt = theta.reshape(1, -1)
        
        yy = cy + rr * xp.sin(tt)
        xx = cx + rr * xp.cos(tt)
        
        coords = xp.stack([yy, xx], axis=0)
        return map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    
    def _fourier_magnitude(self, img):
        """Fourier magnitude for rotation/scale estimation"""
        xp = self.bm.xp
        
        mag = xp.abs(xp.fft.fftshift(xp.fft.fft2(img)))
        mag = xp.log1p(mag)
        
        ny, nx = mag.shape
        cy, cx = ny // 2, nx // 2
        r = max(4, min(ny, nx) // 32)
        yy, xx = xp.ogrid[:ny, :nx]
        dc_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        
        mag = mag.copy()
        mag[dc_mask] = 0
        
        return mag
    
    def estimate_rotation_scale(self, fixed, moving, radial_bins=256, angular_bins=360):
        """Estimate rotation angle and scale factor"""
        xp = self.bm.xp
        
        fixed_mag = self._fourier_magnitude(fixed)
        moving_mag = self._fourier_magnitude(moving)
        
        fixed_lp = self._logpolar_transform(fixed_mag, radial_bins, angular_bins)
        moving_lp = self._logpolar_transform(moving_mag, radial_bins, angular_bins)
        
        d_r, d_theta = self._phase_corr_subpixel(fixed_lp, moving_lp)
        
        angle_deg = -d_theta * 360.0 / angular_bins
        max_radius = min(fixed.shape[-1], fixed.shape[-2]) * 0.5
        log_base = xp.log(max_radius) / radial_bins
        scale = float(xp.exp(d_r * log_base))
        
        return float(angle_deg), scale
    
    def apply_rotation_scale(self, img, angle_deg, scale):
        """Apply rotation and scaling to image"""
        xp = self.bm.xp
        
        if self.bm.is_gpu:
            from cupyx.scipy.ndimage import affine_transform
        else:
            from scipy.ndimage import affine_transform
        
        ny, nx = img.shape[-2:]
        cy = (ny - 1) * 0.5
        cx = (nx - 1) * 0.5
        
        angle = xp.deg2rad(angle_deg)
        c = float(xp.cos(angle))
        s = float(xp.sin(angle))
        
        matrix = xp.asarray([
            [c / scale, s / scale],
            [-s / scale, c / scale],
        ], dtype=xp.float32)
        
        center = xp.asarray([cy, cx], dtype=xp.float32)
        offset = center - matrix @ center
        
        out = affine_transform(img, matrix, offset=offset, order=1, mode="nearest")
        return out.astype(img.dtype)
    
    def register_trs(self, fixed, moving, radius=None, estimate_similarity=False,
                     radial_bins=256, angular_bins=360, return_registered=False):
        """Full TRS (Translation, Rotation, Scale) registration"""
        xp = self.bm.xp
        ny, nx = fixed.shape
        
        from .utils import elliptical_mask, gaussian_flatfield
        
        mask = elliptical_mask(ny, nx, radius, xp) if radius else xp.ones((ny, nx), dtype=bool)
        
        fixed_f = fixed.astype(xp.float32)
        moving_f = moving.astype(xp.float32)
        
        fixed_c = (fixed_f - xp.mean(fixed_f[mask])) * mask
        moving_c = (moving_f - xp.mean(moving_f[mask])) * mask
        
        if estimate_similarity:
            angle_deg, scale = self.estimate_rotation_scale(
                fixed_c, moving_c, radial_bins, angular_bins
            )
            moving_rs = self.apply_rotation_scale(moving_f, angle_deg, scale)
            moving_rs_c = (moving_rs - xp.mean(moving_rs[mask])) * mask
        else:
            angle_deg = 0.0
            scale = 1.0
            moving_rs = moving_f
            moving_rs_c = moving_c
        
        shift_y, shift_x = self._phase_corr_subpixel(fixed_c, moving_rs_c)
        
        if not return_registered:
            return shift_y, shift_x, angle_deg, scale
        
        moving_registered = self.apply_translation(moving_rs, shift_y, shift_x)
        return shift_y, shift_x, angle_deg, scale, moving_registered
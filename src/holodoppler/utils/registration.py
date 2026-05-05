import numpy as np
from scipy.ndimage import gaussian_filter
from matlab_imresize import imresize
import scipy.fft as np_fft


class RegistrationUtils():
    # ------------------------------------------------------------
    # Flatfielding for registration
    # ------------------------------------------------------------

    def _flatfield(self, A, gaussian_width):
        return A / self.gaussian_filter(A,gaussian_width)
    
    # ------------------------------------------------------------
    # Resizing for square output
    # ------------------------------------------------------------

    def resize_fft2_slicewise(self, img, new_h, new_w):
        xp = np
        fft = np_fft
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
    
    def resize_matlab_slicewise(self, img, new_h, new_w):
        xp = np
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

    # ------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------

    @staticmethod
    def _elliptical_mask(ny, nx, radius_frac, xp):
        radius_frac = max(0.0, min(1.0, float(radius_frac)))
        a = (nx / 2) * radius_frac
        b = (ny / 2) * radius_frac

        Y, X = xp.ogrid[:ny, :nx]
        cy, cx = ny / 2, nx / 2

        mask = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
        return mask

    @staticmethod
    def _xcorr2d(fixed, moving, xp):
        f_fixed = xp.fft.fft2(fixed)
        f_moving = xp.fft.fft2(moving)
        cross_power = f_moving * f_fixed.conj()
        cross_power /= (xp.abs(cross_power) + 1e-12)
        return xp.fft.ifft2(cross_power)

    @staticmethod
    def _xcorr2_fft(a, b, xp):
        fa = xp.fft.fft2(a)
        fb = xp.fft.fft2(b)
        return xp.fft.ifft2(fa * xp.conj(fb))
    
    @staticmethod
    def _roll2d(img, peak_y, peak_x, xp):
        return xp.roll(xp.roll(img, peak_y, axis = -2), peak_x, axis = -1)
    
    @staticmethod
    def _signed_peak(ky, kx, ny, nx):
        if ky > ny // 2:
            ky -= ny
        if kx > nx // 2:
            kx -= nx
        return float(ky), float(kx)

    @staticmethod
    def _subpixel_parabola(vm, v0, vp):
        denom = vm - 2.0 * v0 + vp
        if abs(float(denom)) < 1e-12:
            return 0.0
        return 0.5 * float(vm - vp) / float(denom)

    def _phase_corr_subpixel(self, a, b, xp):
        ny, nx = a.shape[-2:]

        fa = xp.fft.fft2(a)
        fb = xp.fft.fft2(b)

        cps = fb * fa.conj()
        cps /= xp.abs(cps) + 1e-12

        corr = xp.fft.ifft2(cps)
        mag = xp.abs(corr)

        ky, kx = xp.unravel_index(xp.argmax(mag), mag.shape)
        ky, kx = int(ky), int(kx)

        peak_y, peak_x = self._signed_peak(ky, kx, ny, nx)

        sub_y = self._subpixel_parabola(
            mag[(ky - 1) % ny, kx],
            mag[ky, kx],
            mag[(ky + 1) % ny, kx],
        )

        sub_x = self._subpixel_parabola(
            mag[ky, (kx - 1) % nx],
            mag[ky, kx],
            mag[ky, (kx + 1) % nx],
        )

        peak_y += sub_y
        peak_x += sub_x

        return -peak_y, -peak_x

    @staticmethod
    def _logpolar(img, radial_bins, angular_bins, xp):
        if xp.__name__ == "cupy":
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

    @staticmethod
    def _fourier_magnitude_for_similarity(img, xp):
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

    def _estimate_rotation_scale(self, fixed, moving, xp, radial_bins=256, angular_bins=360):
        fixed_mag = self._fourier_magnitude_for_similarity(fixed, xp)
        moving_mag = self._fourier_magnitude_for_similarity(moving, xp)

        fixed_lp = self._logpolar(fixed_mag, radial_bins, angular_bins, xp)
        moving_lp = self._logpolar(moving_mag, radial_bins, angular_bins, xp)

        d_r, d_theta = self._phase_corr_subpixel(fixed_lp, moving_lp, xp)

        angle_deg = -d_theta * 360.0 / angular_bins

        max_radius = min(fixed.shape[-1], fixed.shape[-2]) * 0.5
        log_base = xp.log(max_radius) / radial_bins
        scale = float(xp.exp(d_r * log_base))

        return float(angle_deg), scale

    @staticmethod
    def _apply_rotation_scale(img, angle_deg, scale, xp):
        if xp.__name__ == "cupy":
            from cupyx.scipy.ndimage import affine_transform
        else:
            from scipy.ndimage import affine_transform

        ny, nx = img.shape[-2:]
        cy = (ny - 1) * 0.5
        cx = (nx - 1) * 0.5

        angle = xp.deg2rad(angle_deg)
        c = float(xp.cos(angle))
        s = float(xp.sin(angle))

        matrix = xp.asarray(
            [
                [c / scale, s / scale],
                [-s / scale, c / scale],
            ],
            dtype=xp.float32,
        )

        center = xp.asarray([cy, cx], dtype=xp.float32)
        offset = center - matrix @ center

        out = affine_transform(
            img,
            matrix,
            offset=offset,
            order=1,
            mode="nearest",
        )

        return out.astype(img.dtype, copy=False)
    
    @staticmethod
    def new_applyshifts(img, shift_y, shift_x, xp):
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

        return out.astype(img.dtype, copy=False)

    def _registration_trs(
        self,
        fixed,
        moving,
        radius=None,
        *,
        estimate_similarity=False,
        radial_bins=256,
        angular_bins=360,
        return_registered=False,
    ):
        xp = self.xp
        ny, nx = fixed.shape

        mask = (
            self._elliptical_mask(ny, nx, radius, xp)
            if radius
            else xp.ones((ny, nx), dtype=bool)
        )

        fixed_f = fixed.astype(xp.float32, copy=False)
        moving_f = moving.astype(xp.float32, copy=False)

        fixed_c = (fixed_f - xp.mean(fixed_f[mask])) * mask
        moving_c = (moving_f - xp.mean(moving_f[mask])) * mask

        if estimate_similarity:
            angle_deg, scale = self._estimate_rotation_scale(
                fixed_c,
                moving_c,
                xp,
                radial_bins=radial_bins,
                angular_bins=angular_bins,
            )

            moving_rs = self._apply_rotation_scale(
                moving_f,
                angle_deg,
                scale,
                xp,
            )

            moving_rs_c = (moving_rs - xp.mean(moving_rs[mask])) * mask

        else:
            angle_deg = 0.0
            scale = 1.0
            moving_rs = moving_f
            moving_rs_c = moving_c

        shift_y, shift_x = self._phase_corr_subpixel(fixed_c, moving_rs_c, xp)

        if not return_registered:
            return shift_y, shift_x, angle_deg, scale

        moving_registered = self.new_applyshifts(
            moving_rs,
            shift_y,
            shift_x,
            xp,
        )

        return shift_y, shift_x, angle_deg, scale, moving_registered
    
    def applyregistration(self, img, reg, xp):
        """
        Apply a registration tuple to an image.

        Parameters
        ----------
        img : array (numpy or cupy)
        reg : tuple
            (shift_y, shift_x, angle_deg, scale)
            or (shift_y, shift_x) for translation-only
        xp : numpy or cupy module

        Returns
        -------
        registered image
        """
        # --- Parse registration tuple ---
        if len(reg) == 2:
            shift_y, shift_x = reg
            angle_deg = 0.0
            scale = 1.0
        else:
            shift_y, shift_x, angle_deg, scale = reg

        out = img

        # --- Apply rotation + scale (if needed) ---
        if angle_deg != 0.0 or scale != 1.0:
            out = self._apply_rotation_scale(out, angle_deg, scale, xp)

        # --- Apply subpixel translation ---
        out = self.new_applyshifts(out, shift_y, shift_x, xp)

        return out

    def old_registration(self, fixed, moving, radius):
        ny, nx = fixed.shape

        xp = self.xp

        mask = self._elliptical_mask(ny, nx, radius, xp) if radius else xp.ones((ny, nx), dtype=bool)

        _fixed = self.gaussian_filter(fixed,1.5)
        _moving = self.gaussian_filter(moving,1.5)

        fixed_c = (_fixed - xp.mean(_fixed[mask])) * mask
        moving_c = (_moving - xp.mean(_moving[mask])) * mask

        fixed_c = fixed_c / xp.max(xp.abs(fixed_c))
        moving_c = moving_c / xp.max(xp.abs(moving_c))

        xcorr = self._xcorr2_fft(fixed_c, moving_c, xp)

        mag = xp.abs(xcorr)

        ky0, kx0 = xp.unravel_index(xp.argmax(mag), mag.shape)
        peak_y, peak_x = int(ky0), int(kx0)

        # moving_reg = self._roll2d(moving, -peak_y, -peak_x, xp)
        return (peak_y, peak_x)
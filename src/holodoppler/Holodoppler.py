import numpy as np
import traceback
import h5py
from tqdm import tqdm
import cv2
import json
import os
import threading
import queue
import time
import cinereader

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Optional CuPy support
# ------------------------------------------------------------



try:
    import cupy as cp
    import cupyx.scipy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    import cupyx.scipy.ndimage as cp_ndi

    _cupy_available = True
except Exception:
    cp = None
    cp_fft = None
    _cupy_available = False

import scipy.fft as np_fft
from scipy.ndimage import gaussian_filter as np_gaussian_filter
from matlab_imresize.imresize import imresize
import scipy.ndimage as np_ndi
from scipy.interpolate import griddata
import scipy.fftpack as np_fftpack
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Holodoppler:
    """
    Holodoppler processing class for .holo files. 
        adding support for .cine files.

    Backend:
        backend="numpy"  -> CPU
        backend="cupy"   -> GPU (if available)

    Data is always read on CPU then transferred to GPU if needed.
    """

    HOLO_HEADER_SIZE = 64

    def __init__(self, backend = "numpy", pipeline_version = "latest"):

        self.file_path = ""

        self.backend = backend

        self.pipeline_version = pipeline_version

        self._init_backend()

        self._init_pipeline()

        self.fid = None

        self.parameters = dict()

        self.kernels = dict()

    # ------------------------------------------------------------
    # Backend handling
    # ------------------------------------------------------------

    def _init_backend(self):

        if self.backend == "cupy" or self.backend == "cupy_ramdisk" or self.backend == "cupy_diagnostic":
            if not _cupy_available:
                raise RuntimeError("CuPy backend requested but CuPy is not available.")
            self.xp = cp
            self.fft = cp_fft
            self.gaussian_filter = cp_gaussian_filter
            self.ndi = cp_ndi
            self.fftpack = cp_fft
        else:
            self.backend = "numpy"
            self.xp = np
            self.fft = np_fft
            self.gaussian_filter = np_gaussian_filter
            self.ndi = np_ndi
            self.fftpack = np_fftpack
            
    def _to_backend(self, arr):
        if self.backend == "cupy" or self.backend == "cupy_diagnostic" or self.backend == "cupy_ramdisk":
            return cp.asarray(arr)
        return arr

    def _to_numpy(self, arr):
        if self.backend == "cupy" or self.backend == "cupy_diagnostic" or self.backend == "cupy_ramdisk":
            return cp.asnumpy(arr)
        return arr

    def _init_pipeline(self):
        if self.pipeline_version == "latest":
            self._frequency_symmetric_filtering = self._new_frequency_symmetric_filtering
            self._resize = self.resize_fft2_slicewise
            self._registration = self.new_registration
            return
        elif self.pipeline_version == "old":
            self._frequency_symmetric_filtering = self._old_frequency_symmetric_filtering
            self._resize = self.resize_matlab_slicewise
            self._registration = self.old_registration
            self._moment = self._momentkHz

    # ------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------

    def load_file(self, file_path):

        if self.fid is not None:
            self.fid.close()

        _, ext = os.path.splitext(file_path)

        self.file_path = file_path

        if ext == ".holo":
            self.ext = ext
            def _extract_holo_footer(f, w, h, numframes):
                offset = w * h * numframes + 64
                f.seek(offset)
                footer_bytes = f.read()
                if not footer_bytes:
                    return {}
                try:
                    footer_str = footer_bytes.decode("utf-8")
                    return json.loads(footer_str)
                except Exception:
                    return {}
            self.fid = open(self.file_path, "rb")
            header = self.fid.read(self.HOLO_HEADER_SIZE)
            file_header = dict()
            file_header["magic_number"] = ''.join(list(map(chr, header[0:4])))
            file_header["version"] = int.from_bytes(header[4:6], "little")
            file_header["bit_depth"] = int.from_bytes(header[6:8], "little")
            file_header["width"] = int.from_bytes(header[8:12], "little")
            file_header["height"] = int.from_bytes(header[12:16], "little")
            file_header["num_frames"] = int.from_bytes(header[16:20], "little")
            file_header["total_size"] = int.from_bytes(header[20:28], "little")
            file_header["endianness"] = header[28]

            self.file_footer = _extract_holo_footer(self.fid, file_header["width"], file_header["height"], file_header["num_frames"])
            self.file_header = file_header
            self.read_frames = self.read_frames_holo

        elif ext == ".cine":
            self.ext = ext
            self.cine_metadata = cinereader.read_metadata(file_path)
            self.cine_metadata_json = dict(self.cine_metadata.__dict__)
            self.read_frames = self.read_frames_cine

    def _close_file(self):
        if self.fid is not None:
            self.fid.close()

    # ------------------------------------------------------------
    # Reading frames (CPU only, then transferred if needed)
    # ------------------------------------------------------------

    def read_frames_cine(self, first_frame, frame_size):
        _, images, _ = cinereader.read(self.file_path, self.cine_metadata.FirstImageNo + first_frame, frame_size)
        frames = np.stack(images, axis=0)
        return self._to_backend(frames)
    
    def read_frames_holo(self, first_frame, frame_size, fid = None):
        

        try:

            if fid is None:
                fid = self.fid
            
            byte_begin = (
                self.HOLO_HEADER_SIZE
                + self.file_header["width"] * self.file_header["height"] * first_frame * self.file_header["bit_depth"] // 8
            )

            byte_size = self.file_header["width"] * self.file_header["height"] * frame_size * self.file_header["bit_depth"] // 8

            fid.seek(byte_begin)
            raw_bytes = fid.read(byte_size)

            if self.file_header["bit_depth"] == 8:
                utyp = np.uint8
            elif self.file_header["bit_depth"] == 16:
                utyp = np.uint16
            else:
                raise RuntimeError("Unsupported bit depth : Supported bit depth are 8 bits or 16 bits")

            if self.file_header["endianness"] == 1:
                utyp = utyp.newbyteorder('<')

            out = np.frombuffer(raw_bytes, dtype=utyp)

            out = out.reshape(
                (frame_size,self.file_header["height"], self.file_header["width"]),
                order="C"
            )
            
            return self._to_backend(out)

        except Exception:
            traceback.print_exc()
            return None

    # ------------------------------------------------------------
    # Calculation kernels
    # ------------------------------------------------------------

    def _build_fresnel_kernel(self,
                              z,
                              pixel_pitch,
                              wavelength, ny, nx):

        if isinstance(pixel_pitch, (float, int)):
            pixel_pitch = (pixel_pitch, pixel_pitch)

        ppy, ppx = pixel_pitch

        xp = self.xp

        y = (xp.arange(0, ny) - xp.round(ny / 2)) * ppy
        x = (xp.arange(0, nx) - xp.round(nx / 2)) * ppx

        X, Y = xp.meshgrid(x, y)

        kernel = xp.exp(
            1j * xp.pi / (wavelength * z) * (X ** 2 + Y ** 2)
        ).astype(xp.complex64)

        self.kernels["Fresnel"] = kernel[xp.newaxis, ...]

    def _build_angular_kernel(self,
                              z,
                              pixel_pitch,
                              wavelength, ny, nx):

        if isinstance(pixel_pitch, (float, int)):
            pixel_pitch = (pixel_pitch, pixel_pitch)

        ppy, ppx = pixel_pitch

        xp = self.xp

        du = 1.0 / (nx * ppx)
        dv = 1.0 / (ny * ppy)
        u = (xp.arange(1, int(nx) + 1) - 1 - xp.round(nx / 2)) * du
        v = (xp.arange(1, int(ny) + 1) - 1 - xp.round(ny / 2)) * dv
        U, V = xp.meshgrid(u, v)
        kernel = xp.exp(
            2j * xp.pi * z / wavelength *
            xp.sqrt(1.0 - (wavelength * U) ** 2 - (wavelength * V) ** 2)
        )

        self.kernels["AngularSpectrum"] = kernel[xp.newaxis, ...]

    # ------------------------------------------------------------
    # Calculation processing
    # ------------------------------------------------------------

    def _fresnel_transform(self, frames):

        return self.fft.fftshift(
            self.fft.fft2(frames *self.kernels["Fresnel"], axes=(-1, -2), norm="ortho"), axes=(-1, -2)
        )
    
    def _angular_spectrum_transform(self, frames):

        tmp = self.fft.fft2(frames,axes=(-1, -2), norm="ortho") * self.fft.fftshift(self.kernels["AngularSpectrum"],axes=(-1, -2))

        return self.fft.ifft2(tmp,axes=(-1, -2), norm="ortho")

    def _fourier_time_transform(self, H):

        return self.fft.fft(H, axis=0, norm="ortho") 

    # ------------------------------------------------------------
    # SVD filtering
    # (Eigen decomposition of temporal covariance)
    # ------------------------------------------------------------

    def _svd_filter(self, H, svd_threshold):
        
        xp = self.xp

        if svd_threshold < 0:
            return H

        sz = H.shape
        H2 = H.reshape((sz[0],sz[-1] * sz[-2])).T

        cov = H2.conj().T @ H2 
        
        eps = 1e-12
        cov = cov + eps * xp.eye(cov.shape[0], dtype=cov.dtype)

        S, V = xp.linalg.eigh(cov)

        idx = xp.argsort(S)[::-1]
        V = V[:, idx]
        Vt = V[:, :svd_threshold]

        tissue = H2 @ Vt @ Vt.conj().T

        return (H2 - tissue).T.reshape(sz)

    # ------------------------------------------------------------
    # Frequency axis and masks
    # ------------------------------------------------------------

    def _new_frequency_symmetric_filtering(self, batch_size, sampling_freq, low_freq, high_freq = None):

        xp = self.xp

        # fftfreq is backend dependent
        freqs = self.fft.fftfreq(batch_size, 1 / sampling_freq)

        if high_freq is None:
            idxs = xp.abs(freqs) > low_freq
        else:
            idxs = (high_freq > xp.abs(freqs)) & (xp.abs(freqs) > low_freq)

        return idxs, freqs[idxs]
    
    def _old_frequency_symmetric_filtering(self, batch_size, sampling_freq, low_freq, high_freq = None):
        # old and not clean but similar to matlab version

        if high_freq is None:
            high_freq = sampling_freq / 2

        # convert frequencies to indices
        n1 = int(np.ceil(low_freq * batch_size / sampling_freq))
        n2 = int(np.ceil(high_freq * batch_size / sampling_freq))

        # clamp to valid range (MATLAB: 1..size(SH,3))
        n1 = max(min(n1, batch_size), 1)
        n2 = max(min(n2, batch_size), 1)

        # symmetric integration interval
        n3 = batch_size - n2 + 1
        n4 = batch_size - n1 + 1

        # convert to Python indexing (0-based)
        i1 = n1 - 1
        i2 = n2      # exclusive
        i3 = n3 - 1
        i4 = n4      # exclusive

        # frequency ranges (MATLAB inclusive -> +1 in Python)
        f_range = np.arange(n1, n2 + 1) * (sampling_freq / batch_size)
        f_range_sym = np.arange(-n2, -n1 + 1) * (sampling_freq / batch_size)

        freqs = np.concatenate([f_range, f_range_sym], axis=0)

        # boolean mask
        idxs = np.zeros(batch_size, dtype=bool)
        idxs[i1:i2] = True
        idxs[i3:i4] = True

        return self._to_backend(idxs), self._to_backend(freqs)

    # ------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------

    def _moment(self, A, freqs, n):

        xp = self.xp

        return xp.sum(
            A * (freqs[... ,xp.newaxis, xp.newaxis] ** n),
            axis=0
        ).astype(xp.float32)
        
    def _momentkHz(self, A, freqs, n):

        xp = self.xp

        return xp.sum(
            A * ((freqs[... ,xp.newaxis, xp.newaxis] / 1000) ** n),
            axis=0
        ).astype(xp.float32)
    
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
        # Normalize to avoid division by zero
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

    def new_registration(self, fixed, moving, radius):
        ny, nx = fixed.shape

        xp = self.xp

        mask = self._elliptical_mask(ny, nx, radius, xp) if radius else xp.ones((ny, nx), dtype=bool)

        # lo_f, hi_f = xp.percentile(fixed[mask], (0.2, 99.8))
        # _fixed = xp.clip(fixed, lo_f, hi_f)
        # lo_m, hi_m = xp.percentile(moving[mask], (0.2, 99.8))
        # _moving = xp.clip(moving, lo_m, hi_m)
        _fixed = fixed
        _moving = moving

        fixed_c = (_fixed - xp.mean(_fixed[mask])) * mask
        moving_c = (_moving - xp.mean(_moving[mask])) * mask

        xcorr = self._xcorr2d(fixed_c, moving_c, xp)

        mag = xp.abs(xcorr)

        ky0, kx0 = xp.unravel_index(xp.argmax(mag), mag.shape)
        peak_y, peak_x = int(ky0), int(kx0)

        # moving_reg = self._roll2d(moving, -peak_y, -peak_x, xp)
        return (-peak_y, -peak_x)

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
    
    # ------------------------------------------------------------
    # Shack-Hartmann wavefront reconstruction
    # ------------------------------------------------------------

    def _shack_hartmann_constructsubapsimages(self, U0, dx, dy, wavelength, z_prop, f0, f1, fs, time_window, nx_subabs, ny_subabs, svd_threshold):
        xp = self.xp
        Nz, Ny, Nx = U0.shape
        nb_iter = Nz // time_window
        idxs, _ = self._frequency_symmetric_filtering(time_window, fs, f0, high_freq = f1)
        U_subaps_stack = xp.zeros((nb_iter, ny_subabs, nx_subabs, Ny // ny_subabs, Nx // nx_subabs), dtype=xp.float32)
        Qin = self.kernels["Fresnel"]
        for i in range(nb_iter):
            start_idx = i * time_window
            end_idx = start_idx + time_window
            U_chunk = U0[start_idx:end_idx]     
            U_prop_qin = U_chunk * Qin # if zernike_phase is None else U_filter * Qin * xp.exp(- 1j * zernike_phase)
            U_subaps = xp.zeros((ny_subabs, nx_subabs, Ny // ny_subabs, Nx // nx_subabs), dtype=xp.float32)
            for (iy, ix) in [(iy, ix) for iy in range(ny_subabs) for ix in range(nx_subabs)]:
                y_start = iy * (Ny // ny_subabs)
                y_end = y_start + (Ny // ny_subabs)
                x_start = ix * (Nx // nx_subabs)
                x_end = x_start + (Nx // nx_subabs)
                U_subap = U_prop_qin[:, y_start:y_end, x_start:x_end]
                U_subap_filtered = self._svd_filter(U_subap, svd_threshold)
                U_prop_filtered = xp.fft.fft2(U_subap_filtered, axes=(-2, -1))
                U_prop_filtered = xp.fft.fftshift(U_prop_filtered, axes=(-2, -1))
                U_fourier = xp.fft.fft(U_prop_filtered, axis=0)
                U_filter = U_fourier[idxs,:,:]
                S = xp.abs(U_filter)**2
                M0 = xp.mean(S, axis=0)
                U_subaps[iy, ix] = (M0.astype(xp.float32)) #flatfield(M0.astype(xp.float32),15)
            U_subaps_stack[i] = U_subaps
        U_subaps = xp.mean(U_subaps_stack, axis=0)
        return U_subaps   

    def _shack_hartmann_registration(self, fixed, moving, radius):
        xp = self.xp
        def parabolic_peak_1d(line, peak_idx):
            v_m = line[peak_idx - 1]
            v_0 = line[peak_idx]
            v_p = line[peak_idx + 1]
            denom = v_m - 2.0 * v_0 + v_p
            delta = 0.5 * (v_m - v_p) / denom
            return peak_idx + delta
        def _elliptical_mask(ny, nx, radius_frac, xp):
            radius_frac = max(0.0, min(2.0, float(radius_frac)))
            a = (nx / 2) * radius_frac
            b = (ny / 2) * radius_frac
            Y, X = xp.ogrid[:ny, :nx]
            cy, cx = ny / 2, nx / 2
            mask = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
            return mask
        def _xcorr2d(fixed, moving, xp):
            f_fixed = xp.fft.fft2(fixed)
            f_moving = xp.fft.fft2(moving)
            cross_power = f_moving * f_fixed.conj()
            # Normalize to avoid division by zero
            cross_power /= (xp.abs(cross_power) + 1e-12)
            return xp.fft.fftshift(xp.fft.ifft2(cross_power))
        ny, nx = fixed.shape
        mask = _elliptical_mask(ny, nx, radius, xp) if radius else xp.ones((ny, nx), dtype=bool)
        lo_f, hi_f = xp.percentile(fixed[mask], (0.2, 99.8))
        _fixed = xp.clip(fixed, lo_f, hi_f)
        lo_m, hi_m = xp.percentile(moving[mask], (0.2, 99.8))
        _moving = xp.clip(moving, lo_m, hi_m)
        fixed_c = (_fixed - xp.mean(_fixed[mask])) * mask
        moving_c = (_moving - xp.mean(_moving[mask])) * mask
        xcorr = xp.abs(_xcorr2d(fixed_c, moving_c, xp))
        peak_idx = xp.argmax(xcorr)	
        peak_y, peak_x = xp.unravel_index(peak_idx, fixed.shape)
        refined_y = parabolic_peak_1d(xcorr[:, peak_x], peak_y)
        refined_x = parabolic_peak_1d(xcorr[peak_y, :], peak_x)
        center_x = nx/2
        center_y = ny/2
        shift_x = refined_x - center_x
        shift_y = refined_y - center_y
        return xcorr, (shift_y, shift_x)

    def _shack_hartmann_displacement_calculation(self, U_subabs, xp, ref=None):
        
                
        def filter_shifts_pupil(shifts_y, shifts_x, Ny, Nx, sub_ny, sub_nx, radius=0.9):
            shifts_y_c = xp.copy(shifts_y)
            shifts_x_c = xp.copy(shifts_x)
            x = xp.linspace(-1, 1, Nx)
            y = xp.linspace(-1, 1, Ny)
            for (iy, ix) in [(iy, ix) for iy in range(sub_ny) for ix in range(sub_nx)]:
                if x[ix*(Nx//sub_nx) + (Nx//sub_nx)//2]**2+y[iy*(Ny//sub_ny) + (Ny//sub_ny)//2]**2 > radius**2:
                    shifts_y_c[iy,ix], shifts_x_c[iy,ix] = xp.nan, xp.nan
            return shifts_y_c, shifts_x_c	

        def filter_shifts_threshold(shifts_y, shifts_x, Ny, Nx, sub_ny, sub_nx, threshold=0.9):
            shifts_y_c = xp.copy(shifts_y)
            shifts_x_c = xp.copy(shifts_x)

            meanx = xp.nanmean(shifts_x,axis=(0,1))
            meany = xp.nanmean(shifts_y,axis=(0,1))
            stdx = xp.nanstd(shifts_x,axis=(0,1))
            stdy = xp.nanstd(shifts_y,axis=(0,1))
            for (iy, ix) in [(iy, ix) for iy in range(sub_ny) for ix in range(sub_nx)]:
                if abs(shifts_x[iy, ix] - meanx) > threshold * stdx or abs(shifts_y[iy, ix] - meany) > threshold * stdy:
                    shifts_y_c[iy,ix], shifts_x_c[iy,ix] = xp.nan, xp.nan
            return shifts_y_c, shifts_x_c
        xp = self.xp
        ny_subabs, nx_subabs, _, _ = U_subabs.shape
        shifts_y = xp.zeros((ny_subabs, nx_subabs), dtype=xp.float32)
        shifts_x = xp.zeros((ny_subabs, nx_subabs), dtype=xp.float32)
        if (ref is None):
            if not (ny_subabs % 2 == 1 and nx_subabs % 2 == 1):
                print("Warning : Number of sub-apertures must be odd.")
            ref_subap = U_subabs[ny_subabs // 2, nx_subabs // 2] 
        else:
            ref_subap = ref
        for (iy, ix) in [(iy, ix) for iy in range(ny_subabs) for ix in range(nx_subabs)]:
            curr_subap = U_subabs[iy, ix]
            xcorr, (shift_y, shift_x) = self._shack_hartmann_registration(ref_subap, curr_subap, radius=None)
            shifts_y[iy, ix] = shift_y
            shifts_x[iy, ix] = shift_x
            
        # plot_shifts(shifts_y, shifts_x, nx_subabs, ny_subabs, title = "Wavefront Slopes from Sub-aperture Shifts", scale=50)

        shifts_y, shifts_x = filter_shifts_pupil(shifts_y, shifts_x, U_subabs.shape[-2], U_subabs.shape[-1], ny_subabs, nx_subabs, radius=1)
        shifts_y, shifts_x = filter_shifts_threshold(shifts_y, shifts_x, U_subabs.shape[-2], U_subabs.shape[-1], ny_subabs, nx_subabs, threshold=3)
        
        # plot_shifts(shifts_y, shifts_x, nx_subabs, ny_subabs, title = "Wavefront Slopes from Sub-aperture Shifts", scale=50)
        return shifts_y, shifts_x
    
    def _get_zernike_mode2(self, mode_index, Nx, Ny, radius = 2.0):
        xp = self.xp
        # Scale so that the smallest dimension spans [-1, 1]
        if Nx > Ny:
            x = xp.linspace(-Nx / Ny, Nx / Ny, Nx)
            y = xp.linspace(-1, 1, Ny)
        else:
            x = xp.linspace(-1, 1, Nx)
            y = xp.linspace(-Ny / Nx, Ny / Nx, Ny)
        X, Y = xp.meshgrid(x, y)
        R = xp.sqrt(X**2 + Y**2)
        Theta = xp.arctan2(Y, X)
        Z = xp.full_like(R, xp.nan)
        mask = R <= radius
        if mode_index == 1:
            Z[mask] = 1  # Piston
        elif mode_index == 2:
            Z[mask] = 2 * X[mask]  # Tilt X
        elif mode_index == 3:
            Z[mask] = 2 * Y[mask]  # Tilt Y
        elif mode_index == 4:
            Z[mask] = xp.sqrt(3) * (2 * R[mask]**2 - 1)  # Defocus
        elif mode_index == 5:
            Z[mask] = xp.sqrt(6) * R[mask]**2 * xp.sin(2 * Theta[mask])  # Astigmatism 45°
        elif mode_index == 6:
            Z[mask] = xp.sqrt(6) * R[mask]**2 * xp.cos(2 * Theta[mask])  # Astigmatism 0°
        elif mode_index == 7:
            Z[mask] = xp.sqrt(8) * (3*R[mask]**3 - 2*R[mask]) * xp.sin(Theta[mask])  # Coma Y
        elif mode_index == 8:
            Z[mask] = xp.sqrt(8) * (3*R[mask]**3 - 2*R[mask]) * xp.cos(Theta[mask])  # Coma X
        elif mode_index == 9:
            Z[mask] = xp.sqrt(8) * R[mask]**3 * xp.sin(3 * Theta[mask])  # Trefoil Y
        elif mode_index == 10:
            Z[mask] = xp.sqrt(8) * R[mask]**3 * xp.cos(3 * Theta[mask])  # Trefoil X
        elif mode_index == 11:
            Z[mask] = xp.sqrt(5) * (6*R[mask]**4 - 6*R[mask]**2 + 1)  # Spherical
        elif mode_index == 12:
            Z[mask] = xp.sqrt(10) * (4*R[mask]**4 - 3*R[mask]**2) * xp.cos(2 * Theta[mask])  # Astigmatism 0° (higher)
        elif mode_index == 13:
            Z[mask] = xp.sqrt(10) * (4*R[mask]**4 - 3*R[mask]**2) * xp.sin(2 * Theta[mask])  # Astigmatism 45° (higher)
        elif mode_index == 14:
            Z[mask] = xp.sqrt(10) * R[mask]**4 * xp.cos(4 * Theta[mask])  # Quadrafoil X
        elif mode_index == 15:
            Z[mask] = xp.sqrt(10) * R[mask]**4 * xp.sin(4 * Theta[mask])  # Quadrafoil Y
        elif mode_index == 16:
            Z[mask] = xp.sqrt(12) * (10*R[mask]**5 - 12*R[mask]**3 + 3*R[mask]) * xp.cos(Theta[mask])  # Secondary coma X
        elif mode_index == 17:
            Z[mask] = xp.sqrt(12) * (10*R[mask]**5 - 12*R[mask]**3 + 3*R[mask]) * xp.sin(Theta[mask])  # Secondary coma Y
        elif mode_index == 18:
            Z[mask] = xp.sqrt(12) * (5*R[mask]**5 - 4*R[mask]**3) * xp.cos(3 * Theta[mask])  # Secondary trefoil X
        elif mode_index == 19:
            Z[mask] = xp.sqrt(12) * (5*R[mask]**5 - 4*R[mask]**3) * xp.sin(3 * Theta[mask])  # Secondary trefoil Y
        elif mode_index == 20:
            Z[mask] = xp.sqrt(12) * R[mask]**5 * xp.cos(5 * Theta[mask])  # Pentafoil X
        elif mode_index == 21:
            Z[mask] = xp.sqrt(12) * R[mask]**5 * xp.sin(5 * Theta[mask])  # Pentafoil Y
        else:
            raise ValueError("Mode index not implemented")
        return Z
    
    def _get_legendre_mode(self, mode_index, Nx, Ny):
        xp = self.xp

        # Grid in [-1, 1]
        x = xp.linspace(-1, 1, Nx)
        y = xp.linspace(-1, 1, Ny)
        X, Y = xp.meshgrid(x, y)

        # --- Legendre polynomials (up to order 5) ---
        def P0(t): return xp.ones_like(t)
        def P1(t): return t
        def P2(t): return 0.5 * (3*t**2 - 1)
        def P3(t): return 0.5 * (5*t**3 - 3*t)
        def P4(t): return (1/8) * (35*t**4 - 30*t**2 + 3)
        def P5(t): return (1/8) * (63*t**5 - 70*t**3 + 15*t)

        # Precompute
        Px = [P0(X), P1(X), P2(X), P3(X), P4(X), P5(X)]
        Py = [P0(Y), P1(Y), P2(Y), P3(Y), P4(Y), P5(Y)]

        Z = xp.zeros_like(X)

        # Mode mapping (similar spirit to Zernike ordering)
        if mode_index == 1:
            Z = Px[0] * Py[0]  # piston
        elif mode_index == 2:
            Z = Px[1] * Py[0]  # tilt X
        elif mode_index == 3:
            Z = Px[0] * Py[1]  # tilt Y
        elif mode_index == 4:
            Z = Px[2] * Py[0]  # defocus-like (X)
        elif mode_index == 5:
            Z = Px[1] * Py[1]  # astig-like (diagonal)
        elif mode_index == 6:
            Z = Px[0] * Py[2]  # defocus-like (Y)
        elif mode_index == 7:
            Z = Px[3] * Py[0]
        elif mode_index == 8:
            Z = Px[2] * Py[1]
        elif mode_index == 9:
            Z = Px[1] * Py[2]
        elif mode_index == 10:
            Z = Px[0] * Py[3]
        elif mode_index == 11:
            Z = Px[4] * Py[0]
        elif mode_index == 12:
            Z = Px[3] * Py[1]
        elif mode_index == 13:
            Z = Px[2] * Py[2]
        elif mode_index == 14:
            Z = Px[1] * Py[3]
        elif mode_index == 15:
            Z = Px[0] * Py[4]
        elif mode_index == 16:
            Z = Px[5] * Py[0]
        elif mode_index == 17:
            Z = Px[4] * Py[1]
        elif mode_index == 18:
            Z = Px[3] * Py[2]
        elif mode_index == 19:
            Z = Px[2] * Py[3]
        elif mode_index == 20:
            Z = Px[1] * Py[4]
        elif mode_index == 21:
            Z = Px[0] * Py[5]
        else:
            raise ValueError("Mode index not implemented")
    
    def _shack_hartmann_zernike(self, ny, nx, pixel_pitch_y, pixel_pitch_x, wavelength, shifts_y, shifts_x, zernike_modes):
        xp = self.xp
        def make_G_gradient_zernike_matrix(Ny, Nx, mode_indices, n_sub_x, n_sub_y, dx, dy):
            n_modes = len(mode_indices)
            M = xp.zeros((2,n_sub_y, n_sub_x, n_modes))
            for k, idx in enumerate(mode_indices):
                Z = self._get_zernike_mode2(idx, Nx, Ny, radius = 2)
                dZdx = xp.gradient(Z, dx, axis=1)
                dZdy = xp.gradient(Z, dy, axis=0)
                for (iy, ix) in [(iy, ix) for iy in range(n_sub_y) for ix in range(n_sub_x)]:
                    y_start = iy * (Ny // n_sub_y)
                    y_end = y_start + (Ny // n_sub_y)
                    x_start = ix * (Nx // n_sub_x)
                    x_end = x_start + (Nx // n_sub_x)
                    dZdx_subap = dZdx[y_start:y_end, x_start:x_end]
                    dZdy_subap = dZdy[y_start:y_end, x_start:x_end]
                    dZdx_avg = xp.nanmean(dZdx_subap, axis=(0,1))
                    dZdy_avg = xp.nanmean(dZdy_subap, axis=(0,1))
                    M[0 ,iy, ix, k] = dZdy_avg
                    M[1 ,iy, ix, k] = dZdx_avg
            return M
        
        def solve_modes(M, s, plot=False):
            _, ny, nx, nm = M.shape
            A = M.reshape(-1, nm)
            b = s.reshape(-1)
            m = (~xp.isnan(b)) & (~xp.isnan(A).any(1))
            c, r, rank, sv = xp.linalg.lstsq(A[m], b[m], rcond=None)
            recon = xp.full_like(b, xp.nan, dtype=float)
            recon[m] = A[m] @ c
            recon = recon.reshape(2, ny, nx)
            return c, recon
        
        nysubabs, nxsubabs = shifts_y.shape
        
        slopes_y = shifts_y * (wavelength)/(pixel_pitch_y*(ny//nysubabs)) 
        slopes_x = shifts_x * (wavelength)/(pixel_pitch_x*(nx//nxsubabs))
        nysubabs, nxsubabs = shifts_y.shape
        
        s = xp.stack([slopes_y,slopes_x])
        
        if (not "G_gradient_zernike_matrix" in self.kernels):
            G = make_G_gradient_zernike_matrix(ny, nx, zernike_modes, nxsubabs, nysubabs, pixel_pitch_x, pixel_pitch_y) * wavelength / (2*xp.pi)
            self.kernels["G_gradient_zernike_matrix"] = G
        
        zernike_coefs_radians, _ = solve_modes(self.kernels["G_gradient_zernike_matrix"], s) 
        # print(zernike_coefs_radians, "radians")

        zern_phase = xp.sum(xp.stack([coef * self._get_zernike_mode2(idx, nx, ny, radius = 2) for idx, coef in zip(zernike_modes, zernike_coefs_radians)]), axis=0)
        
        return zernike_coefs_radians, zern_phase
    
    def _shack_hartmann_southwell(self, ny, nx, pixel_pitch_y, pixel_pitch_x, wavelength, shifts_y, shifts_x):
        xp = self.xp
        fftpack = self.fftpack
        def southwell_fourier_nan(slope_y, slope_x):
            """NaN-robust Southwell DCT Poisson solver."""
            rows, cols = slope_y.shape
            mask_x = xp.isfinite(slope_x)
            mask_y = xp.isfinite(slope_y)
            div = xp.zeros((rows, cols), dtype=float)
            valid_x = mask_x[:, :-1]
            sx = xp.where(valid_x, slope_x[:, :-1], 0.0)
            div[:, :-1] += sx
            div[:, 1:]  -= sx
            valid_y = mask_y[:-1, :]
            sy = xp.where(valid_y, slope_y[:-1, :], 0.0)
            div[:-1, :] += sy
            div[1:, :]  -= sy
            valid_div = xp.zeros_like(div, dtype=bool)
            valid_div[:, :-1] |= valid_x
            valid_div[:, 1:]  |= valid_x
            valid_div[:-1, :] |= valid_y
            valid_div[1:, :]  |= valid_y
            div = xp.where(valid_div, div, 0.0)
            div_dct = fftpack.dct(
                fftpack.dct(div, axis=0, norm="ortho"),
                axis=1, norm="ortho"
            )
            cx = xp.cos(xp.pi * xp.arange(cols) / cols)
            cy = xp.cos(xp.pi * xp.arange(rows) / rows)
            CX, CY = xp.meshgrid(cx, cy)
            eig = 2.0 * (CX + CY - 2.0)
            eig[0, 0] = 1.0  # avoid division by zero
            phi_dct = div_dct / eig
            phi_dct[0, 0] = 0.0  # remove piston
            phi = fftpack.idct(
                fftpack.idct(phi_dct, axis=1, norm="ortho"),
                axis=0, norm="ortho"
            )
            phi[~valid_div] = xp.nan
            return phi  

        from cupyx.scipy.ndimage import zoom

        def resize_phase_nan_cp(phi, NY, NX, order=1):
            phi = phi.reshape(phi.shape[-2], phi.shape[-1])

            # Mask of valid values
            mask = ~cp.isnan(phi)

            # Replace NaNs with 0
            phi_filled = cp.where(mask, phi, 0)

            # Compute zoom factors
            zoom_y = NY / phi.shape[0]
            zoom_x = NX / phi.shape[1]

            # Interpolate data and mask
            phi_zoom = zoom(phi_filled, (zoom_y, zoom_x), order=order)
            mask_zoom = zoom(mask.astype(cp.float32), (zoom_y, zoom_x), order=order)

            # Avoid division by zero
            eps = 1e-6
            phi_resized = phi_zoom / (mask_zoom + eps)

            # Optional: reintroduce NaNs where confidence is too low
            phi_resized = cp.where(mask_zoom > 0.1, phi_resized, cp.nan)

            return phi_resized.reshape(NY, NX)
        
        def resize_phase_nan_np(phi, NY, NX):
            y, x = np.indices(phi.shape)
            mask = np.isfinite(phi)

            y_new, x_new = np.meshgrid(
                np.linspace(0, phi.shape[0]-1, NY),
                np.linspace(0, phi.shape[1]-1, NX),
                indexing='ij'
            )

            return griddata(
                (y[mask], x[mask]),
                phi[mask],
                (y_new, x_new),
                method='cubic'
            )
            
        if self.xp == np:
            resize_phase_nan = resize_phase_nan_np
        elif self.xp == cp:
            resize_phase_nan = resize_phase_nan_cp
               
        slopes_y = shifts_y * wavelength # /(pixel_pitch_y*(ny//nysubabs)) same as before *(pixel_pitch_y*(ny//nysubabs)) the pitch between subaps before integration
        slopes_x = shifts_x * wavelength 
        southwell_fourier_phase = southwell_fourier_nan(slopes_y, slopes_x) * (2*xp.pi) / wavelength
        southwell_fourier_phase = resize_phase_nan(southwell_fourier_phase, ny, nx)
        return southwell_fourier_phase
    
    def _fresnel_transform_phase(self, frames, phase_term):
        return self.fft.fftshift(
            self.fft.fft2(frames * self.kernels["Fresnel"] * phase_term, axes=(-1, -2), norm="ortho"), axes=(-1, -2)
        )

    # ------------------------------------------------------------
    # Render tools for debug and visualization
    # ------------------------------------------------------------
    
    class FigureCanvasAgg:
        def __init__(self, fig):
            self.fig = fig
            self.canvas = FigureCanvasAgg(fig)

        def draw(self):
            self.canvas.draw()

        def get_image(self):
            width, height = self.canvas.get_width_height()
            return np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        
    class PhasePlotter:
        def __init__(self, title="Wavefront Phase", cmap="twilight", figsize=(6, 6), dpi=100):
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            self.canvas = FigureCanvasAgg(self.fig)
            self.title = title
            self.cmap = cmap

        def plot(self, phase):
            self.ax.clear()
            self.ax.set_title(self.title)
            im = self.ax.imshow((phase.get() + np.pi) % (2*np.pi) - np.pi, cmap=self.cmap)
            self.fig.colorbar(im, ax=self.ax, fraction=0.029, pad=0.04)
            self.ax.set_aspect('equal')
            self.canvas.draw()
            img = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (3,))
            return img
        
    class ShiftsPlotter:
        def __init__(self, title="Wavefront Slopes from Sub-aperture Shifts", figsize=(8, 6), dpi=100, scale=50):
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            self.canvas = FigureCanvasAgg(self.fig)
            self.title = title
            self.scale = scale

        def plot(self, shifts_y, shifts_x):
            self.ax.clear()
            ny_subabs, nx_subabs = shifts_y.shape
            X, Y = np.meshgrid(np.arange(nx_subabs), np.arange(ny_subabs))
            self.ax.quiver(X, Y, shifts_x.get(), shifts_y.get(), scale=self.scale)
            self.ax.set_title(self.title)
            self.ax.set_xlabel('Sub-aperture X Index')
            self.ax.set_ylabel('Sub-aperture Y Index')
            self.ax.set_xlim(-0.5, nx_subabs - 0.5)
            self.ax.set_ylim(-0.5, ny_subabs - 0.5)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_aspect('equal')
            self.canvas.draw()
            img = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (3,))
            return img
        
        class SubapertureMontagePlotter:
            def __init__(self, figsize=(12, 8), dpi=100):
                self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
                self.canvas = FigureCanvasAgg(self.fig)

            def plot(self, U_subaps):
                rows = []
                for iy in range(U_subaps.shape[0]):
                    row_imgs = [self.xp.asnumpy(U_subaps[iy, ix]) for ix in range(U_subaps.shape[1])]
                    rows.append(np.hstack(row_imgs))
                montage_img = np.vstack(rows)
                self.ax.clear()
                self.ax.imshow(montage_img, cmap='gray')
                self.ax.axis('off')
                self.ax.set_title('Montage of Sub-apertures')
                self.canvas.draw()
                img = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (3,))
                return img

        class DebugQueue:
            def __init__(self, maxsize=32, drop_if_full=True, worker_fn=None):
                self.q = queue.Queue(maxsize=maxsize)
                self.drop_if_full = drop_if_full
                self.worker_fn = worker_fn or self._default_worker
                self._stop_event = threading.Event()
                self.thread = threading.Thread(target=self._run, daemon=True)

            def start(self):
                self.thread.start()

            def stop(self):
                self._stop_event.set()
                self.thread.join()

            def push(self, item):
                try:
                    self.q.put(item, block=not self.drop_if_full, timeout=0.01)
                except queue.Full:
                    # Drop silently (debug must not block pipeline)
                    pass

            def _run(self):
                while not self._stop_event.is_set():
                    try:
                        item = self.q.get(timeout=0.1)
                        self.worker_fn(item)
                    except queue.Empty:
                        continue

            def _default_worker(self, item):
                # Fallback behavior: just print metadata
                print(f"[DEBUG] Received item: {type(item)}")
    # ------------------------------------------------------------
    # One batch full pipeline
    # ------------------------------------------------------------

    def render_moments(self, parameters, frames = None):

        if frames is None:
            frames = self.read_frames(parameters["first_frame"], parameters["batch_size"])
        if frames is None:
            raise RuntimeError("Could not read frames properly")

        nt, ny, nx = frames.shape
        
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel":
            if (not "Fresnel" in self.kernels):
                self._build_fresnel_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx)
            
            U_subaps = self._shack_hartmann_constructsubapsimages(frames, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], parameters["z"], parameters["low_freq"], parameters["high_freq"], parameters["sampling_freq"], parameters["batch_size"], parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"], parameters["svd_threshold"]) # construct small images from the sub apertures of the Shack-Hartmann sensor
            
            def plot_subaps(U_subaps, nx_subabs, ny_subabs):
                rows = []
                for iy in range(ny_subabs):
                    row_imgs = [self.xp.asnumpy(U_subaps[iy, ix]) for ix in range(nx_subabs)]
                    rows.append(np.hstack(row_imgs))  # horizontally stack each row
                montage_img = np.vstack(rows)  # vertically stack all rows
                return montage_img
                # Display montage
                # plt.figure(figsize=(12, 8))
                # plt.imshow(montage_img, cmap='gray')
                # plt.axis('off')  # remove axes
                # plt.title('Montage of Sub-apertures')
                # plt.show()
                
            montage_img = plot_subaps(U_subaps, parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"])
            
            shifts_y, shifts_x = self._shack_hartmann_displacement_calculation(U_subaps, self.xp, ref = None) # get the shifts in pixels in the subapertures images
            nysubabs, nxsubabs = shifts_y.shape
            
            def plot_shifts(shifts_y, shifts_x, nx_subabs, ny_subabs, title="Wavefront Slopes from Sub-aperture Shifts", scale=50):
                fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                canvas = FigureCanvasAgg(fig)
                
                # Derive dimensions from figure
                width = int(fig.get_figwidth() * fig.dpi)
                height = int(fig.get_figheight() * fig.dpi)
                
                X, Y = np.meshgrid(np.arange(nx_subabs), np.arange(ny_subabs))
                ax.quiver(X, Y, shifts_x.get(), shifts_y.get(), scale=scale)
                ax.set_title(title)
                ax.set_xlabel('Sub-aperture X Index')
                ax.set_ylabel('Sub-aperture Y Index')
                ax.set_xlim(-0.5, nx_subabs - 0.5)
                ax.set_ylim(-0.5, ny_subabs - 0.5)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_aspect('equal')
                
                canvas.draw()
                img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
                plt.close(fig)  # Important: free memory
                
                return img
            
            shifts_img = plot_shifts(shifts_y, shifts_x, parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"], title = "Wavefront Slopes from Sub-aperture Shifts", scale=50)
            if parameters["shack_hartmann_zernike_fit"] :
                coefs, phase = self._shack_hartmann_zernike(ny, nx, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], shifts_y, shifts_x, parameters["shack_hartmann_zernike_fit_modes"]) # fit the shifts to Zernike polynomials to get the wavefront phase
            elif parameters["shack_hartmann_southwell_phase_integration "] :
                phase = self._shack_hartmann_southwell(ny, nx, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], shifts_y, shifts_x)
            # display the phase plt show phase twilight 
            def plot_phase(phase, title="Wavefront Phase"):
                fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
                canvas = FigureCanvasAgg(fig)
                
                # Derive dimensions from figure
                width = int(fig.get_figwidth() * fig.dpi)
                height = int(fig.get_figheight() * fig.dpi)
                
                ax.set_title(title)
                im = ax.imshow((phase.get() + np.pi) % (2*np.pi) - np.pi, cmap="twilight")
                fig.colorbar(im, ax=ax, fraction=0.029, pad=0.04)  # Use fig.colorbar with ax parameter
                ax.set_aspect('equal')
                
                canvas.draw()
                img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
                plt.close(fig)
                
                return img
            
            phase_img = plot_phase(phase, title="Wavefront Phase")
            
            phase_term = self.xp.exp(- 1j * phase) 
            phase_term = self.xp.nan_to_num(phase_term, nan=0.0) # completely mask the nan zone where the phase could'nt be estimated
            holograms = self._fresnel_transform_phase(frames, phase_term)
            hologramsnotfixed = self._fresnel_transform(frames)
        elif parameters["spatial_propagation"] == "Fresnel":
            if (not "Fresnel" in self.kernels):
                self._build_fresnel_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx)
            holograms = self._fresnel_transform(frames)
        elif parameters["spatial_propagation"] == "AngularSpectrum":
            if (not "AngularSpectrum" in self.kernels):
                self._build_angular_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx)
            holograms = self._angular_spectrum_transform(frames)
            
        holograms_f = self._svd_filter(holograms, parameters["svd_threshold"])

        spectrum_f = self._fourier_time_transform(holograms_f)

        # idxs, freqs = self._frequency_symmetric_filtering(frames.shape[-1], parameters["sampling_freq"], parameters["low_freq"])
        idxs, freqs = self._old_frequency_symmetric_filtering(frames.shape[0], parameters["sampling_freq"], parameters["low_freq"], parameters["high_freq"])

        psd = self.xp.abs(spectrum_f[idxs,:,:]) ** 2

        M0 = self._moment(psd, freqs, 0)
        M1 = self._moment(psd, freqs, 1)
        M2 = self._moment(psd, freqs, 2)
        
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel":
            hologramsnotfixed
            hologramsnotfixed_f = self._svd_filter(hologramsnotfixed, parameters["svd_threshold"])
            spectrumnotfixed_f = self._fourier_time_transform(hologramsnotfixed_f)
            psdnotfixed = self.xp.abs(spectrumnotfixed_f[idxs,:,:]) ** 2
            M0notfixed = self._moment(psdnotfixed, freqs, 0)
            
        c = self._to_numpy(coefs) if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and parameters["shack_hartmann_zernike_fit"] else None
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel":
            res = (M0, M1, M2, montage_img, shifts_img, phase_img, self._to_numpy(M0notfixed), [c])
        else:
            res = (M0, M1, M2, None, None, None, None, None)
        return res

    # ------------------------------------------------------------
    # Full video processing
    # ------------------------------------------------------------

    def process_moments_(self, parameters, h5_path = None, mp4_path = None, return_numpy = False, holodoppler_path = False):
        
        
        batch_size = parameters["batch_size"]
        batch_stride = parameters["batch_stride"]
        first_frame = parameters["first_frame"]
        end_frame = parameters["end_frame"]
        if end_frame <= 0:
            if self.ext == ".holo":
                end_frame = self.file_header["num_frames"]
            elif self.ext == ".cine":
                end_frame = self.cine_metadata.ImageCount
        
        # please do not remove it is good
        if batch_stride >= (end_frame-first_frame):
            if batch_size <= (end_frame-first_frame):
                num_batch = 1
            else:
                num_batch = 0
        else:
            num_batch = int((end_frame-first_frame) / batch_stride)

        out_list = []
        debug_list = [None for _ in range(num_batch)]

        if num_batch <= 0:
            return None
        
        
        if parameters["image_registration"]:
            frames = self.read_frames(first_frame, parameters["batch_size_registration"]) # the first frame to be rendered
            M0_reg, _, _, _, _, _, _, _ = self.render_moments(parameters, frames = frames)
            M0_reg = self._flatfield(M0_reg, parameters["registration_flatfield_gw"])
            reg_list = []
            
        if self.backend == "cupy_ramdisk":
            import time

            print("=== Preloading all batches into RAM ===")
            t0 = time.perf_counter()

            # --- Read ALL batches into RAM upfront in parallel ---
            from concurrent.futures import ThreadPoolExecutor
            from queue import Queue

            NUM_IO_WORKERS = 32  # go wide, disk is the bottleneck

            fid_pool = Queue()
            worker_fids = [open(self.file_path, "rb") for _ in range(NUM_IO_WORKERS)]
            for fid in worker_fids:
                fid_pool.put(fid)

            batch_list = [
                (first_frame + i * parameters["batch_stride"], parameters["batch_size"])
                for i in range(num_batch)
            ]

            def read_batch(args):
                idx, (ff, bs) = args
                fid = fid_pool.get()
                try:
                    return idx, self.read_frames(ff, bs, fid=fid)
                finally:
                    fid_pool.put(fid)

            with ThreadPoolExecutor(max_workers=NUM_IO_WORKERS) as executor:
                results = list(executor.map(read_batch, enumerate(batch_list)))

            # Sort by index, guaranteed order
            all_frames = [frames for _, frames in sorted(results, key=lambda x: x[0])]

            print(f"Preload done in {time.perf_counter() - t0:.2f}s — now GPU loop is pure compute")

            for fid in worker_fids:
                fid.close()

            # --- GPU loop: zero disk I/O, pure compute ---
            stream_h2d     = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)

            with stream_h2d:
                d_frames_next = cp.asarray(all_frames[0])
            stream_h2d.synchronize()

            for i in tqdm(range(num_batch)):
                d_frames = d_frames_next

                if i + 1 < num_batch:
                    with stream_h2d:
                        d_frames_next = cp.asarray(all_frames[i + 1])
                    # no sync needed — compute stream will naturally be behind

                with stream_compute:
                    res = self.render_moments(parameters, frames=d_frames)

                if res is None:
                    break

                M0, M1, M2, *debug_imgs = res

                with stream_compute:
                    if parameters["image_registration"]:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list.append((shift_y, shift_x))

                stream_compute.synchronize()
                out_list.append(cp.stack([M0, M1, M2], axis=2))
                debug_list[i] = debug_imgs

            stream_h2d.synchronize()
            cp.cuda.Device().synchronize()
            
        if self.backend == "cupy_diagnostic":
            print("Running in diagnostic mode with CuPy backend. This will report timing for each step of the pipeline to identify bottlenecks.")
            import time
            from collections import defaultdict

            timings = defaultdict(list)

            def cuda_sync_time(stream, label):
                """Synchronize a stream and return elapsed time."""
                t0 = time.perf_counter()
                stream.synchronize()
                t1 = time.perf_counter()
                timings[label].append(t1 - t0)
                return t1 - t0

            stream_h2d     = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)

            # --- Prime first batch ---
            t0 = time.perf_counter()
            frames_next = self.read_frames(first_frame, parameters["batch_size"])
            timings["disk_read"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            with stream_h2d:
                d_frames_next = cp.asarray(frames_next)
            stream_h2d.synchronize()
            timings["h2d_transfer"].append(time.perf_counter() - t0)

            for i in tqdm(range(num_batch)):
                d_frames = d_frames_next

                # --- Disk read ---
                if i + 1 < num_batch:
                    t0 = time.perf_counter()
                    frames_next = self.read_frames(
                        first_frame + (i + 1) * parameters["batch_stride"],
                        parameters["batch_size"]
                    )
                    timings["disk_read"].append(time.perf_counter() - t0)

                    # --- H2D transfer ---
                    t0 = time.perf_counter()
                    with stream_h2d:
                        d_frames_next = cp.asarray(frames_next)
                    stream_h2d.synchronize()
                    timings["h2d_transfer"].append(time.perf_counter() - t0)

                # --- Compute ---
                t0 = time.perf_counter()
                with stream_compute:
                    res = self.render_moments(parameters, frames=d_frames)
                stream_compute.synchronize()
                timings["compute"].append(time.perf_counter() - t0)

                if res is None:
                    break

                M0, M1, M2 = res

                # --- Registration ---
                if parameters["image_registration"]:
                    t0 = time.perf_counter()
                    with stream_compute:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list.append((shift_y, shift_x))
                    stream_compute.synchronize()
                    timings["registration"].append(time.perf_counter() - t0)

                # --- Stack ---
                t0 = time.perf_counter()
                out_list.append(cp.stack([M0, M1, M2], axis=2))
                timings["stack"].append(time.perf_counter() - t0)

            stream_h2d.synchronize()
            cp.cuda.Device().synchronize()

            # --- Report ---
            import numpy as np
            print("\n=== BOTTLENECK REPORT ===")
            total_time = sum(sum(v) for v in timings.values())
            for label, times in sorted(timings.items(), key=lambda x: -sum(x[1])):
                arr = np.array(times)
                pct = 100 * arr.sum() / total_time
                print(
                    f"{label:20s} | "
                    f"total: {arr.sum():.3f}s ({pct:5.1f}%) | "
                    f"mean: {arr.mean()*1000:.1f}ms | "
                    f"max: {arr.max()*1000:.1f}ms | "
                    f"min: {arr.min()*1000:.1f}ms | "
                    f"std: {arr.std()*1000:.1f}ms"
                )
            print(f"{'TOTAL':20s} | {total_time:.3f}s")
            print("=========================\n")

        if self.backend == "cupy":
            stream_h2d = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)

            # --- Prefetch first batch ---
            frames_next = self.read_frames(
                first_frame,
                parameters["batch_size"]
            )

            with stream_h2d:
                d_frames_next = cp.asarray(frames_next)  # async if pinned

            for i in tqdm(range(num_batch)):

                # Swap buffers
                d_frames = d_frames_next

                # --- Prefetch next batch (CPU side) ---
                if i + 1 < num_batch:
                    
                    with stream_h2d:
                        d_frames_next = self.read_frames(
                        first_frame + (i + 1) * parameters["batch_stride"],
                        parameters["batch_size"]
                    )

                # --- Compute current batch ---
                with stream_compute:
                    res = self.render_moments(parameters, frames=d_frames)

                if res is None:
                    break

                M0, M1, M2, *debug_imgs = res

                # --- Register current batch ---
                with stream_compute:
                    if parameters["image_registration"]:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list.append((shift_y, shift_x))

                stream_compute.synchronize()
                
                out_list.append(
                    cp.stack([M0, M1, M2], axis=2)
                )
                debug_list[i] = debug_imgs

            # Ensure transfers complete
            stream_h2d.synchronize()

            cp.cuda.Device().synchronize()
        else:
            for i in tqdm(range(num_batch)):

                try:

                    frames = self.read_frames(first_frame + i * parameters["batch_stride"] , parameters["batch_size"])

                    res = self.render_moments(parameters, frames = frames)

                    if res is None:
                        break

                    M0, M1, M2, *debug_imgs = res

                    if parameters["image_registration"]:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list.append((shift_y, shift_x))

                    out_list.append(
                        self.xp.stack([M0, M1, M2], axis=2)
                    )
                    debug_list[i] = debug_imgs

                except Exception:
                    traceback.print_exc()
                    break

            if len(out_list) == 0:
                return None

        vid = self.xp.stack(out_list, axis=3)
        if debug_list[0][0] is not None:
            streams = [[], [], [], [], []]
            for tup in debug_list:
                if tup is not None:
                    for i, img in enumerate(tup):
                        streams[i].append(img if img is not None else np.zeros_like(streams[i][0] if streams[i] else img))
            import numpy as np
            vid_debug = [np.stack(stream, axis=2) for stream in streams if stream]
        else:
            vid_debug = None
            
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel":
            zernike_coefs = self._to_numpy(vid_debug[4]) if vid_debug and len(vid_debug) > 4 else None
        else:
            zernike_coefs = None

        if parameters["square"]:
            m = max(vid.shape[0], vid.shape[1])
            # vid = self.resize_fft2_slicewise(vid, m, m)
            import numpy as np
            vid = self._to_numpy(vid).astype(np.float64)
            vid = self._resize(vid, m, m)
            
        if parameters["transpose"]:
            vid = self.xp.transpose(vid, axes=(1, 0, 2, 3))
            
        if parameters["flip_x"]:
            vid = self.xp.flip(vid, axis=1)
        
        if parameters["flip_y"]:
            vid = self.xp.flip(vid, axis=0)

        if (h5_path is not None) or (mp4_path is not None) or holodoppler_path:
            vid = self._to_numpy(vid)
            
            
        def save_to_h5path(h5_path, v, parameters, reg_list = None, zernike_coefs = None):
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("moment0", data=v[:, :, :, 0])
                f.create_dataset("moment1", data=v[:, :, :, 1])
                f.create_dataset("moment2", data=v[:, :, :, 2])
                f.create_dataset("HD_parameters", data=json.dumps(parameters))
                if parameters["image_registration"]:
                    f.create_dataset("registration", data=self._to_numpy(self.xp.array(reg_list)))
                    
                if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and zernike_coefs is not None:
                    f.create_dataset("zernike_coefs_radians", data=self._to_numpy((zernike_coefs).astype(np.float64)))

                def json_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    raise TypeError(f"Type {type(obj)} not serializable")

                if self.ext == ".holo":
                    f.create_dataset("holo_header", data=json.dumps(self.file_header, default=json_serializer))
                    f.create_dataset("holo_footer", data=json.dumps(self.file_footer, default=json_serializer))
                elif self.ext == ".cine":
                    f.create_dataset("cine_metadata", data=json.dumps(self.cine_metadata_json, default=json_serializer))

        if h5_path is not None:
            save_to_h5path(h5_path, np.permute_dims(v, (3, 0, 1, 2)), parameters, reg_list if parameters["image_registration"] else None, zernike_coefs)

        if mp4_path is not None:
            pass
        if holodoppler_path is True:
            # make the new directory at the same level as the input file with its name then _HD{idx} where idx is the number of existing holodoppler output directories for this file
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            dir_name = f"{base_name}_HD_"
            parent_dir = os.path.dirname(self.file_path)
            existing_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(dir_name)]
            idx = len(existing_dirs) + 1    
            holodoppler_dir_name = f"{dir_name}{idx}"
            holodoppler_path = os.path.join(parent_dir, holodoppler_dir_name)
            os.makedirs(holodoppler_path, exist_ok=True)
            # make a png, mp4 json and h5 sub directories with their respective content
            png_dir = os.path.join((holodoppler_path), "png")
            mp4_dir = os.path.join((holodoppler_path), "mp4")
            json_dir = os.path.join((holodoppler_path), "json")
            h5_dir = os.path.join((holodoppler_path), "raw")
            print(f"Saving output to: {holodoppler_path}")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(mp4_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(h5_dir, exist_ok=True)
            # save pngs
            for i in range(vid.shape[2]):
                plt.imsave(os.path.join(png_dir, f"moment_{i}.png"), np.mean(vid[:, :, i, :],axis=2), cmap="gray")
            # save m0 as mp4 and avi
            vid_norm = vid[:, :, 0, :]  # shape (H, W, T)
            vid_norm = vid_norm.astype(np.float32)
            vid_norm = (vid_norm - vid_norm.min()) / (vid_norm.max() - vid_norm.min())
            vid_norm = (vid_norm * 255).astype(np.uint8)
            # save mp4
            height, width = vid.shape[0], vid.shape[1]
            duration = 1/parameters["sampling_freq"] * (end_frame-first_frame) 
            fps = num_batch / duration
            out_mp4 = cv2.VideoWriter(os.path.join(mp4_dir, "moment_0.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width, height),isColor=False)
            for i in range(num_batch):
                frame = vid_norm[:, :, i]
                out_mp4.write(frame)
            out_mp4.release()
            # save avi
            out_avi = cv2.VideoWriter(os.path.join(mp4_dir, "moment_0.avi"),cv2.VideoWriter_fourcc(*'XVID'),fps,(width, height),isColor=False)
            for i in range(num_batch):
                frame = vid_norm[:, :, i]
                out_avi.write(frame)
            out_avi.release()
            #save debug mp4s if they exist
            if vid_debug is not None:
                for idx, v in enumerate(vid_debug):
                    print(v.shape)
                    height, width = v.shape[0], v.shape[1]
                    isColor = (v.ndim == 4 and v.shape[3] == 3)
                    if not isColor and (v.max() - v.min()) > 0:
                        v = v.astype(np.float32)
                        v = (v - v.min()) / (v.max() - v.min())
                        v = (v * 255).astype(np.uint8)
                    duration = 1 / parameters["sampling_freq"] * (end_frame - first_frame)
                    fps = num_batch / duration
                    out_avi = cv2.VideoWriter(
                        os.path.join(mp4_dir, f"debug_{idx}.avi"),
                        cv2.VideoWriter_fourcc(*'XVID'),
                        fps,
                        (width, height),
                        isColor=isColor
                    )
                    for i in range(num_batch):
                        
                        if v.ndim == 3:
                            frame = v[:, :, i]  # (H, W)
                        else:
                            frame = v[:, :, i, :]  # (H, W, 3)
                        out_avi.write(frame)
                    out_avi.release()
            # save json
            with open(os.path.join(json_dir, "parameters.json"), "w") as f:
                json.dump(parameters, f, indent=4)
            # save h5
            save_to_h5path(os.path.join(h5_dir, f"{holodoppler_dir_name}_output.h5"), np.permute_dims(vid, (3, 1, 0, 2)), parameters, reg_list if parameters["image_registration"] else None, zernike_coefs)
            # add a version.txt file with the version of the holodoppler pipeline used
            with open(os.path.join(holodoppler_path, "version.txt"), "w") as f:
                f.write(f"Python:\n")
                f.write(f"Holodoppler pipeline version: {self.pipeline_version}\n")
                f.write(f"Holodoppler backend: {self.backend}\n") 

        if return_numpy:
            return self._to_numpy(vid)

        return vid
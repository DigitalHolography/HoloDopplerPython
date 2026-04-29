import matplotlib
import numpy as np
import traceback
import h5py
import scipy
from tqdm import tqdm
import cv2
import json
import os
import threading
import queue
import time
import cinereader
from cupy.cuda.nvtx import RangePush, RangePop
import time
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
from scipy.ndimage import gaussian_filter1d
from matlab_imresize.imresize import imresize
import scipy.ndimage as np_ndi
from scipy.interpolate import griddata
import scipy.fftpack as np_fftpack
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

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

        if "cupy" in self.backend:
            if not _cupy_available:
                raise RuntimeError("CuPy backend requested but CuPy is not available.")
            self.xp = cp
            self.fft = cp_fft
            self.gaussian_filter = cp_gaussian_filter
            self.ndi = cp_ndi
            self.fftpack = cp_fft
        else:
            self.backend = self.backend
            self.xp = np
            self.fft = np_fft
            self.gaussian_filter = np_gaussian_filter
            self.ndi = np_ndi
            self.fftpack = np_fftpack
            
    def _to_backend(self, arr):
        if "cupy" in self.backend:
            return cp.asarray(arr)
        return arr

    def _to_numpy(self, arr):
        if "cupy" in self.backend:
            if isinstance(arr, cp.ndarray):
                return arr.get() 
            return arr  # already NumPy or other type
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
            
            return self._to_backend(out).astype(self.xp.float32)

        except Exception:
            traceback.print_exc()
            return None

    # ------------------------------------------------------------
    # Calculation kernels
    # ------------------------------------------------------------

    def _build_fresnel_kernel_in(self, z, pixel_pitch, wavelength, ny, nx):

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

        self.kernels["Fresnel_in"] = kernel[xp.newaxis, ...]
    
    def _build_fresnel_kernel_out(self, z, pixel_pitch, wavelength, ny, nx):

        if isinstance(pixel_pitch, (float, int)):
            pixel_pitch = (pixel_pitch, pixel_pitch)

        ppy, ppx = pixel_pitch

        xp = self.xp 

        fx = xp.fft.fftfreq(nx, d=ppx)
        fx = xp.fft.fftshift(fx)
        fy = xp.fft.fftfreq(ny, d=ppy)
        fy = xp.fft.fftshift(fy)
        FX, FY = xp.meshgrid(fx, fy)

        X = wavelength * z * FX
        Y = wavelength * z * FY
        k = 2 * xp.pi / wavelength
        phase = 1j * xp.pi / (wavelength * z) * (X**2 + Y**2)
        kernel = (xp.exp(1j * k * z) / (1j * wavelength * z) * xp.exp(phase)).astype(xp.complex64)
        self.kernels["Fresnel_out"] = kernel[xp.newaxis, ...]

    def _build_fresnel_kernel(self, z, pixel_pitch, wavelength, ny, nx, zero_padding = None):
        self._build_fresnel_kernel_in(z, pixel_pitch, wavelength, ny, nx)
        self._build_fresnel_kernel_out(z, pixel_pitch, wavelength, ny, nx)
        
        if zero_padding:
            self.kernels["Fresnel_in"] = self.pad_array_centrally(self.kernels["Fresnel_in"], zero_padding)
            self.kernels["Fresnel_out"] = self.xp.ones_like(self.kernels["Fresnel_in"])


    def _build_angular_kernel(self,
                              z,
                              pixel_pitch,
                              wavelength, ny, nx, zero_padding = None):

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
        
        if zero_padding:
            self.kernels["AngularSpectrum"] = self.pad_array_centrally(self.kernels["AngularSpectrum"], zero_padding)

    # ------------------------------------------------------------
    # Calculation processing
    # ------------------------------------------------------------
    
    def pad_array_centrally(self, arr, new_shape):
        
        xp = self.xp
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ny, nx = arr.shape[-2:]
        new_ny, new_nx = new_shape

        if new_ny < ny or new_nx < nx:
            raise ValueError("new_shape must be greater than or equal to the current last two dimensions")

        pad_y0 = (new_ny - ny) // 2
        pad_y1 = new_ny - ny - pad_y0
        pad_x0 = (new_nx - nx) // 2
        pad_x1 = new_nx - nx - pad_x0

        pad_width = [(0, 0)] * arr.ndim
        pad_width[-2] = (pad_y0, pad_y1)
        pad_width[-1] = (pad_x0, pad_x1)

        return xp.pad(arr, pad_width, mode="constant")
    
    def crop_array_centrally(self, arr, target_shape):
        xp = self.xp

        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)

        ny, nx = arr.shape[-2:]
        tgt_ny, tgt_nx = target_shape

        if tgt_ny > ny or tgt_nx > nx:
            raise ValueError("target_shape must be smaller than or equal to current shape")

        crop_y0 = (ny - tgt_ny) // 2
        crop_y1 = crop_y0 + tgt_ny
        crop_x0 = (nx - tgt_nx) // 2
        crop_x1 = crop_x0 + tgt_nx

        slices = [slice(None)] * arr.ndim
        slices[-2] = slice(crop_y0, crop_y1)
        slices[-1] = slice(crop_x0, crop_x1)

        return arr[tuple(slices)]

    def _fresnel_transform(self, frames, zero_padding = False):
        
        if zero_padding:
            frames = self.pad_array_centrally(frames, zero_padding)

        if self.pipeline_version == "latest" : 
            return self.fft.fftshift(
                self.fft.fft2(frames *self.kernels["Fresnel_in"], axes=(-1, -2), norm="ortho"), axes=(-1, -2)
            ) *self.kernels["Fresnel_out"]
        else :
            return self.fft.fftshift(
                self.fft.fft2(frames *self.kernels["Fresnel_in"], axes=(-1, -2), norm="ortho"), axes=(-1, -2)
            )
    
    def _angular_spectrum_transform(self, frames, zero_padding = False):
        
        if zero_padding:
            frames = self.pad_array_centrally(frames, zero_padding)

        tmp = self.fft.fft2(frames,axes=(-1, -2), norm="ortho") * self.fft.fftshift(self.kernels["AngularSpectrum"],axes=(-1, -2))

        return self.fft.ifft2(tmp,axes=(-1, -2), norm="ortho")

    def _fourier_time_transform(self, H):

        return self.fft.fft(H, axis=0, norm="ortho") 

    # ------------------------------------------------------------
    # SVD filtering
    # (Eigen decomposition of temporal covariance)
    # ------------------------------------------------------------

    def _svd_filter(self, H, svd_threshold):
        RangePush("SVD filtering")
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

        H2 -= H2 @ Vt @ Vt.conj().T
        RangePop()
        return H2.T.reshape(sz)
    
    @staticmethod
    def randomized_svd(H2, k, xp):
        n_random = k + 5

        # random projection
        Omega = xp.random.randn(H2.shape[1], n_random, dtype=H2.dtype)
        Y = H2 @ Omega

        # orthonormalize
        Q, _ = xp.linalg.qr(Y)

        # smaller SVD
        B = Q.conj().T @ H2
        Ub, S, Vh = xp.linalg.svd(B, full_matrices=False)

        U = Q @ Ub
        return U, S, Vh
    
    def _svd_filter2(self, H, svd_threshold):
        print(H.dtype)
        RangePush("SVD filtering")
        xp = self.xp

        if svd_threshold < 0:
            return H

        sz = H.shape

        # reshape: (pixels, channels)
        H2 = H.reshape(sz[0], -1).T   # (131072, 320)

        # --- SVD (GPU optimized) ---
        U, S, Vh = xp.linalg.svd(H2, full_matrices=False)

        # keep only leading components
        Vh_t = Vh[:svd_threshold]

        # projection (tissue)

        H2 -= H2 @ Vh_t.T @ Vh_t

        RangePop()
        return H2.T.reshape(sz)
    
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
        
        # _fixed = self.gaussian_filter(fixed,1.5)
        # _moving = self.gaussian_filter(moving,1.5)

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

    def _svd_filter_shack_hartmann(self, U_subaps, svd_threshold):
        """Batched SVD filter for all subapertures at once (ny_sub, nx_sub, sub_ny, sub_nx, Nz)."""
        xp = self.xp
        if svd_threshold < 0:
            return U_subaps

        ny_s, nx_s, sub_ny, sub_nx, nz = U_subaps.shape
        B = ny_s * nx_s

        # (B, ny*nx, nz)
        H2 = U_subaps.reshape(B, sub_ny * sub_nx, nz)

        eps = 1e-12
        # Batched covariance (B, nz, nz)
        cov = xp.einsum('bpi,bpj->bij', H2.conj(), H2) + eps * xp.eye(nz, dtype=H2.dtype)

        # eigh per subaperture (CuPy does not support batched eigh)
        V_list = []
        for b in range(B):
            _, Vb = xp.linalg.eigh(cov[b])   # ascending order, (nz, nz)
            V_list.append(Vb[:, ::-1])        # flip to descending, keep top-k
        Vt = xp.stack([Vb[:, :svd_threshold] for Vb in V_list], axis=0)  # (B, nz, k)

        # Project out tissue: H2 - (H2 @ Vt) @ Vt^H
        H2_Vt = xp.einsum('bpi,bik->bpk', H2, Vt)            # (B, sub_ny*sub_nx, k)
        proj   = xp.einsum('bpk,bjk->bpj', H2_Vt, Vt.conj()) # (B, sub_ny*sub_nx, nz)

        return (H2 - proj).reshape(ny_s, nx_s, sub_ny, sub_nx, nz)

    def _shack_hartmann_constructsubapsimages(self, U0, dx, dy, wavelength, z_prop, f0, f1, fs,
                                           time_window, nx_subabs, ny_subabs, svd_threshold):
        RangePush("Shack-Hartmann subaperture construction")
        xp = self.xp
        
        Nz, Ny, Nx = U0.shape
        sub_ny, sub_nx = Ny // ny_subabs, Nx // nx_subabs

        idxs, _ = self._frequency_symmetric_filtering(time_window, fs, f0, high_freq=f1)

        # --- Fresnel kernel (cached) ---
        if "Fresnel_in" not in self.kernels:
            self._build_fresnel_kernel(z_prop, (dy, dx), wavelength, Ny, Nx)
            
        Qin = self.crop_array_centrally(self.kernels["Fresnel_in"], (Ny, Nx))

        # Fresnel-multiply once on full field
        U_prop_qin = U0 * Qin  # (Nz, Ny, Nx)

        # --- Crop to central region that tiles exactly into subapertures ---
        crop_ny, crop_nx = sub_ny * ny_subabs, sub_nx * nx_subabs
        y0, x0 = (Ny - crop_ny) // 2, (Nx - crop_nx) // 2
        U_prop_qin = U_prop_qin[:, y0:y0 + crop_ny, x0:x0 + crop_nx]  # (Nz, crop_ny, crop_nx)

        # Reshape into subapertures: (ny_s, nx_s, sub_ny, sub_nx, Nz)
        U_subap_all = (U_prop_qin
                    .reshape(Nz, ny_subabs, sub_ny, nx_subabs, sub_nx)
                    .transpose(1, 3, 2, 4, 0))

        # Batched SVD filter across all subapertures
        U_subap_all = self._svd_filter_shack_hartmann(U_subap_all, svd_threshold)
        # (ny_s, nx_s, sub_ny, sub_nx, Nz) -> (B, Nz, sub_ny, sub_nx)
        B = ny_subabs * nx_subabs
        U_subap_all = U_subap_all.reshape(B, sub_ny, sub_nx, Nz).transpose(0, 3, 1, 2)

        # Batched FFT2 + temporal FFT
        U_f2 = xp.fft.fftshift(xp.fft.fft2(U_subap_all, axes=(-2, -1)), axes=(-2, -1))  # (B, Nz, sub_ny, sub_nx)
        U_ft = xp.fft.fft(U_f2, axis=1)[:, idxs, :, :]                                  # (B, |idxs|, sub_ny, sub_nx)

        # Power spectrum mean over frequency band
        M0 = xp.mean(xp.abs(U_ft) ** 2, axis=1)                                          # (B, sub_ny, sub_nx)
        U_subaps = M0.reshape(ny_subabs, nx_subabs, sub_ny, sub_nx).astype(xp.float32)
        RangePop()
        return U_subaps
    
    def _shack_hartmann_constructsubapsimages_angular_spectrum(self,U0,dx,dy,wavelength,z_prop,f0,f1,fs,time_window,nx_subabs,ny_subabs,svd_threshold):

        xp = self.xp
        Nz, Ny, Nx = U0.shape

        sub_ny = Ny // ny_subabs
        sub_nx = Nx // nx_subabs

        crop_ny = sub_ny * ny_subabs
        crop_nx = sub_nx * nx_subabs
        y0 = (Ny - crop_ny) // 2
        x0 = (Nx - crop_nx) // 2

        idxs, _ = self._frequency_symmetric_filtering(
            time_window,
            fs,
            f0,
            high_freq=f1,
        )

        # Global angular-spectrum multiplication in the Fourier plane.
        if "AngularSpectrum" not in self.kernels:
            self._build_angular_kernel(z_prop, (dy, dx), wavelength, Ny, Nx)

        H = self.crop_array_centrally(self.kernels["AngularSpectrum"], (Ny, Nx))
        if H.ndim == 2:
            H = H[None, :, :]

        U_fft = self.fft.fft2(U0, axes=(-2, -1)) * self.fft.fftshift(H, axes=(-2, -1))

        # Crop the global Fourier plane so it tiles exactly into subapertures.
        U_fft = U_fft[:, y0 : y0 + crop_ny, x0 : x0 + crop_nx]
        
        U_fft = self.fft.fftshift(U_fft, axes=(-2, -1)) # fft shift to get low freq in the center now

        # Split Fourier plane into local subaperture tiles:
        # (Nz, crop_ny, crop_nx) -> (Nz, ny_subabs, nx_subabs, sub_ny, sub_nx)
        U_subap_all = (
            U_fft.reshape(Nz, ny_subabs, sub_ny, nx_subabs, sub_nx)
            .transpose(0, 1, 3, 2, 4)
        )

        # Local inverse FFT per subaperture.
        
        U_subap_all = self.fft.ifftshift(U_subap_all, axes=(-2, -1)) #ifft shift to prepare for ifft2 to correct the previous global fftshift
        U_subap_all = self.fft.ifft2(U_subap_all, axes=(-2, -1))

        # SVD expects: (ny_subabs, nx_subabs, sub_ny, sub_nx, Nz)
        U_subap_all = U_subap_all.transpose(1, 2, 3, 4, 0)
        U_subap_all = xp.ascontiguousarray(U_subap_all)

        # U_subap_all = self._svd_filter_shack_hartmann(
        #     U_subap_all,
        #     svd_threshold,
        # )

        # Temporal FFT expects: (B, Nz, sub_ny, sub_nx)
        B = ny_subabs * nx_subabs
        U_subap_all = (
            U_subap_all.reshape(B, sub_ny, sub_nx, Nz)
            .transpose(0, 3, 1, 2)
        )

        U_ft = self.fft.fft(U_subap_all, axis=1)[:, idxs, :, :]
        M0 = xp.mean(xp.abs(U_ft) ** 2, axis=1)

        return M0.reshape(ny_subabs, nx_subabs, sub_ny, sub_nx).astype(
            xp.float32,
            copy=False,
        )

    def _shack_hartmann_displacement_calculation(self, U_subabs, xp, pupil_threshold = 1.0, deviation_threshold = 3.0, ref = None):
        """Vectorized Shack-Hartmann displacement with single FFT2 call."""
        RangePush("Shack-Hartmann displacement calculation")
        ny_s, nx_s, Ny, Nx = U_subabs.shape
        
        # Reference sub-aperture
        if ref is None:
            if not (ny_s % 2 == nx_s % 2 == 1):
                print("Warning: number of sub-apertures must be odd.")
            ref = U_subabs[ny_s // 2, nx_s // 2]
        
        # Stack all sub-apertures: (ny_s*nx_s, Ny, Nx)
        moving_stack = U_subabs.reshape(ny_s * nx_s, Ny, Nx)
        
        # Zero-mean
        ref_zm = ref - xp.mean(ref)
        moving_zm = moving_stack - xp.mean(moving_stack, axis=(1, 2), keepdims=True)
        
        # Single batched FFT2 for all correlations
        F_ref = xp.fft.fft2(ref_zm)
        F_moving = xp.fft.fft2(moving_zm)
        cp = F_moving * F_ref.conj()
        xcorr = xp.abs(xp.fft.fftshift(xp.fft.ifft2(cp / (xp.abs(cp) + 1e-12)), axes=(1, 2)))
        
        # Peak locations - reshape to allow simple slicing
        # Flatten once
        xcorr_2d = xcorr.reshape(-1, Ny * Nx)

        # Peak indices
        peaks = xp.argmax(xcorr_2d, axis=1)
        py = peaks // Nx
        px = peaks % Nx

        # Clip peaks to avoid boundary (do it ONCE)
        py = xp.clip(py, 1, Ny - 2)
        px = xp.clip(px, 1, Nx - 2)

        n = py.shape[0]
        idx = xp.arange(n)

        # --- Direct neighbor access (no fancy indexing grids) ---
        v0 = xcorr[idx, py, px]

        vm_y = xcorr[idx, py - 1, px]
        vp_y = xcorr[idx, py + 1, px]

        vm_x = xcorr[idx, py, px - 1]
        vp_x = xcorr[idx, py, px + 1]

        # --- Parabolic refinement ---
        den_y = vm_y - 2 * v0 + vp_y + 1e-12
        den_x = vm_x - 2 * v0 + vp_x + 1e-12

        shift_y = py + 0.5 * (vm_y - vp_y) / den_y - Ny / 2
        shift_x = px + 0.5 * (vm_x - vp_x) / den_x - Nx / 2

        shift_y = shift_y.reshape(ny_s, nx_s)
        shift_x = shift_x.reshape(ny_s, nx_s)

        # --- Pupil mask for outlier rejection ---
        xs = xp.linspace(-1, 1, nx_s)
        ys = xp.linspace(-1, 1, ny_s)
        YY, XX = xp.meshgrid(ys, xs, indexing='ij')
        pupil_mask = (XX**2 + YY**2) <= pupil_threshold

        # --- Filtering ---
        shift_y_flat = shift_y[pupil_mask]
        shift_x_flat = shift_x[pupil_mask]

        mean_y = xp.mean(shift_y_flat)
        mean_x = xp.mean(shift_x_flat)

        std_y = xp.std(shift_y_flat)
        std_x = xp.std(shift_x_flat)

        thresh_y = deviation_threshold * std_y
        thresh_x = deviation_threshold * std_x

        bad = (
            (xp.abs(shift_y - mean_y) > thresh_y) |
            (xp.abs(shift_x - mean_x) > thresh_x) |
            (~pupil_mask)
        )

        shift_y[bad] = xp.nan
        shift_x[bad] = xp.nan
        
        RangePop()
        return shift_y.astype(xp.float32), shift_x.astype(xp.float32)
    
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
        return Z.astype(xp.float32) #important to be float32
    
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
        xp = np # go to numpy because it is FASTER (don'nt know why maybe lstsq)
        def make_G_gradient_zernike_matrix(Ny, Nx, mode_indices, n_sub_x, n_sub_y, dx, dy):
            n_modes = len(mode_indices)
            M = xp.zeros((2,n_sub_y, n_sub_x, n_modes), dtype=xp.float32)
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
        
        slopes_y = self._to_numpy(shifts_y) * (wavelength)/(pixel_pitch_y*(ny//nysubabs)) 
        slopes_x = self._to_numpy(shifts_x) * (wavelength)/(pixel_pitch_x*(nx//nxsubabs))
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
        xp = np
        fftpack = scipy.fftpack
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
            
        if xp == np:
            resize_phase_nan = resize_phase_nan_np
            shifts_y = self._to_numpy(shifts_y)
            shifts_x = self._to_numpy(shifts_x)
        elif xp == cp:
            resize_phase_nan = resize_phase_nan_cp
               
        slopes_y = shifts_y * wavelength # /(pixel_pitch_y*(ny//nysubabs)) same as before *(pixel_pitch_y*(ny//nysubabs)) the pitch between subaps before integration
        slopes_x = shifts_x * wavelength 
        southwell_fourier_phase = southwell_fourier_nan(slopes_y, slopes_x) * (2*xp.pi) / wavelength
        southwell_fourier_phase = resize_phase_nan(southwell_fourier_phase, ny, nx)
        return southwell_fourier_phase
    
    def _fresnel_transform_phase(self, frames, phase_term, zero_padding = False):
        
        if zero_padding:
            frames = self.pad_array_centrally(frames, zero_padding)
            
        if self.pipeline_version == "latest" : 

            return self.fft.fftshift(
                self.fft.fft2(frames *self.kernels["Fresnel_in"] * phase_term, axes=(-1, -2), norm="ortho"), axes=(-1, -2)
            ) *self.kernels["Fresnel_out"]
        else :
            return self.fft.fftshift(
            self.fft.fft2(frames * self.kernels["Fresnel_in"] * phase_term, axes=(-1, -2), norm="ortho"), axes=(-1, -2)
            )
            
    def _angular_spectrum_transform_phase(self, frames, phase_term, zero_padding = False):
        
        if zero_padding:
            frames = self.pad_array_centrally(frames, zero_padding)
            
        return self.fft.ifft2(
            self.fft.fft2(frames, axes=(-1, -2)) * self.fft.fftshift(self.kernels["AngularSpectrum"] * phase_term, axes=(-1, -2)) 
        )

    # ------------------------------------------------------------
    # Render tools for debug and visualization
    # ------------------------------------------------------------
           
    class PhasePlotter:
        def __init__(self, title="Wavefront Phase", cmap="twilight", figsize=(6, 6), dpi=100):
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            self.canvas = FigureCanvasAgg(self.fig)
            self.title = title
            self.cmap = cmap

        def plot(self, phase):
            if isinstance(phase, cp.ndarray):
                phase = cp.asnumpy(phase)   
            self.ax.clear()
            self.ax.set_title(self.title)
            im = self.ax.imshow((phase + np.pi) % (2*np.pi) - np.pi, cmap=self.cmap)
            self.fig.colorbar(im, ax=self.ax, fraction=0.029, pad=0.04)
            self.ax.set_aspect('equal')
            self.canvas.draw()
            img = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3]  # drop alpha channel
            return img
        def close(self):
            plt.close(self.fig)
        
    class ShiftsPlotter:
        def __init__(self, title="Wavefront Slopes from Sub-aperture Shifts", figsize=(8, 6), dpi=100, scale=50):
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            self.canvas = FigureCanvasAgg(self.fig)
            self.title = title
            self.scale = scale

        def plot(self, shifts_y, shifts_x):
            if isinstance(shifts_y, cp.ndarray):
                shifts_y = cp.asnumpy(shifts_y)
            if isinstance(shifts_x, cp.ndarray):
                shifts_x = cp.asnumpy(shifts_x)
            self.ax.clear()
            ny_subabs, nx_subabs = shifts_y.shape
            X, Y = np.meshgrid(np.arange(nx_subabs), np.arange(ny_subabs))
            self.ax.quiver(X, Y, shifts_x, shifts_y, scale=self.scale)
            self.ax.set_title(self.title)
            self.ax.set_xlabel('Sub-aperture X Index')
            self.ax.set_ylabel('Sub-aperture Y Index')
            self.ax.set_xlim(-0.5, nx_subabs - 0.5)
            self.ax.set_ylim(-0.5, ny_subabs - 0.5)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_aspect('equal')
            self.canvas.draw()
            img = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3]  # drop alpha channel
            return img
        def close(self):
            plt.close(self.fig)
        
    class SubapertureMontagePlotter:
        def __init__(self, figsize=(12, 8), dpi=100):
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            self.canvas = FigureCanvasAgg(self.fig)
        def plot(self, U_subaps):
            if isinstance(U_subaps, cp.ndarray):
                U_subaps = cp.asnumpy(U_subaps)
            rows = []
            for iy in range(U_subaps.shape[0]):
                row_imgs = [U_subaps[iy, ix] for ix in range(U_subaps.shape[1])]
                rows.append(np.hstack(row_imgs))
            montage_img = np.vstack(rows)
            return montage_img
        def close(self):
            plt.close(self.fig)
        
    def init_plot_debug(self):
        matplotlib.use("Agg")
        self.montage_plotter = self.SubapertureMontagePlotter()
        self.shifts_plotter = self.ShiftsPlotter()
        self.phase_plotter = self.PhasePlotter()
    
    def plot_debug(self, res, i):
        if "U_subaps" in res:
            montage_plotter = self.SubapertureMontagePlotter()
            montage_img = montage_plotter.plot(res["U_subaps"])
            montage_plotter.close()
                
            # cv2.imshow("Sub-aperture Montage", montage_img)
        if "shifts_y" in res and "shifts_x" in res:
            shifts_plotter = self.ShiftsPlotter()
            shifts_img = shifts_plotter.plot(res["shifts_y"], res["shifts_x"])
            shifts_plotter.close()
            # cv2.imshow("Sub-aperture Shifts", shifts_img)
        if "phase" in res:
            phase_plotter = self.PhasePlotter()
            phase_img = phase_plotter.plot(res["phase"])
            phase_plotter.close()
            # cv2.imshow("Reconstructed Phase", phase_img)
        if "M0notfixed" in res:
            M0notfixedimg = self._to_numpy((res["M0notfixed"] - np.min(res["M0notfixed"])) / (np.max(res["M0notfixed"]) - np.min(res["M0notfixed"])) * 255)
            # cv2.imshow("M0 without Phase Correction", M0notfixed_img)
        return { "montage": montage_img, "shifts": shifts_img, "phase": phase_img, "M0notfixed": M0notfixedimg }
        
    # ------------------------------------------------------------
    # One batch full pipeline
    # ------------------------------------------------------------

    def render_moments(self, parameters, frames = None, tictoc = False):
        RangePush("render_moments")
        
        def tic():
            if tictoc:
                return time.perf_counter(), time.process_time()
        def toc(t1, name="", arr=None):
            if tictoc:
                print(name)
                t2 = time.perf_counter(), time.process_time()
                print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
                print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
                if arr is not None : 
                    print(arr.dtype)
            
        
        
        
        t1 = tic()
        if frames is None:
            frames = self.read_frames(parameters["first_frame"], parameters["batch_size"])
        if frames is None:
            raise RuntimeError("Could not read frames properly")

        nt, ny, nx = frames.shape
        
        res = {} # intitialze result dict
        
        if parameters["shack_hartmann"] :
            t2 = tic()
            
            toc(t2, "Fresnel kernel build time")
            t2 = tic()
            if parameters["spatial_propagation"] == "Fresnel" :
                if (not "Fresnel_in" in self.kernels):
                    self._build_fresnel_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
                U_subaps = self._shack_hartmann_constructsubapsimages(frames, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], parameters["z"], parameters["low_freq"], parameters["high_freq"], parameters["sampling_freq"], nt, parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"], parameters["svd_threshold"]) # construct small images from the sub apertures of the Shack-Hartmann sensor
            elif parameters["spatial_propagation"] == "AngularSpectrum" :
                if (not "AngularSpectrum" in self.kernels):
                    self._build_angular_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
                U_subaps = self._shack_hartmann_constructsubapsimages_angular_spectrum(frames, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], parameters["z"], parameters["low_freq"], parameters["high_freq"], parameters["sampling_freq"], nt, parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"], parameters["svd_threshold"]) # construct small images from the sub apertures of the Shack-Hartmann sensor
            toc(t2, "Shack-Hartmann sub-aperture construction time", U_subaps)
            if parameters["debug"]:
                res["U_subaps"] = U_subaps

            t2 = tic()
            shifts_y, shifts_x = self._shack_hartmann_displacement_calculation(U_subaps, self.xp, pupil_threshold = parameters["shack_hartmann_pupil_threshold"], deviation_threshold = parameters["shack_hartmann_deviation_threshold"], ref = None) # get the shifts in pixels in the subapertures images
            toc(t2, "Shack-Hartmann displacement calculation time", shifts_y)
            if parameters["debug"]:
                res["shifts_y"] = shifts_y
                res["shifts_x"] = shifts_x
            t2 = tic()
            if parameters["shack_hartmann_zernike_fit"] :
                coefs, phase = self._shack_hartmann_zernike(ny, nx, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], shifts_y, shifts_x, parameters["shack_hartmann_zernike_fit_modes"]) # fit the shifts to Zernike polynomials to get the wavefront phase
                res["coefs"] = coefs
                if parameters["debug"]:
                    res["phase"] = phase
            elif parameters["shack_hartmann_southwell_phase_integration "] :
                phase = self._shack_hartmann_southwell(ny, nx, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], shifts_y, shifts_x)
                if parameters["debug"]:
                    res["phase"] = phase
            toc(t2, "Shack-Hartmann phase reconstruction time",phase)
            
            t2 = tic()
            phase_term = self.xp.exp(- 1j * phase) 
            phase_term = self.xp.nan_to_num(phase_term, nan=0.0) # completely mask the nan zone where the phase could'nt be estimated
            if parameters["zero_padding"]:
                phase_term = self.pad_array_centrally(phase_term, parameters["zero_padding"])
            if parameters["spatial_propagation"] == "Fresnel" :
                holograms = self._fresnel_transform_phase(frames, phase_term, zero_padding = parameters["zero_padding"])            
            elif parameters["spatial_propagation"] == "AngularSpectrum" :
                holograms = self._angular_spectrum_transform_phase(frames, phase_term, zero_padding = parameters["zero_padding"])   
            
            toc(t2, "Shack-Hartmann phase correction and Fresnel transform time", holograms)
            
            if parameters["debug"]:
                t2 = tic()
                if parameters["spatial_propagation"] == "Fresnel" :
                    hologramsnotfixed = self._fresnel_transform(frames, zero_padding = parameters["zero_padding"])
                elif parameters["spatial_propagation"] == "AngularSpectrum" :
                    hologramsnotfixed = self._angular_spectrum_transform(frames, zero_padding = parameters["zero_padding"]) 
                    
                toc(t2, "Fresnel transform without phase correction time", hologramsnotfixed)
        elif parameters["spatial_propagation"] == "Fresnel":
            if (not "Fresnel_in" in self.kernels):
                self._build_fresnel_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
            holograms = self._fresnel_transform(frames, zero_padding = parameters["zero_padding"])
        elif parameters["spatial_propagation"] == "AngularSpectrum":
            if (not "AngularSpectrum" in self.kernels):
                self._build_angular_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
            holograms = self._angular_spectrum_transform(frames, zero_padding = parameters["zero_padding"])
        t2 = tic()   
        holograms_f = self._svd_filter(holograms, parameters["svd_threshold"])
        toc(t2, "SVD filtering time", holograms_f)

        t2 = tic()
        spectrum_f = self._fourier_time_transform(holograms_f)

        idxs, freqs = self._frequency_symmetric_filtering(frames.shape[0], parameters["sampling_freq"], parameters["low_freq"], parameters["high_freq"])
        toc(t2, "Frequency filtering time",freqs)
        t2 = tic()
        psd = self.xp.abs(spectrum_f[idxs,:,:]) ** 2
        toc(t2, "PSD computation time",psd)

        t2 = tic()
        M0 = self._moment(psd, freqs, 0)
        M1 = self._moment(psd, freqs, 1)
        M2 = self._moment(psd, freqs, 2)
        toc(t2, "Moment computation time",M0)
        
        if parameters["shack_hartmann"] and parameters["debug"]: # to compute the res without the phase correction for debug purposes
            t2 = tic()
            hologramsnotfixed_f = self._svd_filter(hologramsnotfixed, parameters["svd_threshold"])
            spectrumnotfixed_f = self._fourier_time_transform(hologramsnotfixed_f)
            psdnotfixed = self.xp.abs(spectrumnotfixed_f[idxs,:,:]) ** 2
            M0notfixed = self._moment(psdnotfixed, freqs, 0)
            res["M0notfixed"] = M0notfixed
            toc(t2, "Moment computation svd fourier without phase correction time")
            
        res["M0"] = M0
        res["M1"] = M1
        res["M2"] = M2
        RangePop()
        
        toc(t1, "total render_moments time")
        
        
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

        if num_batch <= 0:
            return None
        
        if parameters["debug"]:
            import threading
            import queue
            
            self.init_plot_debug()

            # --- create plotting worker ---
            def plotting_worker(in_q, out_q, stop_event):
                while not stop_event.is_set() or not in_q.empty():
                    try:
                        i, res = in_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    out = self.plot_debug(res, i)
                    out_q.put((i, out))
                    in_q.task_done()
                    
            debugin_queue = queue.Queue(maxsize=14)  # limit to avoid memory blowup
            debugout_queue = queue.Queue()
            stop_event = threading.Event()
            debug_thread = threading.Thread(
                target=plotting_worker,
                args=(debugin_queue, debugout_queue, stop_event),
                daemon=True
            )
            debug_thread.start()
        
        if parameters["image_registration"]:
            frames = self.read_frames(first_frame, parameters["batch_size_registration"]) # the first frame to be rendered
            M0_reg = self.render_moments(parameters, frames = frames)["M0"] # render the first frame to be used as reference for the registration
            M0_reg = self._flatfield(M0_reg, parameters["registration_flatfield_gw"])
            reg_list = [None] * num_batch
            
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and parameters["shack_hartmann_zernike_fit"]:
            coefs_list = [None] * num_batch

        if self.backend == "cupy":
            stream_h2d = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)

            # --- Prefetch first batch ---
            frames_next = self.read_frames(
                first_frame,
                parameters["batch_size"]
            )

            with stream_h2d:
                d_frames_next = cp.asarray(frames_next) 

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

                M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                
                if parameters["debug"]:
                    # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                    debugin_queue.put((i, res))

                # --- Register current batch ---
                with stream_compute:
                    if parameters["image_registration"]:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list[i] = (shift_y, shift_x)
                
                stream_compute.synchronize()
                
                out_list.append(
                    cp.stack([M0, M1, M2], axis=2)
                )
                if "coefs" in res:
                    coefs_list[i] = res["coefs"]
                

            # Ensure transfers complete
            stream_h2d.synchronize()

            cp.cuda.Device().synchronize()

        elif self.backend == "cupyRAM":
            import queue
            import threading
            import time
            from collections import deque
            
            # Profiling data collection
            profile_data = {
                'read_times': deque(maxlen=100),  # Keep last 100 reads
                'transfer_times': deque(maxlen=100),
                'compute_times': deque(maxlen=100),
                'registration_times': deque(maxlen=100),
                'queue_wait_times': deque(maxlen=100),
                'total_batch_times': deque(maxlen=100),
                'queue_sizes': deque(maxlen=100),
            }
            
            # Create queues for frames and results
            frame_queue = queue.Queue(maxsize=4)  # Configurable queue depth
            result_queue = queue.Queue()
            
            # Control flags
            stop_reader = threading.Event()
            reader_done = threading.Event()
            
            # Reader thread function - continuously reads frames
            def reader_thread_func():
                frame_idx = first_frame
                batch_num = 0
                
                while batch_num < num_batch and not stop_reader.is_set():
                    read_start = time.perf_counter()
                    
                    # Read frames
                    frames = self.read_frames(frame_idx, batch_size)
                    
                    read_time = time.perf_counter() - read_start
                    profile_data['read_times'].append(read_time)
                    
                    # Put into queue (will block if queue is full)
                    queue_start = time.perf_counter()
                    frame_queue.put((batch_num, frame_idx, frames))
                    queue_time = time.perf_counter() - queue_start
                    profile_data['queue_wait_times'].append(queue_time)
                    
                    # Track queue size
                    profile_data['queue_sizes'].append(frame_queue.qsize())
                    
                    batch_num += 1
                    frame_idx += batch_stride
                
                reader_done.set()
            
            # Start reader thread
            reader_thread = threading.Thread(target=reader_thread_func, daemon=True)
            reader_thread.start()
            
            # CUDA streams for async operations
            stream_h2d = cp.cuda.Stream(non_blocking=True)
            stream_compute = cp.cuda.Stream(non_blocking=True)
            stream_registration = cp.cuda.Stream(non_blocking=True) if parameters["image_registration"] else None
            
            # Pre-start with first batch
            first_batch_num, first_frame_idx, first_frames = frame_queue.get()
            
            # Async transfer to GPU
            transfer_start = time.perf_counter()
            with stream_h2d:
                d_frames_current = cp.asarray(first_frames)
            stream_h2d.synchronize()
            transfer_time = time.perf_counter() - transfer_start
            profile_data['transfer_times'].append(transfer_time)
            
            current_batch_num = first_batch_num
            current_frame_idx = first_frame_idx
            d_frames = d_frames_current
            
            # Main processing loop
            processed_batches = 0
            next_batch_prefetched = False
            
            for i in tqdm(range(num_batch)):
                batch_start_time = time.perf_counter()
                
                # Check if we need to get next batch from queue
                if not next_batch_prefetched and i > 0:
                    queue_wait_start = time.perf_counter()
                    next_batch_num, next_frame_idx, next_frames = frame_queue.get()
                    queue_wait_time = time.perf_counter() - queue_wait_start
                    profile_data['queue_wait_times'].append(queue_wait_time)
                    
                    # Async transfer next batch
                    transfer_start = time.perf_counter()
                    with stream_h2d:
                        d_frames_next = cp.asarray(next_frames)
                    stream_h2d.synchronize()
                    transfer_time = time.perf_counter() - transfer_start
                    profile_data['transfer_times'].append(transfer_time)
                    
                    next_batch_prefetched = True
                
                # Compute current batch
                compute_start = time.perf_counter()
                with stream_compute:
                    res = self.render_moments(parameters, frames=d_frames)
                
                if res is None:
                    break
                
                M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                if parameters["debug"]:
                    # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                    debugin_queue.put((i, res))
                if "coefs" in res:
                    coefs_list[i] = res["coefs"]
                    
                compute_time = time.perf_counter() - compute_start
                profile_data['compute_times'].append(compute_time)
                
                # Registration processing (if enabled)
                if parameters["image_registration"]:
                    reg_start = time.perf_counter()
                    with stream_registration:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list[i] = (shift_y, shift_x)
                    
                    if stream_registration:
                        stream_registration.synchronize()
                    reg_time = time.perf_counter() - reg_start
                    profile_data['registration_times'].append(reg_time)
                
                # Ensure compute is done
                stream_compute.synchronize()
                
                # Store result
                out_list.append(cp.stack([M0, M1, M2], axis=2))
                
                # Swap buffers for next iteration
                if next_batch_prefetched and i + 1 < num_batch:
                    d_frames = d_frames_next
                    current_batch_num = next_batch_num
                    next_batch_prefetched = False
                    processed_batches += 1
                
                # Profile total batch time
                batch_time = time.perf_counter() - batch_start_time
                profile_data['total_batch_times'].append(batch_time)
            
            # Signal reader to stop and wait for completion
            stop_reader.set()
            reader_thread.join(timeout=5)
            
            # Ensure all CUDA operations complete
            stream_h2d.synchronize()
            stream_compute.synchronize()
            if stream_registration:
                stream_registration.synchronize()
            cp.cuda.Device().synchronize()
            
            # Print profiling summary
            if parameters.get("enable_profiling", True):
                print("\n" + "="*60)
                print("PROFILING SUMMARY - cupyRAM Backend")
                print("="*60)

                import numpy as np
                
                def print_stats(name, times):
                    if len(times) > 0:
                        avg = np.mean(times) * 1000  # Convert to ms
                        std = np.std(times) * 1000
                        min_t = np.min(times) * 1000
                        max_t = np.max(times) * 1000
                        print(f"{name:20s}: Avg={avg:6.2f}ms ±{std:5.2f}ms, Min={min_t:6.2f}ms, Max={max_t:6.2f}ms")
                    else:
                        print(f"{name:20s}: No data")
                
                print_stats("File Read Time", profile_data['read_times'])
                print_stats("Queue Wait Time", profile_data['queue_wait_times'])
                print_stats("H2D Transfer Time", profile_data['transfer_times'])
                print_stats("Compute Time", profile_data['compute_times'])
                print_stats("Registration Time", profile_data['registration_times'])
                print_stats("Total Batch Time", profile_data['total_batch_times'])
                
                if len(profile_data['queue_sizes']) > 0:
                    avg_queue = np.mean(profile_data['queue_sizes'])
                    max_queue = np.max(profile_data['queue_sizes'])
                    print(f"\nQueue Statistics:")
                    print(f"  Average Queue Size: {avg_queue:.1f}")
                    print(f"  Max Queue Size: {max_queue}")
                
                # Calculate throughput
                if len(profile_data['total_batch_times']) > 0:
                    total_time = np.sum(profile_data['total_batch_times'])
                    total_batches = len(profile_data['total_batch_times'])
                    total_frames = total_batches * batch_size
                    print(f"\nThroughput:")
                    print(f"  Batches/sec: {total_batches/total_time:.2f}")
                    print(f"  Frames/sec: {total_frames/total_time:.2f}")
                    
                    # Utilization metrics
                    compute_total = np.sum(profile_data['compute_times'])
                    read_total = np.sum(profile_data['read_times'])
                    transfer_total = np.sum(profile_data['transfer_times'])
                    
                    if total_time > 0:
                        print(f"\nUtilization (of total time):")
                        print(f"  Compute: {compute_total/total_time*100:.1f}%")
                        print(f"  File I/O: {read_total/total_time*100:.1f}%")
                        print(f"  H2D Transfer: {transfer_total/total_time*100:.1f}%")
                        
                        # Overlap efficiency
                        ideal_time = max(compute_total, read_total + transfer_total)
                        overlap_efficiency = ideal_time / total_time * 100 if total_time > 0 else 0
                        print(f"  Overlap Efficiency: {overlap_efficiency:.1f}%")
                
                print("="*60 + "\n")

        elif self.backend == "numpy multiprocessing":
            

            print("CPU count :" , cpu_count)

            def process_batch(args):
                i, first_frame, parameters, self_state = args
                try:
                    frames = self_state.read_frames(first_frame + i * parameters["batch_stride"], parameters["batch_size"])
                    res = self_state.render_moments(parameters, frames=frames)
                    if res is None:
                        return i, None
                    M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                    if parameters["debug"]:
                        # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                        debugin_queue.put((i, res))
                    if "coefs" in res:
                        coefs_list[i] = res["coefs"]
                    shift_y = shift_x = None
                    if parameters["image_registration"]:
                        M0_ff = self_state._flatfield(M0, parameters["registration_flatfield_gw"])
                        shift_y, shift_x = self_state._registration(self_state.M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self_state._roll2d(M0, shift_y, shift_x, np)
                        M1 = self_state._roll2d(M1, shift_y, shift_x, np)
                        M2 = self_state._roll2d(M2, shift_y, shift_x, np)
                    return i, (M0, M1, M2, debug_imgs, shift_y, shift_x)
                except Exception:
                    traceback.print_exc()
                    return i, None

            out_list = [None] * num_batch
            debug_list = [None] * num_batch

            with ProcessPoolExecutor(max_workers=cpu_count()) as pool:
                futures = {pool.submit(process_batch, (i, first_frame, parameters, self)): i for i in range(num_batch)}
                for fut in tqdm(as_completed(futures), total=num_batch):
                    i, result = fut.result()
                    if result is None:
                        for f in futures: f.cancel()
                        break
                    M0, M1, M2, debug_imgs, shift_y, shift_x = result
                    out_list[i] = np.stack([M0, M1, M2], axis=2)
                    debug_list[i] = debug_imgs
                    if shift_y is not None:
                        reg_list[i] = (shift_y, shift_x)

            out_list = [x for x in out_list if x is not None]
            debug_list = {i: v for i, v in enumerate(debug_list) if v is not None}

        else:
            for i in tqdm(range(num_batch)):

                try:

                    frames = self.read_frames(first_frame + i * parameters["batch_stride"] , parameters["batch_size"])

                    res = self.render_moments(parameters, frames = frames)

                    if res is None:
                        break

                    M0, M1, M2 = res["M0"], res["M1"], res["M2"]
                    if parameters["debug"]:
                        # debug_imgs = {k: res[k] for k in res if k not in ["M0", "M1", "M2"]}
                        debugin_queue.put((i, res))
                    if "coefs" in res:
                        coefs_list[i] = res["coefs"]

                    if parameters["image_registration"]:
                        M0_ff = self._flatfield(M0, parameters["registration_flatfield_gw"])
                        (shift_y, shift_x) = self._registration(M0_reg, M0_ff, parameters["registration_disc_ratio"])
                        M0 = self._roll2d(M0, shift_y, shift_x, self.xp)
                        M1 = self._roll2d(M1, shift_y, shift_x, self.xp)
                        M2 = self._roll2d(M2, shift_y, shift_x, self.xp)
                        reg_list[i] = (shift_y, shift_x)

                    out_list.append(
                        self.xp.stack([M0, M1, M2], axis=2)
                    )
                    debug_list[i] = debug_imgs

                except Exception:
                    traceback.print_exc()
                    break

            if len(out_list) == 0:
                return None
        
        if parameters["shack_hartmann"] and parameters["spatial_propagation"] == "Fresnel" and all(coefs is not None for coefs in coefs_list):
            zernike_coefs = self._to_numpy(self.xp.array(coefs_list))
        else:
            zernike_coefs = None
        
        vid = self.xp.stack(out_list, axis=3) 
            
        if parameters["debug"]:
            debugin_queue.join()
            stop_event.set()
            debug_thread.join()
            
            debug_results = {}
            while not debugout_queue.empty():
                i, res = debugout_queue.get()
                debug_results[i] = res

            if any(res is not None for res in debug_results.values()):
                streams = {"montage": [], "shifts": [], "phase": [], "M0notfixed": []}
                for i in range(len(debug_results)):
                    dic = debug_results[i]
                    for key, img in dic.items():
                        if parameters["square"] and key in ["montage", "M0notfixed"]:
                            m = max(img.shape[0], img.shape[1])
                            img = self._resize(img, m, m)
                        streams[key].append(img)
                import numpy as np
                vid_debug = [np.stack(stream, axis=2) for stream in streams.values() if len(stream) > 0]
            else:
                
                vid_debug = None
        else:
            vid_debug = None

        if parameters["accumulation"] > 1:
            acc = parameters["accumulation"] 
            ny, nx, nimgs, nt = vid.shape
            vid = self.xp.reshape(vid[:,:,:,:(nt//acc)*acc],(ny, nx, nimgs, nt//acc, acc)) @ self.xp.ones(acc) # / acc

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
            
        def save_to_h5path(h5_path, v, parameters, reg_list = None, zernike_coefs = None, git_commit = None):
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
                if git_commit is not None:
                    f.create_dataset("git_commit", data=git_commit)

        if holodoppler_path is True:
            # make the new directory at the same level as the input file with its name then _HD{idx} where idx is the max index of existing holodoppler output directories for this file + 1
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            dir_name = f"{base_name}_HD"
            parent_dir = os.path.dirname(self.file_path)
            holodoppler_dir_name = f"{dir_name}"
            holodoppler_path = os.path.join(parent_dir, holodoppler_dir_name)
            os.makedirs(holodoppler_path, exist_ok=True)
            # make a png, mp4 json and h5 sub directories with their respective content
            png_dir = os.path.join((holodoppler_path), "png")
            mp4_dir = os.path.join((holodoppler_path), "mp4")
            avi_dir = os.path.join((holodoppler_path), "avi")
            json_dir = os.path.join((holodoppler_path), "json")
            h5_dir = os.path.join((holodoppler_path), "raw")
            print(f"Saving output to: {holodoppler_path}")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(mp4_dir, exist_ok=True)
            os.makedirs(avi_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(h5_dir, exist_ok=True)
            
            # save m0 as mp4 and avi
            def normalize(arr):
                arr = arr.astype(np.float32)
                lo, hi = arr.min(), arr.max()
                return ((arr - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else arr.astype(np.uint8)

            def temporal_gaussian(arr, sigma):
                if sigma == 0 :
                    return arr
                return gaussian_filter1d(arr.astype(np.float32), sigma=sigma, axis=2)

            def write_video(path, frames, fps, fourcc, is_color=False):
                h, w, n = frames.shape[0], frames.shape[1], frames.shape[2]
                out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h), isColor=is_color)
                for i in range(n):
                    out.write(frames[:, :, i] if frames.ndim == 3 else frames[:, :, i, :])
                out.release()

            def save_pair(stem, frames, fps, mp4_dir, avi_dir, sigma = 4.0, is_color=False, save_png=True):
                frames = temporal_gaussian(frames, sigma) # removing for clarity only raw output
                frames = normalize(frames)
                
                # print(frames.shape, frames.dtype, type(frames))
                write_video(os.path.join(mp4_dir, f"{stem}.mp4"), frames, min(fps, 65), "mp4v", is_color)
                write_video(os.path.join(avi_dir, f"{stem}.avi"), frames, min(fps, 65), "XVID", is_color)
                if save_png:
                    plt.imsave(os.path.join(png_dir, f"{stem}.png"), np.mean(frames, axis=2), cmap="gray")

            fps = num_batch / (end_frame - first_frame) * parameters["sampling_freq"]

            save_pair("moment_0", vid[:, :, 0, :], fps, mp4_dir, avi_dir, sigma=0)
            save_pair("moment_1", vid[:, :, 1, :], fps, mp4_dir, avi_dir, sigma=0)
            save_pair("moment_2", vid[:, :, 2, :], fps, mp4_dir, avi_dir, sigma=0)

            if parameters["debug"]:
                def flatfield3D(arr, gw):
                    if arr.ndim != 3:
                        raise ValueError("Input array must be 3D")
                    if gw <= 1:
                        return arr
                    blurred = np_gaussian_filter(arr, sigma=(gw, gw, 1))
                    blurred[blurred == 0] = 1
                    return arr / blurred
                save_pair("moment_0_slidingavg_flatfield", flatfield3D(vid[:, :, 0, :], parameters["registration_flatfield_gw"]), fps, mp4_dir, avi_dir, sigma=1.50)
            

            if parameters["debug"] and vid_debug is not None:
                for idx, v in enumerate(vid_debug):
                    save_pair(f"debug_{idx}", v, fps, mp4_dir, avi_dir, sigma=0, is_color=v.ndim == 4, save_png=False)

            # save json
            with open(os.path.join(json_dir, "parameters.json"), "w") as f:
                json.dump(parameters, f, indent=4)
                
            # get git info 
            # if git is available write the current commit hash
            try:
                import subprocess
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                git_txt = "Git commit hash: " + git_commit + "\n"
                # check if there are uncommited changes and add a warning if there are
                git_status = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
                if git_status:
                    git_txt += "Warning: There are uncommited changes in the repository, the results may not be reproducible\n"
            except Exception:
                git_txt = "Git commit hash: Not available\n"
            # save h5
            save_to_h5path(os.path.join(h5_dir, f"{holodoppler_dir_name}_output.h5"), np.permute_dims(vid, (3, 1, 0, 2)), parameters, reg_list if parameters["image_registration"] else None, zernike_coefs, git_commit=git_txt)
            # add a version.txt file with the version of the holodoppler pipeline used
            with open(os.path.join(holodoppler_path, "git_version.txt"), "w") as f:
                f.write(f"Python:\n")
                f.write(f"Holodoppler pipeline version: {self.pipeline_version}\n")
                f.write(f"Holodoppler backend: {self.backend}\n") 
                f.write(f"{git_txt}")
                
            with open(os.path.join(holodoppler_path, "version_holodoppler.txt"), "w") as f:
                f.write(f"py 0.1.0")
            if self.ext == ".holo":
                with open(os.path.join(holodoppler_path, "version_holovibes.txt"), "w") as f:
                    f.write(f"{self.file_footer.get('info',{}).get('holovibes_version', 'unknown')}")

        if return_numpy:
            return self._to_numpy(vid)

        return vid
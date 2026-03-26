import numpy as np
import traceback
import h5py
from tqdm import tqdm
import cv2
import json
import os

import cinereader

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Optional CuPy support
# ------------------------------------------------------------



try:
    import cupy as cp
    import cupyx.scipy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    _cupy_available = True
except Exception:
    cp = None
    cp_fft = None
    _cupy_available = False

import scipy.fft as np_fft
from scipy.ndimage import gaussian_filter as np_gaussian_filter
from matlab_imresize.imresize import imresize


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

        if self.backend == "cupy":
            if not _cupy_available:
                raise RuntimeError("CuPy backend requested but CuPy is not available.")
            self.xp = cp
            self.fft = cp_fft
            self.gaussian_filter = cp_gaussian_filter
        else:
            self.backend = "numpy"
            self.xp = np
            self.fft = np_fft
            self.gaussian_filter = np_gaussian_filter
            
    def _to_backend(self, arr):
        if self.backend == "cupy":
            return cp.asarray(arr)
        return arr

    def _to_numpy(self, arr):
        if self.backend == "cupy":
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
    
    def read_frames_holo(self, first_frame, frame_size):

        try:

            byte_begin = (
                self.HOLO_HEADER_SIZE
                + self.file_header["width"] * self.file_header["height"] * first_frame * self.file_header["bit_depth"] // 8
            )

            byte_size = self.file_header["width"] * self.file_header["height"] * frame_size * self.file_header["bit_depth"] // 8

            self.fid.seek(byte_begin)
            raw_bytes = self.fid.read(byte_size)

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

        if svd_threshold < 0:
            return H

        xp = self.xp

        sz = H.shape
        H2 = H.reshape((sz[0],sz[-1] * sz[-2])).T

        cov = H2.conj().T @ H2

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
    
    def _old_frequency_symmetric_filtering(self, batch_size, fs, f1, f2 = None):
        # old and not clean but similar to matlab version

        if f2 is None:
            f2 = fs/2

        # convert frequencies to indices
        n1 = int(np.ceil(f1 * batch_size / fs))
        n2 = int(np.ceil(f2 * batch_size / fs))

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
        f_range = np.arange(n1, n2 + 1) * (fs / batch_size)
        f_range_sym = np.arange(-n2, -n1 + 1) * (fs / batch_size)

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
        xp = self.xp
        fft = np_fft
        img = img.astype(xp.float32)

        h, w = img.shape[:2]
        rest = img.shape[2:]

        # Flatten trailing dimensions
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

        # Flatten trailing dimensions
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
    # One batch full pipeline
    # ------------------------------------------------------------

    def render_moments(self, parameters, frames = None):

        if frames is None:
            frames = self.read_frames(parameters["first_frame"], parameters["batch_size"])
        if frames is None:
            raise RuntimeError("Could not read frames properly")

        nt, ny, nx = frames.shape

        if parameters["spatial_propagation"] == "Fresnel":
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

        return M0, M1, M2

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
        
        
        if parameters["image_registration"]:
            frames = self.read_frames(first_frame, parameters["batch_size_registration"]) # the first frame to be rendered
            M0_reg, _, _ = self.render_moments(parameters, frames = frames)
            M0_reg = self._flatfield(M0_reg, parameters["registration_flatfield_gw"])
            reg_list = []

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

                M0, M1, M2 = res

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

                    M0, M1, M2 = res

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

                except Exception:
                    traceback.print_exc()
                    break

            if len(out_list) == 0:
                return None

        vid = self.xp.stack(out_list, axis=3)

        if parameters["square"]:
            m = max(vid.shape[0], vid.shape[1])
            # vid = self.resize_fft2_slicewise(vid, m, m)
            vid = self._to_numpy(vid).astype(np.float64)
            vid = self._resize(vid, m, m)
            
        if parameters["transpose"]:
            vid = self.xp.transpose(vid, axes=(1, 0, 2, 3))
            
        if parameters["flip_x"]:
            vid = self.xp.flip(vid, axis=1)
        
        if parameters["flip_y"]:
            vid = self.xp.flip(vid, axis=0)

        if (h5_path is not None) or (mp4_path is not None) or holodoppler_path:
            v = self._to_numpy(vid)
            
        def save_to_h5path(h5_path, v, parameters, reg_list = None):
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("moment0", data=v[:, :, :, 0])
                f.create_dataset("moment1", data=v[:, :, :, 1])
                f.create_dataset("moment2", data=v[:, :, :, 2])
                f.create_dataset("HD_parameters", data=json.dumps(parameters))
                if parameters["image_registration"]:
                    f.create_dataset("registration", data=self._to_numpy(self.xp.array(reg_list)))

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
            save_to_h5path(h5_path, np.permute_dims(v, (3, 0, 1, 2)), parameters, reg_list if parameters["image_registration"] else None)

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
            # save png
            for i in range(v.shape[2]):
                plt.imsave(os.path.join(png_dir, f"moment_{i}.png"), np.mean(v[:, :, i, :],axis=2), cmap="gray")
            vid_norm = v[:, :, 0, :]  # shape (H, W, T)
            vid_norm = vid_norm.astype(np.float32)
            vid_norm = (vid_norm - vid_norm.min()) / (vid_norm.max() - vid_norm.min())
            vid_norm = (vid_norm * 255).astype(np.uint8)
            # save mp4
            height, width = v.shape[0], v.shape[1]
            duration = 1/parameters["sampling_freq"] * (end_frame-first_frame) 
            fps = num_batch / duration
            out_mp4 = cv2.VideoWriter(os.path.join(mp4_dir, "moment_0.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width, height),isColor=False)
            for i in range(num_batch):
                frame = vid_norm[:, :, i]
                out_mp4.write(frame)
            out_mp4.release()
            # save mp4
            out_avi = cv2.VideoWriter(os.path.join(mp4_dir, "moment_0.avi"),cv2.VideoWriter_fourcc(*'XVID'),fps,(width, height),isColor=False)
            for i in range(num_batch):
                frame = vid_norm[:, :, i]
                out_avi.write(frame)
            out_avi.release()
            # save json
            with open(os.path.join(json_dir, "parameters.json"), "w") as f:
                json.dump(parameters, f, indent=4)
            # save h5
            save_to_h5path(os.path.join(h5_dir, f"{holodoppler_dir_name}_output.h5"), np.permute_dims(v, (3, 1, 0, 2)), parameters, reg_list if parameters["image_registration"] else None)
            # add a version.txt file with the version of the holodoppler pipeline used
            with open(os.path.join(holodoppler_path, "version.txt"), "w") as f:
                f.write(f"Python:\n")
                f.write(f"Holodoppler pipeline version: {self.pipeline_version}\n")
                f.write(f"Holodoppler backend: {self.backend}\n") 

        if return_numpy:
            return self._to_numpy(vid)

        return vid
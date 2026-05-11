"""
Holodoppler - Main processing class
"""

import os
import json
import time
import threading
import queue
import traceback
from collections import defaultdict
from importlib.metadata import version

import numpy as np
import cv2
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from .backend import BackendManager
from .file_io import FileReaderFactory
from .propagation import PropagationKernels
from .filtering import Filtering
from .registration import ImageRegistration
from .shack_hartmann import ShackHartmann
from .zernike import ZernikeReconstructor
from .moments import MomentCalculator
from .plotting import DebugPlotterManager
from .utils import (gaussian_flatfield, normalize_image, temporal_gaussian_filter, flatfield3D, 
                    pad_array_centrally, crop_array_centrally, elliptical_mask, resize_fft2_slicewise, resize_matlab_slicewise)


class Holodoppler:
    """
    Holodoppler processing class for .holo and .cine files.
    
    Backend:
        backend="numpy"  -> CPU
        backend="cupy"   -> GPU (if available)
        backend="cupyRAM" -> GPU with async I/O
    
    Pipeline versions:
        "latest" - Current optimized pipeline
        "old" - Legacy MATLAB-compatible pipeline
        "latest_old_reg" - Current pipeline with old registration
    """
    
    def __init__(self, backend="numpy", pipeline_version="latest"):
        self.__version__ = version("holodoppler")
        
        
        self.backend_name = backend
        self.pipeline_version = pipeline_version
        
        print(f"HoloDoppler : py{self.__version__}  {self.backend_name}  {self.pipeline_version}")
        
        # Initialize components
        self.bm = BackendManager(backend)
        self.propagation = PropagationKernels(self.bm)
        self.filtering = Filtering(self.bm)
        self.registration = ImageRegistration(self.bm)
        self.shack_hartmann = ShackHartmann(self.bm, self.propagation, self.filtering)
        self.zernike = ZernikeReconstructor(self.bm)
        self.moments = MomentCalculator(self.bm)
        
        # State
        self.file_reader = None
        self.parameters = {}
        
        # Pipeline version configuration
        self._configure_pipeline()
    
    def _configure_pipeline(self):
        """Configure pipeline functions based on version"""
        if self.pipeline_version == "latest":
            self._frequency_filter = self.filtering.frequency_symmetric_filtering
            self._propagate = self.propagation.fresnel_transform
            self._propagate_with_phase = self.propagation.fresnel_transform_with_phase
            self._register = self.registration.register_trs
            self._apply_registration = self.registration.apply_translation
            self._moment = self.moments.moment
            self._resize_to_square = resize_fft2_slicewise
        elif self.pipeline_version == "old":
            self._frequency_filter = self._old_frequency_filter
            self._propagate = self.propagation.fresnel_transform
            self._propagate_with_phase = self.propagation.fresnel_transform_with_phase
            self._register = self.registration.translation_only
            self._apply_registration = self.registration.apply_roll
            self._moment = self.moments.moment_khz
            self._resize_to_square = resize_matlab_slicewise
        elif self.pipeline_version == "latest_old_reg":
            self._frequency_filter = self.filtering.frequency_symmetric_filtering
            self._propagate = self.propagation.fresnel_transform
            self._propagate_with_phase = self.propagation.fresnel_transform_with_phase
            self._register = self.registration.translation_only
            self._apply_registration = self.registration.apply_translation
            self._moment = self.moments.moment
            self._resize_to_square = resize_fft2_slicewise
    
    def _old_frequency_filter(self, batch_size, sampling_freq, low_freq, high_freq=None): # TODO move elsewhere
        """Legacy frequency filtering (MATLAB compatible)"""
        if high_freq is None:
            high_freq = sampling_freq / 2
        
        n1 = int(np.ceil(low_freq * batch_size / sampling_freq))
        n2 = int(np.ceil(high_freq * batch_size / sampling_freq))
        
        n1 = max(min(n1, batch_size), 1)
        n2 = max(min(n2, batch_size), 1)
        
        n3 = batch_size - n2 + 1
        n4 = batch_size - n1 + 1
        
        i1, i2 = n1 - 1, n2
        i3, i4 = n3 - 1, n4
        
        f_range = np.arange(n1, n2 + 1) * (sampling_freq / batch_size)
        f_range_sym = np.arange(-n2, -n1 + 1) * (sampling_freq / batch_size)
        freqs = np.concatenate([f_range, f_range_sym])
        
        idxs = np.zeros(batch_size, dtype=bool)
        idxs[i1:i2] = True
        idxs[i3:i4] = True
        
        return self.bm.to_backend(idxs), self.bm.to_backend(freqs)
    
    # ------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------
    
    def load_file(self, file_path):
        """Load a .holo or .cine file"""
        if self.file_reader is not None:
            self.file_reader.close()
        
        self.file_reader = FileReaderFactory.create(file_path)
        self.file_reader.open()
        
        # Store metadata for export
        if self.file_reader.ext == ".holo":
            self.holo_header = self.file_reader.file_header
            self.holo_footer = self.file_reader.file_footer
        else:
            self.cine_metadata = self.file_reader.metadata_json
    
    def close_file(self):
        """Close the currently open file"""
        if self.file_reader is not None:
            self.file_reader.close()
            self.file_reader = None
    
    def read_frames(self, first_frame, batch_size):
        """Read frames (returns CPU array)"""
        frames = self.file_reader.read_frames(first_frame, batch_size)
        if frames is None:
            return None
        return self.bm.to_backend(frames)
    
    # ------------------------------------------------------------
    # Single batch processing
    # ------------------------------------------------------------
    
    def render_moments(self, parameters, frames=None, registration_ref=None, tictoc=False):
        """Process a single batch of frames"""
        if frames is None:
            frames = self.read_frames(parameters["first_frame"], parameters["batch_size"])
        if frames is None:
            raise RuntimeError("Could not read frames properly")
        
        nt, ny, nx = frames.shape
        res = {}
        
        class Accumulator:
            def __init__(self, batch_size, xp):
                self.xp = xp
                self.batch_size = batch_size
                self.buffers = defaultdict(list)
            
            def add(self, data_dict):
                for k, v in data_dict.items():
                    self.buffers[k].append(v)
                if len(next(iter(self.buffers.values()))) >= self.batch_size:
                    return self.flush()
                return None
            
            def flush(self):
                if not self.buffers:
                    return None
                batch = {k: self.xp.sum(self.xp.stack(v), axis=0) / self.batch_size 
                        for k, v in self.buffers.items()}
                self.buffers.clear()
                return batch
        
        subaps_acc = Accumulator(parameters.get("shack_hartmann_accumulation", 1), self.bm.xp)
        main_acc = Accumulator(parameters.get("accumulation", 1), self.bm.xp)
        
        t0 = time.perf_counter()
        
        def toc(name=""):
            if tictoc:
                dt = time.perf_counter() - t0
                print(name, f"{dt*1000:.3f} ms")

        # Shack-Hartmann phase estimation
        if parameters.get("shack_hartmann", False):
            sub_batch_size = nt // subaps_acc.batch_size
            sub_batch_stride = sub_batch_size
            
            for it in range(subaps_acc.batch_size):
                frames_sub = frames[sub_batch_stride*it:sub_batch_stride*it + sub_batch_size]
                
                if parameters["spatial_propagation"] == "Fresnel":
                    self.propagation.build_fresnel_kernel(
                        parameters["z"], parameters["pixel_pitch"],
                        parameters["wavelength"], ny, nx,
                        zero_padding=parameters.get("zero_padding")
                    )
                    U_subaps = self.shack_hartmann.construct_subapertures_fresnel(
                        frames_sub, parameters["pixel_pitch"], parameters["pixel_pitch"],
                        parameters["wavelength"], parameters["z"],
                        parameters["low_freq"], parameters["high_freq"],
                        parameters["sampling_freq"], frames_sub.shape[0],
                        parameters["shack_hartmann_nx_subap"],
                        parameters["shack_hartmann_ny_subap"],
                        parameters["svd_threshold"]
                    )
                else:  # AngularSpectrum
                    self.propagation.build_angular_kernel(
                        parameters["z"], parameters["pixel_pitch"],
                        parameters["wavelength"], ny, nx,
                        zero_padding=parameters.get("zero_padding")
                    )
                    U_subaps = self.shack_hartmann.construct_subapertures_angular(
                        frames_sub, parameters["pixel_pitch"], parameters["pixel_pitch"],
                        parameters["wavelength"], parameters["z"],
                        parameters["low_freq"], parameters["high_freq"],
                        parameters["sampling_freq"], frames_sub.shape[0],
                        parameters["shack_hartmann_nx_subap"],
                        parameters["shack_hartmann_ny_subap"],
                        parameters["svd_threshold"]
                    )
                
                subaps_acc.add({"U_subaps": U_subaps})
            
            b = subaps_acc.flush()
            if b is not None:
                U_subaps = b["U_subaps"]
            
            if parameters.get("debug"):
                res["U_subaps"] = U_subaps
            
            # Calculate displacements
            shifts_y, shifts_x = self.shack_hartmann.calculate_displacements(
                U_subaps,
                pupil_threshold=parameters.get("shack_hartmann_pupil_threshold", 1.0),
                deviation_threshold=parameters.get("shack_hartmann_deviation_threshold", 3.0),
                shifts_range=parameters.get("shack_hartmann_shifts_pixel_range_threshold", 20.0)
            )
            
            if parameters.get("debug"):
                res["shifts_y"] = shifts_y
                res["shifts_x"] = shifts_x
            
            # Phase reconstruction
            if parameters.get("shack_hartmann_zernike_fit"):
                coefs, phase = self.zernike.fit_zernike(
                    ny, nx, parameters["pixel_pitch"], parameters["pixel_pitch"],
                    parameters["wavelength"], shifts_y, shifts_x,
                    parameters["shack_hartmann_zernike_fit_modes"]
                )
                res["coefs"] = coefs
                if parameters.get("debug"):
                    res["phase"] = phase
            else:
                phase = None
            
            # Apply phase correction
            if phase is not None:
                phase_term = self.bm.xp.exp(-1j * phase)
                phase_term = self.bm.xp.nan_to_num(phase_term, nan=0.0)
                if parameters.get("zero_padding"):
                    phase_term = pad_array_centrally(phase_term, parameters["zero_padding"], self.bm.xp)
            else:
                phase_term = None
        toc("Usubap")
        # Main processing loop
        sub_batch_size = nt // main_acc.batch_size
        sub_batch_stride = sub_batch_size
        
        for it in range(main_acc.batch_size):
            frames_sub = frames[sub_batch_stride*it:sub_batch_stride*it + sub_batch_size]
            
            # Propagate with or without phase correction
            if parameters.get("shack_hartmann") and phase_term is not None:
                if parameters["spatial_propagation"] == "Fresnel":
                    holograms = self._propagate_with_phase(frames_sub, phase_term, 
                                                           zero_padding=parameters.get("zero_padding"))
                else:
                    holograms = self.propagation.angular_spectrum_transform_with_phase(
                        frames_sub, phase_term, zero_padding=parameters.get("zero_padding")
                    )
                
                if parameters.get("debug"):
                    if parameters["spatial_propagation"] == "Fresnel":
                        holograms_not_fixed = self._propagate(frames_sub, 
                                                               zero_padding=parameters.get("zero_padding"))
                    else:
                        holograms_not_fixed = self.propagation.angular_spectrum_transform(
                            frames_sub, zero_padding=parameters.get("zero_padding")
                        )
            else:
                if parameters["spatial_propagation"] == "Fresnel":
                    self.propagation.build_fresnel_kernel(
                        parameters["z"], parameters["pixel_pitch"],
                        parameters["wavelength"], ny, nx,
                        zero_padding=parameters.get("zero_padding")
                    )
                    holograms = self._propagate(frames_sub, zero_padding=parameters.get("zero_padding"))
                else:
                    self.propagation.build_angular_kernel(
                        parameters["z"], parameters["pixel_pitch"],
                        parameters["wavelength"], ny, nx,
                        zero_padding=parameters.get("zero_padding")
                    )
                    holograms = self.propagation.angular_spectrum_transform(
                        frames_sub, zero_padding=parameters.get("zero_padding")
                    )
                holograms_not_fixed = None
            
            # SVD filtering
            holograms_f = self.filtering.svd_filter(holograms, parameters["svd_threshold"])
            
            # Temporal FFT
            spectrum_f = self.filtering.fourier_time_transform(holograms_f)
            
            # Frequency filtering
            idxs, freqs = self._frequency_filter(
                frames_sub.shape[0], parameters["sampling_freq"],
                parameters["low_freq"], parameters.get("high_freq")
            )
            psd = self.bm.xp.abs(spectrum_f[idxs, :, :]) ** 2
            
            res_batch = {
                "M0": self._moment(psd, freqs, 0),
                "M1": self._moment(psd, freqs, 1),
                "M2": self._moment(psd, freqs, 2)
            }
            
            if parameters.get("debug"):
                res_batch["spectrum_line"] = self.bm.xp.mean(
                    self.bm.xp.abs(spectrum_f[:, :, :]) ** 2, axis=(-1, -2)
                )
                res_batch["freqs"] = freqs
                
                if holograms_not_fixed is not None:
                    holograms_not_fixed_f = self.filtering.svd_filter(holograms_not_fixed, 
                                                                       parameters["svd_threshold"])
                    spectrum_not_fixed = self.filtering.fourier_time_transform(holograms_not_fixed_f)
                    psd_not_fixed = self.bm.xp.abs(spectrum_not_fixed[idxs, :, :]) ** 2
                    res_batch["M0notfixed"] = self._moment(psd_not_fixed, freqs, 0)
            
            b = main_acc.add(res_batch)
        toc(name="Propag")
        
        if b is not None:
            res.update(b)
        
        # Registration
        if parameters.get("image_registration") and registration_ref is not None:
            M0_ff = gaussian_flatfield(res["M0"], parameters["registration_flatfield_gw"], 
                                        self.bm.gaussian_filter)
            reg = self._register(registration_ref, M0_ff, parameters.get("registration_disc_ratio"))
            res["M0"] = self._apply_registration(res["M0"], reg)
            res["M1"] = self._apply_registration(res["M1"], reg)
            res["M2"] = self._apply_registration(res["M2"], reg)
            res["registration"] = reg
        toc(name="Reg")
        
        return res
    
    # ------------------------------------------------------------
    # Full video processing
    # ------------------------------------------------------------
    
    def process_moments(self, parameters, mp4_path = None, 
                        return_numpy = False, holodoppler_path = True):
        """Process entire video"""
        
        batch_size = parameters["batch_size"]
        batch_stride = parameters["batch_stride"]
        first_frame = parameters["first_frame"]
        end_frame = parameters.get("end_frame", 0)
        
        if end_frame <= 0:
            if self.file_reader.ext == ".holo":
                end_frame = self.file_reader.file_header["num_frames"]
            else:
                end_frame = self.file_reader.metadata["ImageCount"]
        
        if batch_stride >= (end_frame - first_frame):
            num_batch = 1 if batch_size <= (end_frame - first_frame) else 0
        else:
            num_batch = int((end_frame - first_frame) / batch_stride)
        
        if num_batch <= 0:
            return None
        
        out_list = []
        
        # Debug setup
        debug_manager = DebugPlotterManager(parameters) if parameters.get("debug") else None
        
        if parameters.get("debug") and debug_manager is not None:
            import threading
            import queue
            
            debug_results = {}
            res_store = {}
            lock = threading.Lock()
            debug_queue = queue.Queue(maxsize=14)
            stop_event = threading.Event()
            
            def plotting_worker():
                while not stop_event.is_set() or not debug_queue.empty():
                    try:
                        i = debug_queue.get(timeout=0.1)
                        with lock:
                            res = res_store.pop(i)
                        out = debug_manager.plot_all(res)
                        with lock:
                            debug_results[i] = out
                        debug_queue.task_done()
                    except queue.Empty:
                        continue
            
            debug_thread = threading.Thread(target=plotting_worker, daemon=True)
            debug_thread.start()
        
        # Registration reference
        if parameters.get("image_registration"):
            frames_reg = self.read_frames(first_frame, parameters["batch_size_registration"])
            M0_reg = self.render_moments(parameters, frames=frames_reg)["M0"]
            M0_reg = gaussian_flatfield(M0_reg, parameters["registration_flatfield_gw"], 
                                         self.bm.gaussian_filter)
            M0_reg = self.bm.to_backend(M0_reg)
        else:
            M0_reg = None
        
        coefs_list = [None] * num_batch if parameters.get("shack_hartmann") else None
        reg_list = [None] * num_batch if parameters.get("image_registration") else None
        
        # Main processing loop with GPU streaming if enabled
        if self.backend_name =="cupy":
            self._process_gpu_streaming(parameters, num_batch, first_frame, batch_stride,
                                        batch_size, M0_reg, out_list, coefs_list, reg_list,
                                        debug_manager, debug_queue, res_store, lock if parameters.get("debug") else None)
        elif self.backend_name =="cupyRAM":
            self._process_gpu_streaming_onram(parameters, num_batch, first_frame, batch_stride,
                                    batch_size, M0_reg, out_list, coefs_list, reg_list,
                                    debug_manager, debug_queue, res_store, lock if parameters.get("debug") else None)
        else:
            self._process_cpu(parameters, num_batch, first_frame, batch_stride,
                             batch_size, M0_reg, out_list, coefs_list, reg_list,
                             debug_manager, debug_queue, res_store, lock if parameters.get("debug") else None)
        
        # Stack results
        vid = self.bm.xp.stack(out_list, axis=3)
        
        # Cleanup debug
        if parameters.get("debug") and debug_manager is not None:
            debug_queue.join()
            stop_event.set()
            debug_thread.join()
            debug_manager.close_all()
            
        # Convert to numpy for saving
        vid_np = self.bm.to_numpy(vid)
        for i in range(num_batch) :
            coefs_list[i] =self.bm.to_numpy(coefs_list[i])
            reg_list[i] =self.bm.to_numpy(reg_list[i])
        if parameters["debug"]:
            vid_debug = {
                key: np.moveaxis(np.stack([self.bm.to_numpy(debug_results[k][key]) for k in range(num_batch)]), 0, -1)
                for key in debug_results[0]
            }
        else:
            vid_debug = {}
        
        # Post-processing: spatial transforms
        if parameters.get("square"):
            m = max(vid_np.shape[0], vid_np.shape[1])
            vid_np = self._resize_to_square(vid_np, m, m)
        if parameters.get("transpose"):
            vid_np = np.transpose(vid_np, axes=(1, 0, 2, 3))
        if parameters.get("flip_x"):
            vid_np = np.flip(vid_np, axis=1)
        if parameters.get("flip_y"):
            vid_np = np.flip(vid_np, axis=0)
        
        # Save outputs
        self._save_outputs(mp4_path, holodoppler_path, vid_np, vid_debug, parameters,
                          reg_list, coefs_list, end_frame, first_frame, num_batch)
        
        self.close_file()
        plt.close('all')
        
        if return_numpy:
            return vid_np
        
        return None
    
    def _process_cpu(self, parameters, num_batch, first_frame, batch_stride,
                     batch_size, M0_reg, out_list, coefs_list, reg_list,
                     debug_manager, debug_queue, res_store, lock):
        """CPU processing loop"""
        for i in tqdm(range(num_batch)):
            frames = self.read_frames(first_frame + i * batch_stride, batch_size)
            res = self.render_moments(parameters, frames=frames, registration_ref=M0_reg)
            
            if res is None:
                break
            
            out_list.append(self.bm.xp.stack([res["M0"], res["M1"], res["M2"]], axis=2))
            
            if "coefs" in res and coefs_list is not None:
                coefs_list[i] = res["coefs"]
            if "registration" in res and reg_list is not None:
                reg_list[i] = res["registration"]
            
            if debug_manager is not None and lock is not None:
                with lock:
                    res_store[i] = res
                debug_queue.put(i)
    
    def _process_gpu_streaming(self, parameters, num_batch, first_frame, batch_stride,
                               batch_size, M0_reg, out_list, coefs_list, reg_list,
                               debug_manager, debug_queue, res_store, lock):
        """GPU streaming processing loop"""
        import cupy as cp
        
        stream_h2d = cp.cuda.Stream(non_blocking=True)
        stream_compute = cp.cuda.Stream(non_blocking=True)
        
        # Prefetch first batch
        frames_next = self.read_frames(first_frame, batch_size)
        with stream_h2d:
            d_frames_next = cp.asarray(frames_next)
        
        for i in tqdm(range(num_batch)):
            d_frames = d_frames_next
            
            # Prefetch next batch
            if i + 1 < num_batch:
                with stream_h2d:
                    frames_next = self.read_frames(
                        first_frame + (i + 1) * batch_stride, batch_size
                    )
                    d_frames_next = cp.asarray(frames_next)
            
            # Compute current batch
            with stream_compute:
                res = self.render_moments(parameters, frames=d_frames, registration_ref=M0_reg)
            
            if res is None:
                break
            
            out_list.append(cp.stack([res["M0"], res["M1"], res["M2"]], axis=2))
            
            if "coefs" in res and coefs_list is not None:
                coefs_list[i] = res["coefs"]
            if "registration" in res and reg_list is not None:
                reg_list[i] = res["registration"]
            
            if debug_manager is not None and lock is not None:
                with lock:
                    res_store[i] = res
                debug_queue.put(i)
            
            stream_compute.synchronize()
        
        stream_h2d.synchronize()
        cp.cuda.Device().synchronize()
        
    def _process_gpu_streaming_onram(self, parameters, num_batch, first_frame, batch_stride,
                                    batch_size, M0_reg, out_list, coefs_list, reg_list,
                                    debug_manager, debug_queue, res_store, lock):
        """GPU streaming processing loop with CPU RAM prefetch queue."""
        import queue, threading
        import cupy as cp
        from tqdm import tqdm

        frame_queue = queue.Queue(maxsize=4)
        stop_reader = threading.Event()

        def reader():
            frame_idx = first_frame
            for i in range(num_batch):
                if stop_reader.is_set():
                    break
                frames = self.read_frames(frame_idx, batch_size)
                frame_queue.put((i, frames))
                frame_idx += batch_stride
            frame_queue.put(None)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        stream_h2d = cp.cuda.Stream(non_blocking=True)
        stream_compute = cp.cuda.Stream(non_blocking=True)

        item = frame_queue.get()
        if item is None:
            return

        _, frames = item
        with stream_h2d:
            d_frames = cp.asarray(frames)
        stream_h2d.synchronize()

        for i in tqdm(range(num_batch)):
            next_item = frame_queue.get() if i + 1 < num_batch else None

            d_frames_next = None
            if next_item is not None:
                _, frames_next = next_item
                with stream_h2d:
                    d_frames_next = cp.asarray(frames_next)

            with stream_compute:
                res = self.render_moments(parameters, frames=d_frames, registration_ref=M0_reg)

            stream_compute.synchronize()

            if res is None:
                break

            out_list.append(cp.stack([res["M0"], res["M1"], res["M2"]], axis=2))

            if "coefs" in res and coefs_list is not None:
                coefs_list[i] = res["coefs"]
            if "registration" in res and reg_list is not None:
                reg_list[i] = res["registration"]

            if debug_manager is not None and debug_queue is not None and lock is not None:
                with lock:
                    res_store[i] = res
                debug_queue.put(i)

            if d_frames_next is not None:
                stream_h2d.synchronize()
                d_frames = d_frames_next

        stop_reader.set()
        reader_thread.join(timeout=5)

        stream_h2d.synchronize()
        stream_compute.synchronize()
        cp.cuda.Device().synchronize()
    
    def _save_outputs(self, mp4_path, holodoppler_path, vid, vid_debug, parameters,
                      reg_list, coefs_list, end_frame, first_frame, num_batch):
        """Save outputs to disk"""
        
        # if h5_path is not None:
            # with h5py.File(h5_path, "w") as f:
            #     f.create_dataset("moment0", data=vid[:, :, :, 0])
            #     f.create_dataset("moment1", data=vid[:, :, :, 1])
            #     f.create_dataset("moment2", data=vid[:, :, :, 2])
            #     f.create_dataset("HD_parameters", data=json.dumps(parameters))
            #     if reg_list is not None and any(r is not None for r in reg_list):
            #         f.create_dataset("registration", data=self.bm.to_numpy(self.bm.xp.array(reg_list)))
            #     if coefs_list is not None and any(c is not None for c in coefs_list):
            #         f.create_dataset("zernike_coefs_radians", data=self.bm.to_numpy(self.bm.xp.array(coefs_list)))
                
            #     # Metadata
            #     if hasattr(self, 'holo_header'):
            #         f.create_dataset("holo_header", data=json.dumps(self.holo_header))
            #         f.create_dataset("holo_footer", data=json.dumps(self.holo_footer))
            #     elif hasattr(self, 'cine_metadata'):
            #         f.create_dataset("cine_metadata", data=json.dumps(self.cine_metadata))
                
            #     try:
            #         git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            #         f.create_dataset("git_commit", data=git_commit)
            #     except Exception:
            #         pass
        fps = num_batch / (end_frame - first_frame) * parameters["sampling_freq"]
        fps = min(fps, 65)
            
        if mp4_path is not None:
            self._write_video(mp4_path, vid[:, :, 0, :], fps)
        
        if holodoppler_path is not None:
            self._save_holodoppler_output(vid, vid_debug, parameters, fps,
                                          reg_list, coefs_list, end_frame, first_frame, num_batch)
    
    def _write_video(self, path, frames, fps, fourcc="mp4v"):
        """Write grayscale or color video."""
        if frames.ndim == 3:
            # (H, W, N)
            h, w, n = frames.shape
            is_color = False
        elif frames.ndim == 4:
            # (H, W, C, N)
            h, w, c, n = frames.shape
            is_color = c > 1
        else:
            raise ValueError(f"Unsupported frame shape: {frames.shape}")
        out = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (w, h),
            isColor=is_color,
        )
        for i in range(n):
            if frames.ndim == 3:
                frame = normalize_image(frames[:, :, i])
            else:
                frame = normalize_image(frames[:, :, :, i])
            out.write(frame)
        out.release()
    
    def _save_holodoppler_output(self, vid, vid_debug, parameters, fps, reg_list, coefs_list, end_frame, first_frame, num_batch):
        """Save complete Holodoppler output directory"""
        import subprocess
        import json
        
        base_name = os.path.splitext(os.path.basename(self.file_reader.file_path))[0]
        dir_name = f"{base_name}_HD"
        parent_dir = os.path.dirname(self.file_reader.file_path)
        full_path = os.path.join(parent_dir, base_name, dir_name)
        os.makedirs(full_path, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["png", "mp4", "avi", "json", "h5"]:
            os.makedirs(os.path.join(full_path, subdir), exist_ok=True)
        
        def save_pair(name, frames, png=True):
            self._write_video(os.path.join(full_path, "mp4", f"{name}.mp4"), frames, fps, "mp4v")
            self._write_video(os.path.join(full_path, "avi", f"{name}.avi"), frames, fps, "MJPG")
            if png:
                plt.imsave(os.path.join(full_path, "png", f"{name}.png"), np.mean(frames, axis=-1), cmap="gray")
        
        save_pair("moment_0", vid[:, :, 0, :])
        save_pair("moment_1", vid[:, :, 1, :])
        save_pair("moment_2", vid[:, :, 2, :])
        save_pair("moment_0_flatfield", flatfield3D(vid[:, :, 0, :], parameters["registration_flatfield_gw"]))
        
        for key, video_debug in vid_debug.items():
            save_pair(f"debug_{key}", video_debug, png=False)

                
            
        
        # Save JSON parameters
        with open(os.path.join(full_path, "json", "parameters_holodoppler.json"), "w") as f:
            json.dump(parameters, f, indent=4)
        
        # Save version info
        with open(os.path.join(full_path, "git_version.txt"), "w") as f:
            f.write(f"Holodoppler pipeline version: {self.pipeline_version}\n")
            f.write(f"Holodoppler backend: {self.backend_name}\n")
            try:
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                f.write(f"Git commit hash: {git_commit}\n")
            except Exception:
                f.write("Git commit hash: Not available\n")
        with open(os.path.join(full_path, "version_holodoppler.txt"), "w") as f:
            f.write(f"py{self.__version__}")
        with open(os.path.join(full_path, "info_holodoppler.txt"), "w") as f:
            f.write(f"py{self.__version__}  {self.backend_name}  {self.pipeline_version}")
            
        # copy camera aquisition metadata information if any
        if self.file_reader.ext == ".holo":
            with open(os.path.join(full_path, "version_holovibes.txt"), "w") as f:
                f.write(f"{self.file_reader.file_footer.get('info',{}).get('holovibes_version', 'unknown')}")
            with open(os.path.join(full_path, "json", "holovibes_footer.json"), "w") as f:
                json.dump(self.file_reader.file_footer, f, indent=4)
                
        # Dispose of GPU arrays to free memory
        # if "cupy" in self.bm.backend_name:
        #     import cupy as cp
        #     del d_frames
        #     if 'd_frames_next' in locals():
        #         del d_frames_next
        #     cp.cuda.Device().synchronize()
        #     cp.get_default_memory_pool().free_all_blocks()
        #     cp.get_default_pinned_memory_pool().free_all_blocks()
        
        plt.close('all') # close any open figures to free memory
        
        # Save HDF5
        with h5py.File(os.path.join(full_path, "h5", f"{dir_name}_output.h5"), "w") as f:
            f.create_dataset("moment0", data=vid[:, :, :, 0])
            f.create_dataset("moment1", data=vid[:, :, :, 1])
            f.create_dataset("moment2", data=vid[:, :, :, 2])
            f.create_dataset("HD_parameters", data=json.dumps(parameters))
            f.create_dataset("HD_info", data=f"py{self.__version__}  {self.backend_name}  {self.pipeline_version}")
            if parameters["image_registration"]:
                reg_array = np.array(reg_list, dtype=np.float32)
                f.create_dataset("registration", data=reg_array)
            if parameters["shack_hartmann"] and parameters["shack_hartmann_zernike_fit"] and coefs_list is not None:
                coefs_zernike = np.stack(coefs_list).astype(np.float32)
                dset = f.create_dataset("zernike_coefs_radians", data=coefs_zernike)
                dset.attrs["noll_indices"] = parameters["shack_hartmann_zernike_fit_modes"]

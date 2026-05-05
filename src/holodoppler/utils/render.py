from cupy.cuda.nvtx import RangePush, RangePop
import matplotlib
import numpy as np
import traceback
import h5py
from tqdm import tqdm
import cv2
import json
import os
from cupy.cuda.nvtx import RangePush, RangePop
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as np_gaussian_filter
from scipy.ndimage import gaussian_filter1d

import cupy as cp
import cupyx.scipy.fft as cp_fft


class Renderer():
    # ------------------------------------------------------------
    # One batch full pipeline
    # ------------------------------------------------------------

    def render_moments(self, parameters, frames = None, registration_ref = None, tictoc = False):
        RangePush("render_moments")
        
        def tic():
            if tictoc:
                return time.perf_counter(), time.process_time()
        def toc(t1, name="", arr=None):
            if tictoc:
                print(name)
                t2 = time.perf_counter(), time.process_time()
                print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
                # print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
                if arr is not None and isinstance(arr, (np.ndarray, cp.ndarray)): 
                    print(arr.dtype)
        
        if frames is None:
            frames = self.read_frames(parameters["first_frame"], parameters["batch_size"])
        if frames is None:
            raise RuntimeError("Could not read frames properly")

        nt, ny, nx = frames.shape
        
        res = {} # intitialze result dict

        from collections import defaultdict

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

                batch = {
                    k: self.xp.sum(self.xp.stack(v),axis=0) / self.batch_size 
                    for k, v in self.buffers.items()
                }
                self.buffers.clear()
                return batch

        subapsacc = Accumulator(parameters["shack_hartmann_accumulation"], self.xp)

        mainacc = Accumulator(parameters["accumulation"], self.xp)

        t1 = tic()
        if parameters["shack_hartmann"] :
            
            t2 = tic()
            sub_batch_size = (nt // subapsacc.batch_size)
            sub_batch_stride = sub_batch_size
            assert sub_batch_size > 0
            for it in range(subapsacc.batch_size) :
                frames_sub = frames[sub_batch_stride*it:sub_batch_stride*it+sub_batch_size]
                if parameters["spatial_propagation"] == "Fresnel" :
                    if (not "Fresnel_in" in self.kernels):
                        self._build_fresnel_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
                    U_subaps = self._shack_hartmann_constructsubapsimages(frames_sub, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], parameters["z"], parameters["low_freq"], parameters["high_freq"], parameters["sampling_freq"], frames_sub.shape[0], parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"], parameters["svd_threshold"]) # construct small images from the sub apertures of the Shack-Hartmann sensor
                elif parameters["spatial_propagation"] == "AngularSpectrum" :
                    if (not "AngularSpectrum" in self.kernels):
                        self._build_angular_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
                    U_subaps = self._shack_hartmann_constructsubapsimages_angular_spectrum(frames_sub, parameters["pixel_pitch"], parameters["pixel_pitch"], parameters["wavelength"], parameters["z"], parameters["low_freq"], parameters["high_freq"], parameters["sampling_freq"], nt, parameters["shack_hartmann_nx_subap"], parameters["shack_hartmann_ny_subap"], parameters["svd_threshold"]) # construct small images from the sub apertures of the Shack-Hartmann sensor
                b = subapsacc.add({"U_subaps":U_subaps})
            if b is not None:
                U_subaps = b["U_subaps"]
            toc(t2, "Usubaps calculation",U_subaps)

            if parameters["debug"]:
                res["U_subaps"] = U_subaps

            t2 = tic()
            shifts_y, shifts_x = self._shack_hartmann_displacement_calculation(U_subaps, self.xp, pupil_threshold = parameters["shack_hartmann_pupil_threshold"], deviation_threshold = parameters["shack_hartmann_deviation_threshold"], shifts_pixel_range_threshold = parameters["shack_hartmann_shifts_pixel_range_threshold"], ref = None) # get the shifts in pixels in the subapertures images
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
            toc(t2, "Shack-Hartmann phase reconstruction time")
            
            phase_term = self.xp.exp(- 1j * phase) 
            phase_term = self.xp.nan_to_num(phase_term, nan=0.0) # completely mask the nan zone where the phase could'nt be estimated
            if parameters["zero_padding"]:
                phase_term = self.pad_array_centrally(phase_term, parameters["zero_padding"])
        
        toc(t1, "Shack-Hartmann total phase calculation time")
            
            
        t1 = tic()
        res_ = {}
        sub_batch_size = (nt // mainacc.batch_size)
        sub_batch_stride = sub_batch_size
        assert sub_batch_size > 0
        for it in range(mainacc.batch_size) :
            frames_sub = frames[sub_batch_stride*it:sub_batch_stride*it+sub_batch_size]
            if parameters["shack_hartmann"]:
                if parameters["spatial_propagation"] == "Fresnel" :
                    holograms = self._fresnel_transform_phase(frames_sub, phase_term, zero_padding = parameters["zero_padding"])            
                elif parameters["spatial_propagation"] == "AngularSpectrum" :
                    holograms = self._angular_spectrum_transform_phase(frames_sub, phase_term, zero_padding = parameters["zero_padding"])   
                if parameters["debug"]:
                    if parameters["spatial_propagation"] == "Fresnel" :
                        hologramsnotfixed = self._fresnel_transform(frames_sub, zero_padding = parameters["zero_padding"])
                    elif parameters["spatial_propagation"] == "AngularSpectrum" :
                        hologramsnotfixed = self._angular_spectrum_transform(frames_sub, zero_padding = parameters["zero_padding"]) 
                        
            elif parameters["spatial_propagation"] == "Fresnel":
                if (not "Fresnel_in" in self.kernels):
                    self._build_fresnel_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
                holograms = self._fresnel_transform(frames_sub, zero_padding = parameters["zero_padding"])
            elif parameters["spatial_propagation"] == "AngularSpectrum":
                if (not "AngularSpectrum" in self.kernels):
                    self._build_angular_kernel(parameters["z"],parameters["pixel_pitch"],parameters["wavelength"], ny, nx, zero_padding = parameters["zero_padding"])
                holograms = self._angular_spectrum_transform(frames_sub, zero_padding = parameters["zero_padding"])
            
            holograms_f = self._svd_filter(holograms, parameters["svd_threshold"])

            spectrum_f = self._fourier_time_transform(holograms_f)

            idxs, freqs = self._frequency_symmetric_filtering(frames_sub.shape[0], parameters["sampling_freq"], parameters["low_freq"], parameters["high_freq"])
            psd = self.xp.abs(spectrum_f[idxs,:,:]) ** 2

            if parameters["debug"]:
                res_["spectrum_line"] = self.xp.mean(self.xp.abs(spectrum_f[:,:,:]) ** 2, axis=(-1, -2)) # spectrum line at the center of the image for debug purposes
                res_["freqs"] = freqs

            res_["M0"] = self._moment(psd, freqs, 0)
            res_["M1"] = self._moment(psd, freqs, 1)
            res_["M2"] = self._moment(psd, freqs, 2)
            
            if parameters["shack_hartmann"] and parameters["debug"]: # to compute the res without the phase correction for debug purposes
                hologramsnotfixed_f = self._svd_filter(hologramsnotfixed, parameters["svd_threshold"])
                spectrumnotfixed_f = self._fourier_time_transform(hologramsnotfixed_f)
                psdnotfixed = self.xp.abs(spectrumnotfixed_f[idxs,:,:]) ** 2
                M0notfixed = self._moment(psdnotfixed, freqs, 0)
                res_["M0notfixed"] = M0notfixed

            b = mainacc.add(res_)
        if b is not None:
            res.update(b)

        toc(t1, "Spatial and temporal transforms total calculation time")
        
        # --- Register current batch ---  
        if parameters["image_registration"] and registration_ref is not None:
            t2 = tic()
            M0_ff = self._flatfield(res["M0"], parameters["registration_flatfield_gw"])
            reg = self._registration(registration_ref, M0_ff, parameters["registration_disc_ratio"])
            res["M0"] = self._applyregistration(res["M0"], reg, self.xp)
            res["M1"] = self._applyregistration(res["M1"], reg, self.xp)
            res["M2"] = self._applyregistration(res["M2"], reg, self.xp)
            res["registration"] = reg
            toc(t2, "Image registration time", reg)
            
        RangePop()
        
        toc(t1, "total render_moments time")
        
        
        return res
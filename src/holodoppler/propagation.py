"""
Propagation kernels (Fresnel and Angular Spectrum)
"""

import numpy as np


class PropagationKernels:
    """Manages propagation kernels for Fresnel and Angular Spectrum methods"""
    
    def __init__(self, backend_manager):
        self.bm = backend_manager
        self.kernels = {}
    
    def build_fresnel_kernel_in(self, z, pixel_pitch, wavelength, ny, nx):
        """Build input Fresnel kernel"""
        xp = self.bm.xp
        
        if isinstance(pixel_pitch, (float, int)):
            pixel_pitch = (pixel_pitch, pixel_pitch)
        
        ppy, ppx = pixel_pitch
        
        y = (xp.arange(0, ny) - xp.round(ny / 2)) * ppy
        x = (xp.arange(0, nx) - xp.round(nx / 2)) * ppx
        
        X, Y = xp.meshgrid(x, y)
        
        kernel = xp.exp(
            1j * xp.pi / (wavelength * z) * (X ** 2 + Y ** 2)
        ).astype(xp.complex64)
        
        self.kernels["Fresnel_in"] = kernel[xp.newaxis, ...]
    
    def build_fresnel_kernel_out(self, z, pixel_pitch, wavelength, ny, nx):
        """Build output Fresnel kernel"""
        xp = self.bm.xp
        
        if isinstance(pixel_pitch, (float, int)):
            pixel_pitch = (pixel_pitch, pixel_pitch)
        
        ppy, ppx = pixel_pitch
        
        fx = xp.fft.fftfreq(nx, d=ppx)
        fx = xp.fft.fftshift(fx)
        fy = xp.fft.fftfreq(ny, d=ppy)
        fy = xp.fft.fftshift(fy)
        FX, FY = xp.meshgrid(fx, fy)
        
        X = wavelength * z * FX
        Y = wavelength * z * FY
        phase = 1j * xp.pi / (wavelength * z) * (X**2 + Y**2)
        kernel = (xp.exp(1j * 2 * xp.pi / wavelength * z) / 
                  (1j * wavelength * z) * xp.exp(phase)).astype(xp.complex64)
        
        self.kernels["Fresnel_out"] = kernel[xp.newaxis, ...]
    
    def build_fresnel_kernel(self, z, pixel_pitch, wavelength, ny, nx, zero_padding=None):
        """Build both Fresnel kernels"""
        self.build_fresnel_kernel_in(z, pixel_pitch, wavelength, ny, nx)
        self.build_fresnel_kernel_out(z, pixel_pitch, wavelength, ny, nx)
        
        if zero_padding:
            from .utils import pad_array_centrally
            self.kernels["Fresnel_in"] = pad_array_centrally(
                self.kernels["Fresnel_in"], zero_padding, self.bm.xp
            )
    
    def build_angular_kernel(self, z, pixel_pitch, wavelength, ny, nx, zero_padding=None):
        """Build Angular Spectrum kernel"""
        xp = self.bm.xp
        
        if isinstance(pixel_pitch, (float, int)):
            pixel_pitch = (pixel_pitch, pixel_pitch)
        
        ppy, ppx = pixel_pitch
        
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
            from .utils import pad_array_centrally
            self.kernels["AngularSpectrum"] = pad_array_centrally(
                self.kernels["AngularSpectrum"], zero_padding, self.bm.xp
            )
    
    def fresnel_transform(self, frames, zero_padding=False, use_output_kernel=True):
        """Apply Fresnel transform"""
        xp = self.bm.xp
        fft = self.bm.fft
        
        if zero_padding:
            from .utils import pad_array_centrally
            frames = pad_array_centrally(frames, zero_padding, xp)
        
        result = fft.fftshift(
            fft.fft2(frames * self.kernels["Fresnel_in"], axes=(-1, -2), norm="ortho"),
            axes=(-1, -2)
        )
        
        if use_output_kernel and "Fresnel_out" in self.kernels:
            result = result * self.kernels["Fresnel_out"]
        
        return result
    
    def fresnel_transform_with_phase(self, frames, phase_term, zero_padding=False):
        """Apply Fresnel transform with phase correction"""
        xp = self.bm.xp
        fft = self.bm.fft
        
        if zero_padding:
            from .utils import pad_array_centrally
            frames = pad_array_centrally(frames, zero_padding, xp)
        
        result = fft.fftshift(
            fft.fft2(frames * self.kernels["Fresnel_in"] * phase_term, 
                    axes=(-1, -2), norm="ortho"),
            axes=(-1, -2)
        )
        
        if "Fresnel_out" in self.kernels:
            result = result * self.kernels["Fresnel_out"]
        
        return result
    
    def angular_spectrum_transform(self, frames, zero_padding=False):
        """Apply Angular Spectrum transform"""
        xp = self.bm.xp
        fft = self.bm.fft
        
        if zero_padding:
            from .utils import pad_array_centrally
            frames = pad_array_centrally(frames, zero_padding, xp)
        
        tmp = fft.fft2(frames, axes=(-1, -2), norm="ortho")
        tmp = tmp * fft.fftshift(self.kernels["AngularSpectrum"], axes=(-1, -2))
        
        return fft.ifft2(tmp, axes=(-1, -2), norm="ortho")
    
    def angular_spectrum_transform_with_phase(self, frames, phase_term, zero_padding=False):
        """Apply Angular Spectrum transform with phase correction"""
        xp = self.bm.xp
        fft = self.bm.fft
        
        if zero_padding:
            from .utils import pad_array_centrally
            frames = pad_array_centrally(frames, zero_padding, xp)
        
        return fft.ifft2(
            fft.fft2(frames, axes=(-1, -2)) * 
            fft.fftshift(self.kernels["AngularSpectrum"] * phase_term, axes=(-1, -2))
        )
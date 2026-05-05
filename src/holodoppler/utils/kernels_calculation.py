import numpy as np

class KernelsCalculator():
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
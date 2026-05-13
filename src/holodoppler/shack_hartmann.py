"""
Shack-Hartmann wavefront sensing
"""

import numpy as np


class ShackHartmann:
    """Shack-Hartmann wavefront sensor processing"""
    
    def __init__(self, backend_manager, propagation_kernels, filtering):
        self.bm = backend_manager
        self.prop = propagation_kernels
        self.filter = filtering
    
    def construct_subapertures_fresnel(self, U0, dx, dy, wavelength, z_prop, f0, f1, fs,
                                        time_window, nx_subabs, ny_subabs, svd_threshold):
        """Construct subaperture images using Fresnel propagation"""
        xp = self.bm.xp
        Nz, Ny, Nx = U0.shape
        sub_ny, sub_nx = Ny // ny_subabs, Nx // nx_subabs
        
        # Frequency filtering
        idxs, _ = self.filter.frequency_symmetric_filtering(time_window, fs, f0, high_freq=f1)
        
        # Fresnel kernel
        if "Fresnel_in" not in self.prop.kernels:
            self.prop.build_fresnel_kernel(z_prop, (dy, dx), wavelength, Ny, Nx)
        
        from .utils import crop_array_centrally
        Qin = crop_array_centrally(self.prop.kernels["Fresnel_in"], (Ny, Nx), xp)
        
        U_prop_qin = U0 * Qin
        
        # Crop to tiling region
        crop_ny, crop_nx = sub_ny * ny_subabs, sub_nx * nx_subabs
        y0, x0 = (Ny - crop_ny) // 2, (Nx - crop_nx) // 2
        U_prop_qin = U_prop_qin[:, y0:y0 + crop_ny, x0:x0 + crop_nx]
        
        # Reshape into subapertures
        U_subap_all = (U_prop_qin
                      .reshape(Nz, ny_subabs, sub_ny, nx_subabs, sub_nx)
                      .transpose(1, 3, 2, 4, 0))
        
        # SVD filtering
        U_subap_all = self.filter.svd_filter_batched(U_subap_all, svd_threshold)
        
        B = ny_subabs * nx_subabs
        U_subap_all = U_subap_all.reshape(B, sub_ny, sub_nx, Nz).transpose(0, 3, 1, 2)
        
        # FFT2 + temporal FFT
        U_f2 = xp.fft.fftshift(xp.fft.fft2(U_subap_all, axes=(-2, -1)), axes=(-2, -1))
        U_ft = xp.fft.fft(U_f2, axis=1)[:, idxs, :, :]
        
        # Power spectrum
        M0 = xp.mean(xp.abs(U_ft) ** 2, axis=1)
        
        return M0.reshape(ny_subabs, nx_subabs, sub_ny, sub_nx).astype(xp.float32)
    
    def construct_subapertures_angular(self, U0, dx, dy, wavelength, z_prop, f0, f1, fs,
                                        time_window, nx_subabs, ny_subabs, svd_threshold):
        """Construct subaperture images using Angular Spectrum propagation"""
        xp = self.bm.xp
        Nz, Ny, Nx = U0.shape
        sub_ny, sub_nx = Ny // ny_subabs, Nx // nx_subabs
        
        freq_idxs, _ = self.filter.frequency_symmetric_filtering(time_window, fs, f0, high_freq=f1)
        
        # Angular spectrum kernel
        if "AngularSpectrum" not in self.prop.kernels:
            self.prop.build_angular_kernel(z_prop, (dy, dx), wavelength, Ny, Nx)
        
        from .utils import crop_array_centrally
        H = crop_array_centrally(self.prop.kernels["AngularSpectrum"], (Ny, Nx), xp)
        if H.ndim == 2:
            H = H[None, :, :]
        
        U_fft = self.prop.bm.fft.fft2(U0, axes=(-2, -1)) * self.prop.bm.fft.fftshift(H, axes=(-2, -1))
        
        crop_ny, crop_nx = sub_ny * ny_subabs, sub_nx * nx_subabs
        y0, x0 = (Ny - crop_ny) // 2, (Nx - crop_nx) // 2
        U_fft = U_fft[:, y0:y0 + crop_ny, x0:x0 + crop_nx]
        U_fft = self.prop.bm.fft.fftshift(U_fft, axes=(-2, -1))
        
        # Split into subapertures
        U_subap_all = (U_fft.reshape(Nz, ny_subabs, sub_ny, nx_subabs, sub_nx)
                      .transpose(0, 1, 3, 2, 4))
        
        U_subap_all = self.prop.bm.fft.ifftshift(U_subap_all, axes=(-2, -1))
        U_subap_all = self.prop.bm.fft.ifft2(U_subap_all, axes=(-2, -1))
        
        U_subap_all = U_subap_all.transpose(1, 2, 3, 4, 0)
        U_subap_all = xp.ascontiguousarray(U_subap_all)
        
        # SVD filtering
        U_subap_all = self.filter.svd_filter_batched(U_subap_all, svd_threshold)
        
        B = ny_subabs * nx_subabs
        U_subap_all = U_subap_all.reshape(B, sub_ny, sub_nx, Nz).transpose(0, 3, 1, 2)
        
        U_ft = self.filter.fourier_time_transform(U_subap_all)[:, freq_idxs, :, :]
        M0 = xp.mean(xp.abs(U_ft) ** 2, axis=1)
        
        return M0.reshape(ny_subabs, nx_subabs, sub_ny, sub_nx).astype(xp.float32)
    
    def calculate_displacements(self, U_subaps, pupil_threshold=1.0, deviation_threshold=3.0,
                                 shifts_range=20.0, ref=None):
        """Calculate subaperture displacements using phase correlation"""
        xp = self.bm.xp
        ny_s, nx_s, Ny, Nx = U_subaps.shape
        
        if ref is None:
            ref = U_subaps[ny_s // 2, nx_s // 2]
        
        moving_stack = U_subaps.reshape(ny_s * nx_s, Ny, Nx)
        
        ref_zm = ref - xp.mean(ref)
        moving_zm = moving_stack - xp.mean(moving_stack, axis=(1, 2), keepdims=True)
        
        F_ref = xp.fft.fft2(ref_zm)
        F_moving = xp.fft.fft2(moving_zm)
        cp_corr = F_moving * F_ref.conj()
        xcorr = xp.abs(xp.fft.fftshift(xp.fft.ifft2(cp_corr / (xp.abs(cp_corr) + 1e-12)), axes=(1, 2)))
        
        xcorr_2d = xcorr.reshape(-1, Ny * Nx)
        peaks = xp.argmax(xcorr_2d, axis=1)
        py = peaks // Nx
        px = peaks % Nx
        
        py = xp.clip(py, 1, Ny - 2)
        px = xp.clip(px, 1, Nx - 2)
        
        n = py.shape[0]
        idx = xp.arange(n)
        
        v0 = xcorr[idx, py, px]
        vm_y = xcorr[idx, py - 1, px]
        vp_y = xcorr[idx, py + 1, px]
        vm_x = xcorr[idx, py, px - 1]
        vp_x = xcorr[idx, py, px + 1]
        
        den_y = vm_y - 2 * v0 + vp_y + 1e-12
        den_x = vm_x - 2 * v0 + vp_x + 1e-12
        
        shift_y = py + 0.5 * (vm_y - vp_y) / den_y - Ny / 2
        shift_x = px + 0.5 * (vm_x - vp_x) / den_x - Nx / 2
        
        shift_y = shift_y.reshape(ny_s, nx_s)
        shift_x = shift_x.reshape(ny_s, nx_s)
        
        # Pupil mask for outlier rejection
        xs = xp.linspace(-1, 1, nx_s)
        ys = xp.linspace(-1, 1, ny_s)
        YY, XX = xp.meshgrid(ys, xs, indexing='ij')
        pupil_mask = (XX**2 + YY**2) <= pupil_threshold
        
        shift_y_flat = shift_y[pupil_mask]
        shift_x_flat = shift_x[pupil_mask]
        
        mean_y = xp.mean(shift_y_flat) if len(shift_y_flat) > 0 else 0
        mean_x = xp.mean(shift_x_flat) if len(shift_x_flat) > 0 else 0
        std_y = xp.std(shift_y_flat) if len(shift_y_flat) > 0 else 1
        std_x = xp.std(shift_x_flat) if len(shift_x_flat) > 0 else 1
        
        thresh_y = deviation_threshold * std_y
        thresh_x = deviation_threshold * std_x
        
        bad = ((xp.abs(shift_y - mean_y) > thresh_y) |
               (xp.abs(shift_x - mean_x) > thresh_x) |
               (xp.abs(shift_y) > shifts_range) |
               (xp.abs(shift_x) > shifts_range) |
               (~pupil_mask))
        
        shift_y[bad] = xp.nan
        shift_x[bad] = xp.nan
        
        return shift_y.astype(xp.float32), shift_x.astype(xp.float32)
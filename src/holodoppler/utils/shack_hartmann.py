from cupy.cuda.nvtx import RangePush, RangePop
import numpy as np
import cupy as cp
import scipy 
from scipy.interpolate import griddata

class ShackHartmannUtils():
    
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

    def _shack_hartmann_displacement_calculation(self, U_subabs, xp, pupil_threshold = 1.0, deviation_threshold = 3.0, shifts_pixel_range_threshold = 20.0, ref = None):
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

        # --- Pixel range filtering ---
        bad = (
            (xp.abs(shift_y - mean_y) > thresh_y) |
            (xp.abs(shift_x - mean_x) > thresh_x) |
            (xp.abs(shift_y) > shifts_pixel_range_threshold) |
            (xp.abs(shift_x) > shifts_pixel_range_threshold) |
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

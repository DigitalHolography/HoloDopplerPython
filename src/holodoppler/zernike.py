"""
Zernike polynomial generation and wavefront reconstruction
"""

import numpy as np


class ZernikeReconstructor:
    """Zernike polynomial-based wavefront reconstruction"""
    
    def __init__(self, backend_manager):
        self.bm = backend_manager
    
    def get_zernike_mode(self, mode_index, Nx, Ny, radius=2.0):
        """Generate Zernike mode on a grid"""
        xp = self.bm.xp
        
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
        
        if mode_index == 1:      # Piston
            Z[mask] = 1
        elif mode_index == 2:    # Tilt X
            Z[mask] = 2 * X[mask]
        elif mode_index == 3:    # Tilt Y
            Z[mask] = 2 * Y[mask]
        elif mode_index == 4:    # Defocus
            Z[mask] = xp.sqrt(3) * (2 * R[mask]**2 - 1)
        elif mode_index == 5:    # Astigmatism 45°
            Z[mask] = xp.sqrt(6) * R[mask]**2 * xp.sin(2 * Theta[mask])
        elif mode_index == 6:    # Astigmatism 0°
            Z[mask] = xp.sqrt(6) * R[mask]**2 * xp.cos(2 * Theta[mask])
        elif mode_index == 7:    # Coma Y
            Z[mask] = xp.sqrt(8) * (3*R[mask]**3 - 2*R[mask]) * xp.sin(Theta[mask])
        elif mode_index == 8:    # Coma X
            Z[mask] = xp.sqrt(8) * (3*R[mask]**3 - 2*R[mask]) * xp.cos(Theta[mask])
        elif mode_index == 9:    # Trefoil Y
            Z[mask] = xp.sqrt(8) * R[mask]**3 * xp.sin(3 * Theta[mask])
        elif mode_index == 10:   # Trefoil X
            Z[mask] = xp.sqrt(8) * R[mask]**3 * xp.cos(3 * Theta[mask])
        elif mode_index == 11:   # Spherical
            Z[mask] = xp.sqrt(5) * (6*R[mask]**4 - 6*R[mask]**2 + 1)
        elif mode_index == 12:   # Astigmatism higher 0°
            Z[mask] = xp.sqrt(10) * (4*R[mask]**4 - 3*R[mask]**2) * xp.cos(2 * Theta[mask])
        elif mode_index == 13:   # Astigmatism higher 45°
            Z[mask] = xp.sqrt(10) * (4*R[mask]**4 - 3*R[mask]**2) * xp.sin(2 * Theta[mask])
        elif mode_index == 14:   # Quadrafoil X
            Z[mask] = xp.sqrt(10) * R[mask]**4 * xp.cos(4 * Theta[mask])
        elif mode_index == 15:   # Quadrafoil Y
            Z[mask] = xp.sqrt(10) * R[mask]**4 * xp.sin(4 * Theta[mask])
        elif mode_index == 16:   # Secondary coma X
            Z[mask] = xp.sqrt(12) * (10*R[mask]**5 - 12*R[mask]**3 + 3*R[mask]) * xp.cos(Theta[mask])
        elif mode_index == 17:   # Secondary coma Y
            Z[mask] = xp.sqrt(12) * (10*R[mask]**5 - 12*R[mask]**3 + 3*R[mask]) * xp.sin(Theta[mask])
        elif mode_index == 18:   # Secondary trefoil X
            Z[mask] = xp.sqrt(12) * (5*R[mask]**5 - 4*R[mask]**3) * xp.cos(3 * Theta[mask])
        elif mode_index == 19:   # Secondary trefoil Y
            Z[mask] = xp.sqrt(12) * (5*R[mask]**5 - 4*R[mask]**3) * xp.sin(3 * Theta[mask])
        elif mode_index == 20:   # Pentafoil X
            Z[mask] = xp.sqrt(12) * R[mask]**5 * xp.cos(5 * Theta[mask])
        elif mode_index == 21:   # Pentafoil Y
            Z[mask] = xp.sqrt(12) * R[mask]**5 * xp.sin(5 * Theta[mask])
        else:
            raise ValueError(f"Mode index {mode_index} not implemented")
        
        return Z.astype(xp.float32)
    
    def fit_zernike(self, ny, nx, pixel_pitch_y, pixel_pitch_x, wavelength,
                    shifts_y, shifts_x, zernike_modes):
        """Fit Zernike polynomials to displacement data"""
        xp = self.bm.xp
        # Use numpy for LSTQ (CUDA doesn't have batched lstsq efficiently)
        
        nysubabs, nxsubabs = shifts_y.shape
        
        slopes_y = (shifts_y) * wavelength / (pixel_pitch_y * (ny // nysubabs)) # z / z
        slopes_x = (shifts_x) * wavelength / (pixel_pitch_x * (nx // nxsubabs)) # z / z
        
        s = xp.stack([slopes_y, slopes_x])
        
        # Build gradient matrix
        n_modes = len(zernike_modes)
        G = xp.zeros((2, nysubabs, nxsubabs, n_modes), dtype=xp.float32)
        
        for k, idx in enumerate(zernike_modes):
            Z = self.get_zernike_mode(idx, nx, ny, radius=2)
            
            dZdx = xp.gradient(Z, pixel_pitch_x, axis=1)
            dZdy = xp.gradient(Z, pixel_pitch_y, axis=0)
            
            sub_ny = ny // nysubabs
            sub_nx = nx // nxsubabs
            
            for iy in range(nysubabs):
                y_start = iy * sub_ny
                y_end = y_start + sub_ny
                for ix in range(nxsubabs):
                    x_start = ix * sub_nx
                    x_end = x_start + sub_nx
                    dZdx_subap = dZdx[y_start:y_end, x_start:x_end]
                    dZdy_subap = dZdy[y_start:y_end, x_start:x_end]
                    G[0, iy, ix, k] = xp.nanmean(dZdy_subap)
                    G[1, iy, ix, k] = xp.nanmean(dZdx_subap)
        
        G *= wavelength / (2 * xp.pi)
        
        # Solve linear system
        A = G.reshape(-1, n_modes)
        b = s.reshape(-1)
        valid = ~xp.isnan(b) & ~xp.isnan(A).any(1)
        
        coefs, _, _, _ = xp.linalg.lstsq(A[valid], b[valid], rcond=None)
        
        # Reconstruct phase
        phase = xp.zeros((ny, nx), dtype=xp.float32)
        for idx, coef in zip(zernike_modes, coefs):
            Z = self.get_zernike_mode(idx, nx, ny, radius=2)
            phase += coef * Z
        
        return coefs.astype(xp.float32), phase.astype(xp.float32)
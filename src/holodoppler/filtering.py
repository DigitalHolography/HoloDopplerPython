"""
Filtering operations: SVD, frequency filtering
"""

import numpy as np


class Filtering:
    """Filtering operations for holographic data"""
    
    def __init__(self, backend_manager):
        self.bm = backend_manager
    
    def svd_filter(self, H, svd_threshold):
        """SVD filtering to remove tissue signal"""
        xp = self.bm.xp
        
        if svd_threshold < 0:
            return H
        
        sz = H.shape
        H2 = H.reshape((sz[0], sz[-1] * sz[-2])).T
        
        cov = H2.conj().T @ H2
        eps = 1e-12
        cov = cov + eps * xp.eye(cov.shape[0], dtype=cov.dtype)
        
        S, V = xp.linalg.eigh(cov)
        idx = xp.argsort(S)[::-1]
        V = V[:, idx]
        Vt = V[:, :svd_threshold]
        
        H2 -= H2 @ Vt @ Vt.conj().T
        return H2.T.reshape(sz)
    
    def tucker_filter(self, H, ranks=None, temporal_modes_to_remove=32):
        """
        Global Tucker/HOSVD filter.

        Parameters
        ----------
        H : complex tensor (T, X, Y)
        ranks : tuple
            Tucker ranks (rt, rx, ry)
        temporal_modes_to_remove : int
            Number of coherent temporal modes to suppress

        Returns
        -------
        Filtered tensor with coherent low-rank components removed.
        """
        
        if temporal_modes_to_remove <=0 :
            return H
        
        if ranks is None:
            ranks = H.shape

        xp = self.bm.xp

        T, X, Y = H.shape
        rt, rx, ry = ranks

        eps = 1e-12

        # ============================================================
        # Mode unfoldings
        # ============================================================

        Ht = H.reshape(T, X * Y)

        Hx = xp.transpose(H, (1, 0, 2)).reshape(X, T * Y)

        Hy = xp.transpose(H, (2, 0, 1)).reshape(Y, T * X)

        # ============================================================
        # Temporal factors
        # ============================================================

        Ct = Ht @ Ht.conj().T
        Ct += eps * xp.eye(T, dtype=H.dtype)

        St, Ut = xp.linalg.eigh(Ct)

        idx = xp.argsort(St)[::-1]
        Ut = Ut[:, idx[:rt]]

        # ============================================================
        # Spatial X factors
        # ============================================================

        Cx = Hx @ Hx.conj().T
        Cx += eps * xp.eye(X, dtype=H.dtype)

        Sx, Ux = xp.linalg.eigh(Cx)

        idx = xp.argsort(Sx)[::-1]
        Ux = Ux[:, idx[:rx]]

        # ============================================================
        # Spatial Y factors
        # ============================================================

        Cy = Hy @ Hy.conj().T
        Cy += eps * xp.eye(Y, dtype=H.dtype)

        Sy, Uy = xp.linalg.eigh(Cy)

        idx = xp.argsort(Sy)[::-1]
        Uy = Uy[:, idx[:ry]]

        # ============================================================
        # Core tensor
        # G = H ×1 Ut^H ×2 Ux^H ×3 Uy^H
        # ============================================================

        G = xp.einsum(
            'ti,xj,yk,txy->ijk',
            Ut.conj(),
            Ux.conj(),
            Uy.conj(),
            H
        )

        # ============================================================
        # Remove coherent temporal modes
        # ============================================================

        G[:temporal_modes_to_remove, :, :] = 0

        # ============================================================
        # Reconstruction
        # Hf = G ×1 Ut ×2 Ux ×3 Uy
        # ============================================================

        Hf = xp.einsum(
            'ti,xj,yk,ijk->txy',
            Ut,
            Ux,
            Uy,
            G
        )

        return Hf
    
    def hankel_tucker_filter(
        self,
        H,
        ranks=(8, 8, 8),
        temporal_modes_to_remove=1,
        hankel_length=16
    ):
        """
        Hankel-Tucker coherent artifact filter.

        Parameters
        ----------
        H : tensor (T, X, Y)
        ranks : tuple
            Tucker ranks
        temporal_modes_to_remove : int
            Number of temporal modes removed
        hankel_length : int
            Temporal Hankel embedding size / Controls frequency selectivity.Typical:hankel_length = T // 4

        Returns
        -------
        Filtered tensor
        """
        

        xp = self.bm.xp

        T, X, Y = H.shape

        L = hankel_length
        K = T - L + 1

        if K <= 1:
            return H

        # ============================================================
        # Hankel embedding
        # Output shape:
        # (L, K, X, Y)
        # ============================================================

        Hh = xp.zeros((L, K, X, Y), dtype=H.dtype)

        for k in range(K):
            Hh[:, k] = H[k:k + L]

        # ============================================================
        # Merge K and spatial dimensions for Tucker
        # shape -> (L, K*X, Y)
        # ============================================================

        Hm = Hh.reshape(L, K * X, Y)

        rt, rx, ry = ranks

        eps = 1e-12

        # ============================================================
        # Unfoldings
        # ============================================================

        Ht = Hm.reshape(L, (K * X) * Y)

        Hx = xp.transpose(Hm, (1, 0, 2)).reshape(K * X, L * Y)

        Hy = xp.transpose(Hm, (2, 0, 1)).reshape(Y, L * (K * X))

        # ============================================================
        # Temporal factors
        # ============================================================

        Ct = Ht @ Ht.conj().T
        Ct += eps * xp.eye(L, dtype=H.dtype)

        St, Ut = xp.linalg.eigh(Ct)

        idx = xp.argsort(St)[::-1]
        Ut = Ut[:, idx[:rt]]

        # ============================================================
        # Spatial factors
        # ============================================================

        Cx = Hx @ Hx.conj().T
        Cx += eps * xp.eye(K * X, dtype=H.dtype)

        Sx, Ux = xp.linalg.eigh(Cx)

        idx = xp.argsort(Sx)[::-1]
        Ux = Ux[:, idx[:rx]]

        Cy = Hy @ Hy.conj().T
        Cy += eps * xp.eye(Y, dtype=H.dtype)

        Sy, Uy = xp.linalg.eigh(Cy)

        idx = xp.argsort(Sy)[::-1]
        Uy = Uy[:, idx[:ry]]

        # ============================================================
        # Core tensor
        # ============================================================

        G = xp.einsum(
            'li,pj,yk,lpy->ijk',
            Ut.conj(),
            Ux.conj(),
            Uy.conj(),
            Hm
        )

        # ============================================================
        # Remove coherent temporal modes
        # ============================================================

        G[:temporal_modes_to_remove, :, :] = 0

        # ============================================================
        # Reconstruction
        # ============================================================

        Hmf = xp.einsum(
            'li,pj,yk,ijk->lpy',
            Ut,
            Ux,
            Uy,
            G
        )

        # ============================================================
        # Undo merged dimensions
        # ============================================================

        Hhf = Hmf.reshape(L, K, X, Y)

        # ============================================================
        # Hankel averaging reconstruction
        # ============================================================

        Hout = xp.zeros((T, X, Y), dtype=H.dtype)

        counts = xp.zeros(T, dtype=H.real.dtype)

        for k in range(K):
            Hout[k:k + L] += Hhf[:, k]

            counts[k:k + L] += 1

        Hout /= counts[:, None, None]

        return Hout
    
    def svd_filter_batched(self, U_subaps, svd_threshold):
        """Batched SVD filter for subapertures"""
        xp = self.bm.xp
        
        if svd_threshold < 0:
            return U_subaps
        
        ny_s, nx_s, sub_ny, sub_nx, nz = U_subaps.shape
        B = ny_s * nx_s
        
        H2 = U_subaps.reshape(B, sub_ny * sub_nx, nz)
        eps = 1e-12
        
        cov = xp.einsum('bpi,bpj->bij', H2.conj(), H2) + eps * xp.eye(nz, dtype=H2.dtype)
        
        V_list = []
        for b in range(B):
            _, Vb = xp.linalg.eigh(cov[b])
            V_list.append(Vb[:, ::-1])
        
        Vt = xp.stack([Vb[:, :svd_threshold] for Vb in V_list], axis=0)
        H2_Vt = xp.einsum('bpi,bik->bpk', H2, Vt)
        proj = xp.einsum('bpk,bjk->bpj', H2_Vt, Vt.conj())
        
        return (H2 - proj).reshape(ny_s, nx_s, sub_ny, sub_nx, nz)
    
    def frequency_symmetric_filtering(self, batch_size, sampling_freq, low_freq, high_freq=None):
        """Create symmetric frequency filter mask"""
        xp = self.bm.xp
        fft = self.bm.fft
        
        freqs = fft.fftfreq(batch_size, 1 / sampling_freq)
        
        if high_freq is None:
            idxs = xp.abs(freqs) > low_freq
        else:
            idxs = (high_freq > xp.abs(freqs)) & (xp.abs(freqs) > low_freq)
        
        return idxs, freqs[idxs]
    
    def fourier_time_transform(self, H):
        """FFT along time axis"""
        return self.bm.fft.fft(H, axis=0, norm="ortho")
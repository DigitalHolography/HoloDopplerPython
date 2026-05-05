from cupy.cuda.nvtx import RangePush, RangePop
class SVD_filter():
    

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
    
    # @staticmethod
    # def randomized_svd(H2, k, xp):
    #     n_random = k + 5

    #     # random projection
    #     Omega = xp.random.randn(H2.shape[1], n_random, dtype=H2.dtype)
    #     Y = H2 @ Omega

    #     # orthonormalize
    #     Q, _ = xp.linalg.qr(Y)

    #     # smaller SVD
    #     B = Q.conj().T @ H2
    #     Ub, S, Vh = xp.linalg.svd(B, full_matrices=False)

    #     U = Q @ Ub
    #     return U, S, Vh
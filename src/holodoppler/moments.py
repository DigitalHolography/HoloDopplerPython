"""
Moment calculations from power spectral density
"""


class MomentCalculator:
    """Calculate spectral moments"""
    
    def __init__(self, backend_manager):
        self.bm = backend_manager
    
    def moment(self, A, freqs, n):
        """Calculate nth spectral moment"""
        xp = self.bm.xp
        return xp.sum(
            A * (freqs[..., xp.newaxis, xp.newaxis] ** n),
            axis=0
        ).astype(xp.float32)
    
    def moment_khz(self, A, freqs, n):
        """Calculate nth spectral moment with frequency in kHz"""
        xp = self.bm.xp
        return xp.sum(
            A * ((freqs[..., xp.newaxis, xp.newaxis] / 1000) ** n),
            axis=0
        ).astype(xp.float32)
"""
Backend management for numpy/cupy switching
"""

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    import cupyx.scipy.ndimage as cp_ndi
    _cupy_available = True
except ImportError:
    cp = None
    cp_fft = None
    cp_gaussian_filter = None
    cp_ndi = None
    _cupy_available = False

import scipy.fft as np_fft
from scipy.ndimage import gaussian_filter as np_gaussian_filter
import scipy.ndimage as np_ndi


class BackendManager:
    """Manages numpy/cupy backend switching"""
    
    def __init__(self, backend="numpy"):
        self.backend_name = backend
        self.xp = None
        self.fft = None
        self.gaussian_filter = None
        self.ndi = None
        self._init_backend()
    
    def _init_backend(self):
        if "cupy" in self.backend_name:
            if not _cupy_available:
                raise RuntimeError("CuPy backend requested but CuPy is not available.")
            self.xp = cp
            self.fft = cp_fft
            self.gaussian_filter = cp_gaussian_filter
            self.ndi = cp_ndi
        else:
            self.xp = np
            self.fft = np_fft
            self.gaussian_filter = np_gaussian_filter
            self.ndi = np_ndi
    
    def to_backend(self, arr):
        if "cupy" in self.backend_name and self.xp is cp:
            return cp.asarray(arr)
        return arr
    
    def to_numpy(self, arr):
        if "cupy" in self.backend_name and isinstance(arr, cp.ndarray):
            return arr.get()
        return arr
    
    @property
    def is_gpu(self):
        return "cupy" in self.backend_name and _cupy_available
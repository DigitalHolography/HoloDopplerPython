import numpy as np
import scipy.fft as np_fft
from scipy.ndimage import gaussian_filter as np_gaussian_filter
import scipy.ndimage as np_ndi
import scipy.fftpack as np_fftpack

try:
    import cupy as cp
    import cupyx.scipy.fft as cp_fft
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    import cupyx.scipy.ndimage as cp_ndi
    _cupy_available = True
except Exception:
    cp = None
    cp_fft = None
    _cupy_available = False

from .utils.reading import FileReader
from .utils.registration import RegistrationUtils
from .utils.propagation import Propagator
from .utils.moments import MomentsCalculator
from .utils.kernels_calculation import KernelsCalculator
from .utils.plotting import PlottingUtils
from .utils.svd_filtering import SVD_filter
from .utils.shack_hartmann import ShackHartmannUtils
from .utils.render import Renderer
from .utils.process import Processor




class Holodoppler(FileReader, RegistrationUtils, Propagator, MomentsCalculator, KernelsCalculator, PlottingUtils, SVD_filter, ShackHartmannUtils, Renderer, Processor):
    """
    Holodoppler processing class for .holo files. 
        adding support for .cine files.

    Backend:
        backend="numpy"  -> CPU
        backend="cupy"   -> GPU (if available)

    Data is always read on CPU then transferred to GPU if needed.
    """

    HOLO_HEADER_SIZE = 64

    def __init__(self, backend = "numpy", pipeline_version = "latest"):

        self.file_path = ""

        self.backend = backend

        self.pipeline_version = pipeline_version

        self._init_backend()

        self._init_pipeline()

        self.fid = None

        self.parameters = dict()

        self.kernels = dict()

    # ------------------------------------------------------------
    # Backend handling
    # ------------------------------------------------------------

    def _init_backend(self):

        if "cupy" in self.backend:
            if not _cupy_available:
                raise RuntimeError("CuPy backend requested but CuPy is not available.")
            self.xp = cp
            self.fft = cp_fft
            self.gaussian_filter = cp_gaussian_filter
            self.ndi = cp_ndi
            self.fftpack = cp_fft
        else:
            self.backend = self.backend
            self.xp = np
            self.fft = np_fft
            self.gaussian_filter = np_gaussian_filter
            self.ndi = np_ndi
            self.fftpack = np_fftpack
            
    def _to_backend(self, arr):
        if "cupy" in self.backend:
            return cp.asarray(arr)
        return arr

    def _to_numpy(self, arr):
        if "cupy" in self.backend:
            if isinstance(arr, cp.ndarray):
                return arr.get() 
            return arr  # already NumPy or other type
        return arr

    def _init_pipeline(self):
        if self.pipeline_version == "latest":
            self._frequency_symmetric_filtering = self._new_frequency_symmetric_filtering
            self._resize = self.resize_fft2_slicewise
            self._registration = self._registration_trs
            self._applyregistration = self.applyregistration
            return
        elif self.pipeline_version == "old":
            self._frequency_symmetric_filtering = self._old_frequency_symmetric_filtering
            self._resize = self.resize_matlab_slicewise
            self._registration = self.old_registration
            self._applyregistration = self._roll2d
            self._moment = self._momentkHz

    
        
    

    
        
        

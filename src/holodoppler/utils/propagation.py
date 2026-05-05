class Propagator():
    # ------------------------------------------------------------
    # Calculation processing
    # ------------------------------------------------------------
    
    def pad_array_centrally(self, arr, new_shape):
        
        xp = self.xp
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ny, nx = arr.shape[-2:]
        new_ny, new_nx = new_shape

        if new_ny < ny or new_nx < nx:
            raise ValueError("new_shape must be greater than or equal to the current last two dimensions")

        pad_y0 = (new_ny - ny) // 2
        pad_y1 = new_ny - ny - pad_y0
        pad_x0 = (new_nx - nx) // 2
        pad_x1 = new_nx - nx - pad_x0

        pad_width = [(0, 0)] * arr.ndim
        pad_width[-2] = (pad_y0, pad_y1)
        pad_width[-1] = (pad_x0, pad_x1)

        return xp.pad(arr, pad_width, mode="constant")
    
    def crop_array_centrally(self, arr, target_shape):
        xp = self.xp

        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape)

        ny, nx = arr.shape[-2:]
        tgt_ny, tgt_nx = target_shape

        if tgt_ny > ny or tgt_nx > nx:
            raise ValueError("target_shape must be smaller than or equal to current shape")

        crop_y0 = (ny - tgt_ny) // 2
        crop_y1 = crop_y0 + tgt_ny
        crop_x0 = (nx - tgt_nx) // 2
        crop_x1 = crop_x0 + tgt_nx

        slices = [slice(None)] * arr.ndim
        slices[-2] = slice(crop_y0, crop_y1)
        slices[-1] = slice(crop_x0, crop_x1)

        return arr[tuple(slices)]

    def _fresnel_transform(self, frames, zero_padding = False):
        
        if zero_padding:
            frames = self.pad_array_centrally(frames, zero_padding)

        if self.pipeline_version == "latest" : 
            return self.fft.fftshift(
                self.fft.fft2(frames *self.kernels["Fresnel_in"], axes=(-1, -2), norm="ortho"), axes=(-1, -2)
            ) *self.kernels["Fresnel_out"]
        else :
            return self.fft.fftshift(
                self.fft.fft2(frames *self.kernels["Fresnel_in"], axes=(-1, -2), norm="ortho"), axes=(-1, -2)
            )
    
    def _angular_spectrum_transform(self, frames, zero_padding = False):
        
        if zero_padding:
            frames = self.pad_array_centrally(frames, zero_padding)

        tmp = self.fft.fft2(frames,axes=(-1, -2), norm="ortho") * self.fft.fftshift(self.kernels["AngularSpectrum"],axes=(-1, -2))

        return self.fft.ifft2(tmp,axes=(-1, -2), norm="ortho")

    def _fourier_time_transform(self, H):

        return self.fft.fft(H, axis=0, norm="ortho") 
import numpy as np
class MomentsCalculator():
    
    # ------------------------------------------------------------
    # Frequency axis and masks
    # ------------------------------------------------------------

    def _new_frequency_symmetric_filtering(self, batch_size, sampling_freq, low_freq, high_freq = None):

        xp = self.xp

        # fftfreq is backend dependent
        freqs = self.fft.fftfreq(batch_size, 1 / sampling_freq)

        if high_freq is None:
            idxs = xp.abs(freqs) > low_freq
        else:
            idxs = (high_freq > xp.abs(freqs)) & (xp.abs(freqs) > low_freq)

        return idxs, freqs[idxs]
    
    def _old_frequency_symmetric_filtering(self, batch_size, sampling_freq, low_freq, high_freq = None):
        # old and not clean but similar to matlab version

        if high_freq is None:
            high_freq = sampling_freq / 2

        # convert frequencies to indices
        n1 = int(np.ceil(low_freq * batch_size / sampling_freq))
        n2 = int(np.ceil(high_freq * batch_size / sampling_freq))

        # clamp to valid range (MATLAB: 1..size(SH,3))
        n1 = max(min(n1, batch_size), 1)
        n2 = max(min(n2, batch_size), 1)

        # symmetric integration interval
        n3 = batch_size - n2 + 1
        n4 = batch_size - n1 + 1

        # convert to Python indexing (0-based)
        i1 = n1 - 1
        i2 = n2      # exclusive
        i3 = n3 - 1
        i4 = n4      # exclusive

        # frequency ranges (MATLAB inclusive -> +1 in Python)
        f_range = np.arange(n1, n2 + 1) * (sampling_freq / batch_size)
        f_range_sym = np.arange(-n2, -n1 + 1) * (sampling_freq / batch_size)

        freqs = np.concatenate([f_range, f_range_sym], axis=0)

        # boolean mask
        idxs = np.zeros(batch_size, dtype=bool)
        idxs[i1:i2] = True
        idxs[i3:i4] = True

        return self._to_backend(idxs), self._to_backend(freqs)

    # ------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------

    def _moment(self, A, freqs, n):

        xp = self.xp

        return xp.sum(
            A * (freqs[... ,xp.newaxis, xp.newaxis] ** n),
            axis=0
        ).astype(xp.float32)
        
    def _momentkHz(self, A, freqs, n):

        xp = self.xp

        return xp.sum(
            A * ((freqs[... ,xp.newaxis, xp.newaxis] / 1000) ** n),
            axis=0
        ).astype(xp.float32)
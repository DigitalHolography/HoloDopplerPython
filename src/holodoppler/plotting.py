"""
Debug plotting utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import cupy as cp
except ImportError:
    cp = None


class ImagePlotter:
    """Simple image plotter"""
    
    def __init__(self):
        pass
    
    def plot(self, image):
        if isinstance(image, cp.ndarray):
            image = image.get()
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-12) * 255
        return image.astype(np.uint8)
    
    def close(self):
        pass


class PhasePlotter:
    """Phase image plotter"""
    
    def __init__(self, relative=False):
        self.relative = relative
    
    def plot(self, phase):
        if isinstance(phase, cp.ndarray):
            phase = cp.asnumpy(phase)
        if self.relative:
            ny, nx = phase.shape
            phase = phase - phase[ny//2, nx//2]
        phase = (phase + np.pi) % (2*np.pi) - np.pi
        norm = (phase + np.pi) / (2*np.pi)
        img = (norm * 255).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)
    
    def close(self):
        pass


class ShiftsPlotter:
    """Vector field plotter for subaperture shifts"""
    
    def __init__(self, title="Wavefront Slopes", figsize=(8, 6), dpi=100, scale=None):
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.title = title
        self.scale = scale
    
    def plot(self, shifts_y, shifts_x):
        if isinstance(shifts_y, cp.ndarray):
            shifts_y = cp.asnumpy(shifts_y)
        if isinstance(shifts_x, cp.ndarray):
            shifts_x = cp.asnumpy(shifts_x)
        
        if self.scale is None:
            mag = np.sqrt(shifts_x**2 + shifts_y**2)
            med = np.median(mag[mag > 0]) if np.any(mag > 0) else 1.0
            scale = 1.0 / (med + 1e-12)
        else:
            scale = self.scale
        
        self.ax.clear()
        ny_subabs, nx_subabs = shifts_y.shape
        X, Y = np.meshgrid(np.arange(nx_subabs), np.arange(ny_subabs))
        self.ax.quiver(X, Y, shifts_x, shifts_y, scale=scale)
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Sub-aperture X Index')
        self.ax.set_ylabel('Sub-aperture Y Index')
        self.ax.set_xlim(-0.5, nx_subabs - 0.5)
        self.ax.set_ylim(-0.5, ny_subabs - 0.5)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_aspect('equal')
        self.canvas.draw()
        
        img = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(self.canvas.get_width_height()[::-1] + (4,))
        return img[..., :3]
    
    def close(self):
        plt.close(self.fig)


class SpectrumPlotter:
    """Spectrum plotter with frequency bands"""
    
    def __init__(self, fs, f1, f2, title="Spectrum",
                 figsize=(8, 6), dpi=100, show_bands=True,
                 ylim=None, use_stem=False):
        self.fs = fs
        self.f1 = f1
        self.f2 = f2
        self.show_bands = show_bands
        self.ylim = ylim
        self.use_stem = use_stem
        
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.title = title
    
    def plot(self, spectrum_line, freqs):
        if isinstance(spectrum_line, cp.ndarray):
            spectrum_line = cp.asnumpy(spectrum_line)
        
        freqs_full = np.fft.fftfreq(len(spectrum_line), d=1/self.fs)
        freqs_full = np.fft.fftshift(freqs_full)
        spectrum_line = np.fft.fftshift(spectrum_line)
        
        spectrum_line[spectrum_line <= 0] = np.nan
        signal_log = np.log10(spectrum_line)
        
        self.ax.clear()
        
        if self.use_stem:
            markerline, stemlines, baseline = self.ax.stem(freqs_full, signal_log, basefmt=" ")
            plt.setp(markerline, color='black')
            plt.setp(stemlines, color='black', linewidth=1)
        else:
            self.ax.plot(freqs_full, signal_log, color='black', linewidth=1)
        
        if self.show_bands:
            r1 = (-self.f2 < freqs_full) & (freqs_full < -self.f1)
            r2 = (self.f1 < freqs_full) & (freqs_full < self.f2)
            self.ax.fill_between(freqs_full[r1], signal_log[r1],
                                  color='lightgray', edgecolor='black')
            self.ax.fill_between(freqs_full[r2], signal_log[r2],
                                  color='lightgray', edgecolor='black')
        
        for val in [self.f1, self.f2, -self.f1, -self.f2]:
            self.ax.axvline(val, linestyle='--', color='black')
        
        self.ax.set_xlim([freqs_full.min(), freqs_full.max()])
        
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        else:
            valid = (~np.isnan(spectrum_line)) & (np.abs(freqs_full) > self.f1)
            if np.any(valid):
                ymin = 0.9 * np.log10(np.nanmin(spectrum_line[valid]))
                ymax = 1.11 * np.log10(np.nanmax(spectrum_line[valid]))
                self.ax.set_ylim([ymin, ymax])
        
        if self.f1 != 0:
            ticks = [-self.f2, -self.f1, self.f1, self.f2]
        else:
            ticks = [-self.f2, self.f2]
        
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([f"{t:.1f}" for t in ticks])
        
        self.ax.set_title(self.title)
        self.ax.set_xlabel('frequency (Hz)')
        self.ax.set_ylabel('log10 S')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.canvas.draw()
        img = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(self.canvas.get_width_height()[::-1] + (4,))
        return img[..., :3]
    
    def close(self):
        plt.close(self.fig)


class SubapertureMontagePlotter:
    """Montage of subaperture images"""
    
    def __init__(self):
        pass
    
    def plot(self, U_subaps):
        if isinstance(U_subaps, cp.ndarray):
            U_subaps = cp.asnumpy(U_subaps)
        
        rows = []
        for iy in range(U_subaps.shape[0]):
            row_imgs = [U_subaps[iy, ix] for ix in range(U_subaps.shape[1])]
            rows.append(np.hstack(row_imgs))
        return np.vstack(rows)
    
    def close(self):
        pass


class DebugPlotterManager:
    """Manages debug plotters"""
    
    def __init__(self, parameters):
        plotters = {
            "montage": SubapertureMontagePlotter(),
            "shifts": ShiftsPlotter(scale=30),
            "shifts_rel": ShiftsPlotter(scale=None),
            "phase": PhasePlotter(),
            "phase_rel": PhasePlotter(relative=True),
            "M0notfixed": ImagePlotter(),
            "spectrum": SpectrumPlotter(
                fs=parameters["sampling_freq"],
                f1=parameters["low_freq"],
                f2=parameters["high_freq"],
                ylim=(12, 20),
                use_stem=False
            ),
        }
        
        self.sources = {
            "montage": lambda res: (res["U_subaps"],),
            "shifts": lambda res: (res["shifts_y"], res["shifts_x"]),
            "shifts_rel": lambda res: (res["shifts_y"], res["shifts_x"]),
            "phase": lambda res: (res["phase"],),
            "phase_rel": lambda res: (res["phase"],),
            "M0notfixed": lambda res: (res["M0notfixed"],),
            "spectrum": lambda res: (res["spectrum_line"], res["freqs"]),
        }
        
        self.plotters = plotters
        self.active = parameters.get("debug", False)
    
    def plot_all(self, res):
        """Plot all available debug outputs"""
        if not self.active:
            return {}
        
        out = {}
        for key, plotter in self.plotters.items():
            if key in self.sources:
                try:
                    args = self.sources[key](res)
                    out[key] = plotter.plot(*args)
                except KeyError:
                    pass
        return out
    
    def close_all(self):
        for plotter in self.plotters.values():
            plotter.close()
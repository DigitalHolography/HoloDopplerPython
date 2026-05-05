import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import cupy as cp

class PlottingUtils():
    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Render tools for debug and visualization
    # ------------------------------------------------------------

    class ImagePlotter:
        def __init__(self):
            pass
        def plot(self, image):
            image_ = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            if isinstance(image_, cp.ndarray):
                image_ = image_.get()
            return image_.astype(np.uint8)
        def close(self):
            pass

    class PhasePlotter:
        def __init__(self, relative=False):
            self.relative = relative  
        def plot(self, phase):
            if isinstance(phase, cp.ndarray):
                phase = cp.asnumpy(phase)
            if self.relative :
                ny, nx = phase.shape
                phase = phase - phase[ny//2,nx//2]
            phase = (phase + np.pi) % (2*np.pi) - np.pi
            norm = (phase + np.pi) / (2*np.pi)
            img = (norm * 255).astype(np.uint8)
            img = np.stack([img, img, img], axis=-1)
            # No resizing → output keeps exact input shape (rectangular preserved)
            return img
        def close(self):
            pass

    class ShiftsPlotter:
        # TODO no matplotlib just cv2
        def __init__(self, title="Wavefront Slopes from Sub-aperture Shifts", figsize=(8, 6), dpi=100, scale=None):
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            self.canvas = FigureCanvasAgg(self.fig)
            self.title = title
            self.scale = scale

        def plot(self, shifts_y, shifts_x):
            if isinstance(shifts_y, cp.ndarray):
                shifts_y = cp.asnumpy(shifts_y)
            if isinstance(shifts_x, cp.ndarray):
                shifts_x = cp.asnumpy(shifts_x)
            # --- AUTO SCALE IF NONE ---
            if self.scale is None:
                mag = np.sqrt(shifts_x**2 + shifts_y**2)
                med = np.median(mag[mag > 0]) if np.any(mag > 0) else 1.0
                scale = 1.0 / (med + 1e-12)
            else:
                scale = self.scale
            self.ax.clear()
            ny_subabs, nx_subabs = shifts_y.shape
            X, Y = np.meshgrid(np.arange(nx_subabs), np.arange(ny_subabs))
            self.ax.quiver(X, Y, shifts_x, shifts_y, scale=self.scale)
            self.ax.set_title(self.title)
            self.ax.set_xlabel('Sub-aperture X Index')
            self.ax.set_ylabel('Sub-aperture Y Index')
            self.ax.set_xlim(-0.5, nx_subabs - 0.5)
            self.ax.set_ylim(-0.5, ny_subabs - 0.5)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_aspect('equal')
            self.canvas.draw()
            img = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(self.canvas.get_width_height()[::-1] + (4,))
            img = img[..., :3]  # drop alpha channel
            return img
        def close(self):
            plt.close(self.fig)
        
    class SpectrumPlotter:

        def __init__(self, fs, f1, f2,
                    title="Spectrum",
                    figsize=(8, 6), dpi=100,
                    show_bands=True,
                    ylim=None,
                    use_stem=False):

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
            freqs = np.fft.fftfreq(len(spectrum_line), d=1/self.fs)

            freqs = np.fft.fftshift(freqs)
            spectrum_line = np.fft.fftshift(spectrum_line)

            # log scale
            spectrum_line[spectrum_line <= 0] = np.nan
            signal_log = np.log10(spectrum_line)

            self.ax.clear()

            # main plot
            if self.use_stem:
                markerline, stemlines, baseline = self.ax.stem(freqs, signal_log, basefmt=" ")
                plt.setp(markerline, color='black')
                plt.setp(stemlines, color='black', linewidth=1)
            else:
                self.ax.plot(freqs, signal_log, color='black', linewidth=1)

            # shaded bands
            if self.show_bands:
                r1 = (-self.f2 < freqs) & (freqs < -self.f1)
                r2 = (self.f1 < freqs) & (freqs < self.f2)

                self.ax.fill_between(freqs[r1], signal_log[r1],
                                    color='lightgray', edgecolor='black')
                self.ax.fill_between(freqs[r2], signal_log[r2],
                                    color='lightgray', edgecolor='black')

            # vertical lines
            for val in [self.f1, self.f2, -self.f1, -self.f2]:
                self.ax.axvline(val, linestyle='--', color='black')

            # x limits
            self.ax.set_xlim([freqs.min(), freqs.max()])

            # y limits
            if self.ylim is not None:
                self.ax.set_ylim(self.ylim)
            else:
                valid = (~np.isnan(spectrum_line)) & (np.abs(freqs) > self.f1)
                if np.any(valid):
                    ymin = 0.9 * np.log10(np.nanmin(spectrum_line[valid]))
                    ymax = 1.11 * np.log10(np.nanmax(spectrum_line[valid]))
                    self.ax.set_ylim([ymin, ymax])

            # ticks
            if self.f1 != 0:
                ticks = [-self.f2, -self.f1, self.f1, self.f2]
            else:
                ticks = [-self.f2, self.f2]

            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels([f"{t:.1f}" for t in ticks])

            # labels
            self.ax.set_title(self.title)
            self.ax.set_xlabel('frequency (Hz)')
            self.ax.set_ylabel('log10 S')

            self.ax.grid(True, linestyle='--', alpha=0.5)

            # render
            self.canvas.draw()
            img = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(self.canvas.get_width_height()[::-1] + (4,))

            return img[..., :3]

        def close(self):
            plt.close(self.fig)
        
    class SubapertureMontagePlotter:
        def __init__(self):
            pass
        def plot(self, U_subaps):
            if isinstance(U_subaps, cp.ndarray):
                U_subaps = cp.asnumpy(U_subaps)
            rows = []
            for iy in range(U_subaps.shape[0]):
                row_imgs = [U_subaps[iy, ix] for ix in range(U_subaps.shape[1])]
                rows.append(np.hstack(row_imgs))
            montage_img = np.vstack(rows)
            return montage_img
        def close(self):
            pass
        
    def init_plot_debug(self, parameters):
        matplotlib.use("Agg")
        # --- CENTRAL DEBUG REGISTRY (edit only here to add new outputs) ---
        self.debug_plotters = {
            "montage": self.SubapertureMontagePlotter(),
            "shifts": self.ShiftsPlotter(scale=30),
            "shifts_rel": self.ShiftsPlotter(scale=None),
            "phase": self.PhasePlotter(),
            "phase_rel": self.PhasePlotter(relative=True),
            "M0notfixed": self.ImagePlotter(),
            "spectrum": self.SpectrumPlotter(
                fs=parameters["sampling_freq"],
                f1=parameters["low_freq"],
                f2=parameters["high_freq"],
                ylim=(12,20),
                use_stem=False
            ),
        }

        # maps key -> function(res) -> args for plotter
        self.debug_sources = {
            "montage": lambda res: (res["U_subaps"],),
            "shifts":  lambda res: (res["shifts_y"], res["shifts_x"]),
            "shifts_rel":  lambda res: (res["shifts_y"], res["shifts_x"]),
            "phase":   lambda res: (res["phase"],),
            "phase_rel": lambda res: (res["phase"],),
            "M0notfixed": lambda res: (res["M0notfixed"],),
            "spectrum": lambda res: (res["spectrum_line"], res["freqs"]),
        }
    
    def plot_debug(self, res, i):
        out = {}

        for key, plotter in self.debug_plotters.items():

            try:
                args = self.debug_sources[key](res)
            except KeyError:
                continue  # required data not present

            out[key] = plotter.plot(*args)

        return out
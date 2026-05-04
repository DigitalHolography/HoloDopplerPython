# plot_zernike.py
from pathlib import Path
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt


def first_h5_in_raw(folder: Path) -> Path:
    raw = folder / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"No 'raw' folder found at: {raw}")

    files = sorted(list(raw.glob("*.h5")) + list(raw.glob("*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 file found in: {raw}")

    return files[0]


def window_array(name: str, n: int) -> np.ndarray:
    name = name.lower()

    if name in ("none", "rect", "rectangular", "boxcar"):
        return np.ones(n)
    if name in ("hann", "hanning"):
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)

    raise ValueError(f"Unsupported window type: {name}")


def lowpass_fft(signal, fs, cutoff_hz):
    if cutoff_hz is None:
        return signal.copy()

    n = signal.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal, axis=0)

    spectrum[freqs > cutoff_hz, :] = 0.0

    return np.fft.irfft(spectrum, n=n, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--fs", type=float, required=True, help="Sampling frequency in Hz")
    parser.add_argument("--window", default="hann", help="none, hann, hamming, blackman")
    parser.add_argument("--window-size", type=int, default=None, help="FFT window size in samples")
    parser.add_argument("--lowpass", type=float, default=None, help="Optional low-pass cutoff in Hz")
    args = parser.parse_args()

    h5_path = first_h5_in_raw(args.folder)
    print(f"Reading: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "zernike_coefs_radians" not in f:
            raise KeyError("Dataset 'zernike_coefs_radians' not found")

        data = np.asarray(f["zernike_coefs_radians"])

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "Expected dataset shape (time, >=3), with first three columns being "
            "defocus, astig1, astig2"
        )

    signals = data[:, :3]
    names = ["Defocus", "Astig 1", "Astig 2"]
    
    # -------------------------------------------------------------------------
    # Save signals to CSV (debug)
    # -------------------------------------------------------------------------
    import csv

    debug_dir = Path("./debug_outputs")
    debug_dir.mkdir(parents=True, exist_ok=True)

    csv_path = debug_dir / "zernike_signals.csv"

    t = np.arange(signals.shape[0]) / args.fs  # reuse if already defined

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "defocus_rad", "astig1_rad", "astig2_rad"])
        for i in range(signals.shape[0]):
            writer.writerow([t[i], signals[i, 0], signals[i, 1], signals[i, 2]])

    print(f"Saved CSV to: {csv_path}")

    n_total = signals.shape[0]
    t = np.arange(n_total) / args.fs

    # -------------------------------------------------------------------------
    # Time signals
    # -------------------------------------------------------------------------

    plt.figure()
    for i, name in enumerate(names):
        plt.plot(t, signals[:, i], label=name)

    plt.title("Zernike coefficients")
    plt.xlabel("Time [s]")
    plt.ylabel("Coefficient [rad]")
    plt.legend()
    plt.grid(True)

    # -------------------------------------------------------------------------
    # FFT
    # -------------------------------------------------------------------------

    n_fft = args.window_size or n_total
    n_fft = min(n_fft, n_total)

    fft_signals = signals[:n_fft]
    w = window_array(args.window, n_fft)[:, None]

    spectrum = np.fft.rfft(fft_signals * w, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / args.fs)

    amp = np.abs(spectrum) / np.sum(w[:, 0])
    amp[1:-1, :] *= 2.0

    plt.figure()
    for i, name in enumerate(names):
        plt.plot(freqs, amp[:, i], label=name)

    plt.title(f"FFT - window={args.window}, size={n_fft}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [rad]")
    plt.legend()
    plt.grid(True)

    # -------------------------------------------------------------------------
    # Filtered time signals
    # -------------------------------------------------------------------------

    if args.lowpass is not None:
        filtered = lowpass_fft(signals, args.fs, args.lowpass)

        plt.figure()
        for i, name in enumerate(names):
            plt.plot(t, filtered[:, i], label=name)

        plt.title(f"Low-pass filtered signals - cutoff={args.lowpass} Hz")
        plt.xlabel("Time [s]")
        plt.ylabel("Coefficient [rad]")
        plt.legend()
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
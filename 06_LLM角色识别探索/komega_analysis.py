"""Space-time (k-omega) analysis of the five PDE datasets.

The single-time-slice spectrum used in the role probe cannot tell a property of the
*state* from a property of the *dynamics*. This script does the honest version:

  1. 2D FFT over (x, t) with a Hann window on both axes, so a peak at (k0, w0)
     means a sustained spatio-temporal structure at that wavenumber and frequency.
  2. For each dominant wavenumber, the omega profile gives the centroid and the
     spread: |w| >> spread  ->  a travelling wave (phase speed c = w/k);
     centroid at w ~ 0 with a broad profile  ->  a non-oscillatory transient.
  3. The modal amplitude a_k(t) = |sum_x u(x,t) e^{-i k x}| gives the per-mode
     growth/decay rate. Fitting rate(k) against k recovers the power law, which is
     the data-side way to decide *how many spatial derivatives the operator needs*:
     rate ~ -D k^2 is diffusion (2nd order), a positive maximum at k0 != 0 needs a
     higher-order term (4th order and up).

Nothing here names a physical quantity; it only reads operators off the data.
"""
import os

import numpy as np
import scipy.io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["Topography_Chemotaxis", "Morphogenesis", "Forced_Swift_Hohenberg",
            "Traffic_Flow_Bottleneck", "Predator_Prey"]


def peaks(power: np.ndarray, k_axis: np.ndarray, w_axis: np.ndarray,
          length: float = 1.0, count: int = 4) -> list[tuple]:
    """Distinct (k, omega) peaks, ignoring the mean mode and neighbouring bins."""
    work = power.copy()
    work[np.abs(k_axis) < 1e-9, :] = 0.0        # the mean mode says nothing
    total = max(work.sum(), 1e-30)
    found = []
    for _ in range(count):
        flat = int(np.argmax(work))
        i, j = np.unravel_index(flat, work.shape)
        k_val, w_val = k_axis[i], w_axis[j]
        if abs(k_val) * length / (2 * np.pi) < 1.0:      # at least mode number 1
            break
        share = work[i, j] / total
        found.append((k_val, w_val, share))
        work[max(0, i - 1):i + 2, :] = 0.0       # suppress the +-k / neighbour pair
        work[:, max(0, j - 1):j + 2] = 0.0
    return found


def modal_rates(u: np.ndarray, x: np.ndarray, t: np.ndarray, k_max: int = 8):
    """Growth rate of each spatial mode, from a straight-line fit of log|a_k(t)|."""
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    rates = {}
    reference = None
    for k_index in range(1, k_max + 1):
        a_k = np.abs(np.sum(u * np.exp(-2j * np.pi * k_index * x / (x[-1] - x[0]))[:, None],
                            axis=0)) * dx
        if k_index == 1:
            reference = float(a_k.max())
        peak = float(a_k.max())
        # a mode sitting at the numerical floor has no meaningful slope: its
        # "growth" is rounding noise. Keep only modes well above the floor.
        if reference is None or peak < 1e-4 * reference:
            rates[k_index] = (float("nan"), peak, "below floor")
            continue
        # linear phase: the first third of the record, before saturation
        stop = max(8, a_k.size // 3)
        window = slice(0, stop)
        slope = np.polyfit(t[window], np.log(np.maximum(a_k[window], 1e-300)), 1)[0]
        rates[k_index] = (slope, peak, "ok")
    return rates


for name in DATASETS:
    mat = scipy.io.loadmat(os.path.join(ROOT, "01_原版复现_LLM-PDESR", "data", name, "data.mat"))
    x = mat["x"].flatten()
    t = mat["t"].flatten()
    u = mat["u"]
    nx, nt = u.shape
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    length = x[-1] - x[0]

    window = np.hanning(nx)[:, None] * np.hanning(nt)[None, :]
    field = (u - u.mean()) * window
    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    k_axis = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi   # rad / length
    w_axis = np.fft.fftshift(np.fft.fftfreq(nt, d=dt)) * 2 * np.pi

    print("=" * 78)
    print(f"### {name}   nx={nx} nt={nt}   domain [{x[0]:.2f},{x[-1]:.2f}]  "
          f"T={t[-1]:.2f}")
    print("  dominant (k, omega) peaks with a Hann window on both axes:")
    for k_val, w_val, share in peaks(power, k_axis, w_axis, length=length):
        speed = (w_val / k_val) if abs(k_val) > 1e-9 else float("nan")
        mode = k_val * length / (2 * np.pi)
        print(f"    k={k_val:6.2f} (mode n={mode:4.1f})  omega={w_val:+8.3f}  "
              f"share={share * 100:5.1f}%   -> phase speed c={speed:+.3f}")

    # omega profile for the strongest |k|: wave or relaxation?
    k_pos = np.abs(k_axis)
    order = np.argsort([power[np.abs(k_axis) == kk, :].sum() for kk in k_pos])[::-1]
    seen: list[float] = []
    for kk in k_pos[order]:
        if kk < 1e-9 or any(abs(kk - s) < 1e-9 for s in seen):
            continue
        seen.append(float(kk))
        mask = np.abs(k_axis - kk) < 1e-9        # positive k only: a +-k average
        if mask.sum() == 0:
            mask = np.abs(k_axis + kk) < 1e-9    # would cancel a travelling wave
        profile = power[mask, :].sum(axis=0)
        total = profile.sum()
        centroid = float((profile * w_axis).sum() / total)
        spread = float(np.sqrt((profile * (w_axis - centroid) ** 2).sum() / total))
        kind = "travelling wave" if abs(centroid) > 2 * spread else "non-oscillatory"
        print(f"  omega profile at |k|={kk:.2f}: centroid {centroid:+.3f}, "
              f"spread {spread:.3f} -> {kind}")
        if len(seen) >= 3:
            break

    rates = modal_rates(u, x, t)
    if rates:
        ks = np.array(sorted(rates))
        slopes = np.array([rates[k][0] for k in ks])
        print("  per-mode rate (log|a_k| slope) and amplitude:")
        for k in ks:
            slope, peak, note = rates[k]
            print(f"    mode {k}: rate {slope:+.3f}   peak amplitude {peak:.3e}   {note}")
        usable = np.isfinite(slopes)
        if usable.sum():
            best = ks[usable][int(np.nanargmax(slopes[usable]))]
            top = float(np.nanmax(slopes[usable]))
            print(f"    fastest mode: k={best} (rate {top:+.3f}); "
                  f"sign: {'growth' if top > 0 else 'decay'}")
        positive = (ks <= 6) & np.isfinite(slopes) & (slopes < 0)
        if positive.sum() >= 3:
            exponent = np.polyfit(np.log(ks[positive]), np.log(-slopes[positive]), 1)[0]
            print(f"    decay rate vs k power law: rate ~ -k^{exponent:.2f} "
                  f"(2 = diffusion / second order, 4 = hyper-diffusion)")

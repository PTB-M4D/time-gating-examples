import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import pandas
from scipy.ndimage import convolve1d
from scipy import signal, special
from scipy.linalg import block_diag

from PyDynamic.uncertainty.propagate_DFT import (
    GUM_iDFT,
    GUM_DFT,
    DFT2AmpPhase,
    AmpPhase2DFT,
)
from PyDynamic.misc import complex_2_real_imag as c2ri
from PyDynamic.misc import real_imag_2_complex as ri2c
from PyDynamic.misc.tools import shift_uncertainty


def load_data(name="", return_mag_phase=False, return_full_cov=True):
    rel_path = "../data/"

    if name == "diag_only":
        # load reflection data
        file_reflection = "Beatty Line s11 MagPhase data.xlsx"
        df = pandas.read_excel(rel_path + file_reflection, skiprows=2)

        # add missing 0Hz-frequency point and construct complex variables
        # phase for a 0Hz is assumed to be zero
        f = np.r_[0, df.iloc[:, 0]]  # GHz
        s_param_mag = np.r_[df.iloc[0, 1], df.iloc[:, 1]]
        s_param_mag_unc = np.r_[df.iloc[0, 2], df.iloc[:, 2]]
        s_param_phase = np.r_[0, df.iloc[:, 3]] / 180 * np.pi
        s_param_phase_unc = np.r_[0, df.iloc[:, 4]] / 180 * np.pi

        # translate into PyDynamic-internal Re/Im-representation
        s_param_UAP = np.square(np.r_[s_param_mag_unc, s_param_phase_unc])
        s_param_ri, s_param_ri_cov = AmpPhase2DFT(
            s_param_mag, s_param_phase, s_param_UAP
        )

    elif name == "empirical_cov":
        file_reflection = "Beatty Line New Type-A Re Im Data.xlsx"
        dfs = pandas.read_excel(rel_path + file_reflection, skiprows=1, sheet_name=None)
        df0 = dfs[list(dfs.keys())[0]]

        f = np.r_[0, df0.iloc[:, 0]]  # GHz

        # load raw data from all individual experiments (separate sheets)
        s_param_runs_raw = np.array(
            [df.iloc[:, 1] + 1j * df.iloc[:, 2] for df in dfs.values()]
        )

        # include new datapoint at 0Hz which is required for discrete Fourier transform
        s_param_runs_0Hz = np.atleast_2d(np.absolute(s_param_runs_raw[:, 0])).T
        s_param_runs = np.concatenate([s_param_runs_0Hz, s_param_runs_raw], axis=1)

        # convert data into PyDynamic-real-imag representation
        s_param_ri_runs = c2ri(s_param_runs)

        # get mean and covariance from enhanced raw data
        s_param_ri = np.mean(s_param_ri_runs, axis=0)
        s_param_ri_cov = np.cov(s_param_ri_runs, rowvar=False)

        # get magnitude-phase representation for representation purposes
        s_param_mag, s_param_phase, s_param_UAP = DFT2AmpPhase(
            s_param_ri, s_param_ri_cov
        )
        s_param_mag_unc, s_param_phase_unc = mag_phase_unc_from_cov(s_param_UAP)

    elif name == "simulated":
        file_reflection = "S11_Sim_0_33_1000.s1p"
        df = pandas.read_csv(rel_path + file_reflection, skiprows=2, sep=" ")

        # load raw data
        f = df["!freq"]
        s_param_ri = np.r_[df["ReS11"], df["ImS11"]]
        s_param_ri_cov = np.zeros((len(s_param_ri), len(s_param_ri)))

        # get magnitude-phase representation for representation purposes
        s_param_mag, s_param_phase, s_param_UAP = DFT2AmpPhase(
            s_param_ri, s_param_ri_cov
        )
        s_param_mag_unc, s_param_phase_unc = mag_phase_unc_from_cov(s_param_UAP)

    if return_full_cov:
        if return_mag_phase:
            return f, s_param_mag, s_param_phase, s_param_UAP
        else:
            return f, s_param_ri, s_param_ri_cov
    else:
        if return_mag_phase:
            return f, s_param_mag, s_param_phase, s_param_mag_unc, s_param_phase_unc
        else:
            return f, s_param_ri, real_imag_unc_from_cov(s_param_ri_cov)


def mag_phase_unc_from_cov(s_param_UAP):
    N = len(s_param_UAP) // 2
    s_param_mag_unc = np.sqrt(np.diag(s_param_UAP)[:N])
    s_param_phase_unc = np.sqrt(np.diag(s_param_UAP)[N:])
    return s_param_mag_unc, s_param_phase_unc


def real_imag_unc_from_cov(s_param_ri_cov):
    return np.sqrt(np.diag(s_param_ri_cov))


def convert_ri_cov_to_mag_phase_unc(s_param_ri, s_param_ri_cov):
    s_param_mag, s_param_phase, s_param_UAP = DFT2AmpPhase(s_param_ri, s_param_ri_cov)
    s_param_mag_unc, s_param_phase_unc = mag_phase_unc_from_cov(s_param_UAP)

    return s_param_mag, s_param_phase, s_param_mag_unc, s_param_phase_unc


def elementwise_multiply(A, B, cov_A, cov_B):
    """
    elementwise multiplication of two real signals A and B
    """

    R = A * B
    cov_R = cov_A @ B @ cov_A.T + cov_B @ A @ cov_B.T

    return R, cov_R


def apply_window(A, W, cov_A, cov_W=None):
    """
    A \in R^2N uses PyDynamic real-imag representation of a complex vector \in C^N
    A = [A_re, A_im]

    W \in R^N is real-valued window

    R is result in real-imag representation, element-wise application of window (separately for real and imag values)
    R = [A_re * W, A_im * W]
    """
    R = A * np.r_[W, W]

    # this results from applying GUM
    # CA = block_diag(np.diag(W), np.diag(W))
    # cov_R = CA @ cov_A @ CA.T

    # this should be the same, but is computationally faster
    WW = np.r_[W, W]
    cov_R = WW * cov_A * WW[:, np.newaxis]

    if isinstance(cov_W, np.ndarray):
        # this results from applying GUM
        # N = len(W)
        # CW = block_diag(np.diag(A[:N]), np.diag(A[N:]))
        # cov_R += CW @ block_diag(cov_W, cov_W) @ CW.T

        # this should be the same, but is computationally faster
        cov_R += A * block_diag(cov_W, cov_W) * A[:, np.newaxis]

    return R, cov_R


def gate(ts, t_start=0.0, t_end=1.0, kind="kaiser", order=0.0):
    # kaiser order: 0 -> rect, +\infty -> gaussian

    mask = np.logical_and(ts >= t_start, ts <= t_end)
    gate = np.zeros(mask.size)

    width = np.sum(mask)
    if width:
        base_shape = signal.get_window((kind, order), width, fftbins=False)
        gate[mask] = base_shape

    # heuristic model for gate unc, probably too complicated :-)
    gate_unc = np.zeros(
        gate.size
    )  # 1e-5*np.abs(signal.filtfilt(*signal.butter(1, 0.30, "lowpass"), np.abs(np.diff(np.r_[gate, 0])), padlen=100))

    return gate, gate_unc


def agilent_gate(ts, t_start=0.0, t_end=1.0, kind="kaiser", order=6.5):
    # base gate
    Gate_rect, Gate_rect_unc = gate(ts, t_start, t_end, kind="kaiser", order=0.0)

    # spectrum of base gate
    gate_rect_ri, gate_rect_ri_cov = GUM_DFT(
        Gate_rect, np.diag(np.square(Gate_rect_unc))
    )

    # gate-window
    width = len(gate_rect_ri // 2)
    window_gate = signal.get_window((kind, order), width, fftbins=False)[width // 2 :]

    gate_times_window_ri, gate_times_window_ri_cov = apply_window(
        gate_rect_ri, window_gate, gate_rect_ri_cov, None
    )

    agilant_gate, agilant_gate_cov = GUM_iDFT(
        gate_times_window_ri, gate_times_window_ri_cov, Nx=ts.size
    )
    agilant_gate_unc = np.sqrt(np.diag(agilant_gate_cov))

    return agilant_gate, agilant_gate_unc


def make_twosided(x):
    # returns the twosided spectrum with f=0 at the start (default numpy style)
    # x = x_re + 1j * x_im
    x_twosided = np.r_[x, np.conjugate(x[1:][::-1])]  # odd signal length
    # x_twosided = np.r_[x, np.conjugate(x[::-1])]  # even signal length (default assumption for rfft)
    return x_twosided


def make_onesided(x):
    # returns the twosided spectrum with f=0 at the start (default numpy style)
    # x = x_re + 1j * x_im, (size = 2*N - 1)
    N = (x.size + 1) // 2  # odd signal length
    # N = x.size // 2   # even signal length
    x_onesided = x[:N]
    return x_onesided


def complex_convolution_of_two_half_spectra(X, Y):
    # complex valued X, Y

    # transform into full spectra
    XX = make_twosided(X)
    YY = make_twosided(Y)

    # otherwise not strict ascending order (numpy default has f=0 at index 0, not in the middle)
    XX = np.fft.fftshift(XX)
    YY = np.fft.fftshift(YY)

    # actual convolution
    RR = convolve1d(XX, YY, mode="wrap") / XX.size

    # undo shifting and make half spectrum
    R = make_onesided(np.fft.ifftshift(RR))

    return R

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
from PyDynamic.misc.tools import shift_uncertainty, trimOrPad


class PlotMethods:
    def __init__(self):
        pass

    def init_mag_phase_plot(self):
        fig, ax = plt.subplots(nrows=4, figsize=(8, 8), tight_layout=True)
        return fig, ax

    def add_data_to_mag_phase_plot(
        self, ax, f, mag, phase, mag_unc=None, phase_unc=None, l=None, c=None, lw=1
    ):
        # plotting arguments
        kwargs = {"label": l, "color": c, "linewidth": lw}

        # plot mag, mag_unc (if available)
        ax[0].plot(f, mag, **kwargs)
        if isinstance(mag_unc, (list, np.ndarray)):
            ax[1].semilogy(f, mag_unc, **kwargs)

        # plot phase, phase_unc (if available)
        ax[2].plot(f, np.rad2deg(phase), **kwargs)
        if isinstance(phase_unc, (list, np.ndarray)):
            ax[3].semilogy(f, np.rad2deg(phase_unc), **kwargs)

    def add_description_mag_phase_plot(self, ax):
        ax[0].legend()
        ax[0].set_title("Frequency Domain")
        ax[0].set_ylabel("magnitude [-]")
        ax[1].set_ylabel("magnitude unc [-]")
        ax[2].set_ylabel("phase [°]")
        ax[3].set_ylabel("phase unc [°]")
        ax[3].set_xlabel("f [GHz]")

        return ax


class BaseMethods:
    def __init__(self):
        self.plotting = PlotMethods()

    ############################################################
    ### high level calls #######################################
    ############################################################

    def compare_different_datasets(self, create_plot=True):
        # main empirical data (Type A, but only very few samples)
        data_emp = self.load_data(
            "empirical_cov", return_mag_phase=True, return_full_cov=False
        )

        # mag-phase-diag only (Type B)
        data_raw = self.load_data(
            "diag_only", return_mag_phase=True, return_full_cov=False
        )

        # simulated data (no unc)
        data_sim = self.load_data(
            "simulated", return_mag_phase=True, return_full_cov=False
        )

        if create_plot:
            # visualize raw input in the frequency domain
            fig, ax = self.plotting.init_mag_phase_plot()

            args_emp = {"l": "statistical cov.", "c": "tab:gray", "lw": 4}
            args_raw = {"l": "diag only (Type B)", "c": "tab:red"}
            args_sim = {"l": "simulated", "c": "tab:blue"}

            self.plotting.add_data_to_mag_phase_plot(ax, *data_emp, **args_emp)
            self.plotting.add_data_to_mag_phase_plot(ax, *data_raw, **args_raw)
            self.plotting.add_data_to_mag_phase_plot(ax, *data_sim, **args_sim)

            ax = self.plotting.add_description_mag_phase_plot(ax)
            plt.show()
    
    def perform_time_gating_method_1(self, data, config):
        
        # data shotcuts
        f = data["f"]
        s_ri = data["s_ri"]
        s_ri_cov = data["s_ri_cov"]

        # apply window in FD
        if config["window"] is not None:
            if config["window"]["val"] is not None:
                window = config["window"]["val"]
                window_cov = config["window"]["cov"]
                s_ri, s_ri_cov = self.apply_window(s_ri, window, s_ri_cov, window_cov)

        # zeropad signal is done indirectly during FFT, however pad-lengths need to be known
        if config["zeropad"] is not None:
            pad_len = config["zeropad"]["pad_len"]
            Nx = config["zeropad"]["Nx"]
        else:
            pad_len = 0
            Nx = len(s_ri) - 1
        Nx_mod = Nx + pad_len
        
        # apply the gate in the TD
        if config["gate"] is not None:
            time = config["gate"]["time"]
            time_mod = np.arange(0, Nx_mod) * (time[1] - time[0]) * Nx / float(Nx_mod) + time[0]  # from scipy resample
            gate_func = config["gate"]["gate_func"]
            gate_array, gate_array_cov = gate_func(time_mod)  # the gate corresponding to a higher time resolution (to match the zero padding)
        
            # convert (modified) reflection data to time domain
            S, S_cov = GUM_iDFT(s_ri, s_ri_cov, Nx=Nx_mod)

            # compensate the padding
            S *= Nx_mod / Nx
            S_cov *= Nx_mod / Nx

            # actual gating
            S_gated = S * gate_array
            S_gated_cov = np.diag(gate_array) @ S_cov @ np.diag(gate_array).T + np.diag(S) @ gate_array_cov @ np.diag(S).T
            s_gated_ri, s_gated_ri_cov = GUM_DFT(S_gated, S_gated_cov)

        # undo zero-padding
        if config["zeropad"] is not None:
            s_gated_ri = trimOrPad(s_gated_ri, length=len(f), real_imag_type=True) * Nx / Nx_mod
            s_gated_ri_cov = trimOrPad(s_gated_ri_cov, length=len(f), real_imag_type=True) * Nx / Nx_mod

        # undo windowing
        if config["window"] is not None:
            if config["window"]["val"] is not None:
                s_gated_ri, s_gated_ri_cov = self.apply_window(s_gated_ri, 1.0 / window, s_gated_ri_cov, None)

        # apply renormalization
        if config["renormalization"] is not None:
            renorm = config["renormalization"]
            s_gated_ri = renorm(s_gated_ri)
        
        return s_gated_ri
    
    def perform_time_gating_method_2(self, data, config):
        # MC around perform_time_gating_method_2_core
        pass
     
    def perform_time_gating_method_2_core(self, data, config):
        
        # data shotcuts
        f = data["f"]
        s_ri = data["s_ri"]

        # apply window in FD
        if config["window"] is not None:
            if config["window"]["val"] is not None:
                window = config["window"]["val"]
                s_ri, _ = self.apply_window(s_ri, window)

        # zeropad signal in FD
        if config["zeropad"] is not None:
            pad_len = config["zeropad"]["pad_len"]
            Nx = config["zeropad"]["Nx"]
            Nx_mod = Nx + pad_len
            
            s_ri = trimOrPad(s_ri, length=len(f) + pad_len//2, real_imag_type=True) * Nx_mod / Nx

        # transfer gate into FD
        if config["gate"] is not None:
            time = config["gate"]["time"]
            gate_func = config["gate"]["gate_func"]
            gate_array, _ = gate_func(time)
        
            gate_spectrum = np.fft.rfft(gate_array)

        # apply gate in the FD
        s_gated = self.complex_convolution_of_two_half_spectra(ri2c(s_ri), gate_spectrum)

        # undo zeropad signal
        if config["zeropad"] is not None:
            s_gated = trimOrPad(s_gated, length=len(f), real_imag_type=False) * Nx / Nx_mod

        # undo windowing
        if config["window"] is not None:
            if config["window"]["val"] is not None:
                s_gated = s_gated / window

        # apply renormalization
        if config["renormalization"] is not None:
            renorm = config["renormalization"]
            s_gated_ri = renorm(s_gated_ri)
        
        # convert back into real-imag-representation
        s_gated_ri = c2ri(s_gated)

        return s_gated_ri


    ############################################################
    ### low level calls ########################################
    ############################################################

    def load_data(self, name="", return_mag_phase=False, return_full_cov=True):
        rel_path = "data/" # "../data/"

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
            dfs = pandas.read_excel(
                rel_path + file_reflection, skiprows=1, sheet_name=None
            )
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
            s_param_mag_unc, s_param_phase_unc = self.mag_phase_unc_from_cov(
                s_param_UAP
            )

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
            s_param_mag_unc, s_param_phase_unc = self.mag_phase_unc_from_cov(
                s_param_UAP
            )

        if return_full_cov:
            if return_mag_phase:
                return f, s_param_mag, s_param_phase, s_param_UAP
            else:
                return f, s_param_ri, s_param_ri_cov
        else:
            if return_mag_phase:
                return f, s_param_mag, s_param_phase, s_param_mag_unc, s_param_phase_unc
            else:
                return f, s_param_ri, self.real_imag_unc_from_cov(s_param_ri_cov)

    def mag_phase_unc_from_cov(self, s_param_UAP):
        N = len(s_param_UAP) // 2
        s_param_mag_unc = np.sqrt(np.diag(s_param_UAP)[:N])
        s_param_phase_unc = np.sqrt(np.diag(s_param_UAP)[N:])
        return s_param_mag_unc, s_param_phase_unc

    def real_imag_unc_from_cov(self, s_param_ri_cov):
        return np.sqrt(np.diag(s_param_ri_cov))

    def convert_ri_cov_to_mag_phase_unc(self, s_param_ri, s_param_ri_cov):
        s_param_mag, s_param_phase, s_param_UAP = DFT2AmpPhase(
            s_param_ri, s_param_ri_cov
        )
        s_param_mag_unc, s_param_phase_unc = self.mag_phase_unc_from_cov(s_param_UAP)

        # sometimes issues if zero division
        s_param_phase = np.nan_to_num(s_param_phase)
        s_param_mag_unc = np.nan_to_num(s_param_mag_unc)
        s_param_phase_unc = np.nan_to_num(s_param_phase_unc)

        return s_param_mag, s_param_phase, s_param_mag_unc, s_param_phase_unc

    def elementwise_multiply(self, A, B, cov_A, cov_B):
        """
        elementwise multiplication of two real signals A and B
        """

        R = A * B
        cov_R = cov_A @ B @ cov_A.T + cov_B @ A @ cov_B.T

        return R, cov_R

    def apply_window(self, A, W, cov_A=None, cov_W=None):
        """
        A \in R^2N uses PyDynamic real-imag representation of a complex vector \in C^N
        A = [A_re, A_im]

        W \in R^N is real-valued window

        R is result in real-imag representation, element-wise application of window (separately for real and imag values)
        R = [A_re * W, A_im * W]
        """
        R = A * np.r_[W, W]
        cov_R = None

        # this results from applying GUM
        # CA = block_diag(np.diag(W), np.diag(W))
        # cov_R = CA @ cov_A @ CA.T

        # this should be the same, but is computationally faster
        if isinstance(cov_A, np.ndarray):
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

    def gate(self, ts, t_start=0.0, t_end=1.0, kind="kaiser", order=0.0):
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
        gate_cov = np.diag(np.square(gate_unc))

        return gate, gate_cov

    def agilent_gate(self, ts, t_start=0.0, t_end=1.0, kind="kaiser", order=6.5):
        # base gate
        Gate_rect, Gate_rect_unc = self.gate(
            ts, t_start, t_end, kind="kaiser", order=0.0
        )

        # spectrum of base gate
        gate_rect_ri, gate_rect_ri_cov = GUM_DFT(
            Gate_rect, np.diag(np.square(Gate_rect_unc))
        )

        # gate-window
        width = len(gate_rect_ri // 2)
        window_gate = signal.get_window((kind, order), width, fftbins=False)[
            width // 2 :
        ]

        gate_times_window_ri, gate_times_window_ri_cov = self.apply_window(
            gate_rect_ri, window_gate, gate_rect_ri_cov, None
        )

        agilant_gate, agilant_gate_cov = GUM_iDFT(
            gate_times_window_ri, gate_times_window_ri_cov, Nx=ts.size
        )
        agilant_gate_unc = np.sqrt(np.diag(agilant_gate_cov))

        return agilant_gate, agilant_gate_unc

    def make_twosided(self, x):
        # returns the twosided spectrum with f=0 at the start (default numpy style)
        # x = x_re + 1j * x_im
        x_twosided = np.r_[x, np.conjugate(x[1:][::-1])]  # odd signal length
        # x_twosided = np.r_[x, np.conjugate(x[::-1])]  # even signal length (default assumption for rfft)
        return x_twosided

    def make_onesided(self, x):
        # returns the twosided spectrum with f=0 at the start (default numpy style)
        # x = x_re + 1j * x_im, (size = 2*N - 1)
        N = (x.size + 1) // 2  # odd signal length
        # N = x.size // 2   # even signal length
        x_onesided = x[:N]
        return x_onesided

    def complex_convolution_of_two_half_spectra(self, X, Y):
        # complex valued X, Y

        # transform into full spectra
        XX = self.make_twosided(X)
        YY = self.make_twosided(Y)

        # otherwise not strict ascending order (numpy default has f=0 at index 0, not in the middle)
        XX = np.fft.fftshift(XX)
        YY = np.fft.fftshift(YY)

        # actual convolution
        RR = convolve1d(XX, YY, mode="wrap") / XX.size

        # undo shifting and make half spectrum
        R = self.make_onesided(np.fft.ifftshift(RR))

        return R

    def window(self, size, kind="nowindow"):
        if kind == "nowindow":
            w = None
            uw = None
        
        if kind == "neutral":
            w = np.ones(size)
            uw = np.zeros((size, size))
        
        if kind == "kaiser":
            w = signal.windows.get_window(("kaiser", 0.5 * np.pi), Nx=size, fftbins=False)
            uw = np.zeros((size, size))
        
        if not kind == "nowindow":
            w *= w.size / np.sum(w)
            uw *= w.size / np.sum(w)
        
        return w, uw
        
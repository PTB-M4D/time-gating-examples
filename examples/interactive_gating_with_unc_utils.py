import copy
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import colors
from PyDynamic.misc import complex_2_real_imag as c2ri
from PyDynamic.misc import real_imag_2_complex as ri2c
from PyDynamic.misc.tools import shift_uncertainty, trimOrPad
from PyDynamic.uncertainty.propagate_DFT import (
    GUM_DFT,
    AmpPhase2DFT,
    DFT2AmpPhase,
    GUM_iDFT,
    DFT_multiply,
)
from scipy import signal, special
from scipy.linalg import block_diag
from scipy.ndimage import convolve1d


class BaseMethods:
    def __init__(self):
        pass

    ############################################################
    ### high level calls #######################################
    ############################################################

    def compare_different_datasets(self, create_plot=True):
        # main empirical data (Type A, but only very few samples)
        data_emp = self.load_data("empirical_cov", return_full_cov=False)

        # mag-phase-diag only (Type B)
        data_raw = self.load_data("diag_only", return_full_cov=False)

        # simulated data (no unc)
        data_sim = self.load_data("simulated", return_full_cov=False)

        if create_plot:
            args_emp = {"l": "statistical cov.", "c": "tab:gray", "lw": 4}
            args_raw = {"l": "diag only (Type B)", "c": "tab:red"}
            args_sim = {"l": "simulated", "c": "tab:blue"}

            plotdata = [
                [data_emp, args_emp],
                [data_raw, args_raw],
                [data_sim, args_sim],
            ]

            self.mag_phase_plot(plotdata)

    def perform_time_gating_method_1(self, data, config, return_internal_data=False):
        # data shotcuts
        f = data["f"]
        s_ri = data["s_ri"]
        s_ri_cov = data["s_ri_cov"]

        # result data structure
        result = self.init_return_dict(return_internal_data)

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
            time_mod = (
                np.arange(0, Nx_mod) * (time[1] - time[0]) * Nx / float(Nx_mod)
                + time[0]
            )  # from scipy resample
            gate_func = config["gate"]["gate_func"]
            gate_array, gate_array_cov = gate_func(
                time_mod
            )  # the gate corresponding to a higher time resolution (to match the zero padding)

            if return_internal_data:
                result["internal"]["gate"]["t"] = time_mod
                result["internal"]["gate"]["val"] = gate_array
                result["internal"]["gate"]["cov"] = gate_array_cov

            # convert (modified) reflection data to time domain
            S, S_cov = GUM_iDFT(s_ri, s_ri_cov, Nx=Nx_mod)

            # compensate the padding
            S *= Nx_mod / Nx
            S_cov *= Nx_mod / Nx

            if return_internal_data:
                result["internal"]["modified"]["t"] = time_mod
                result["internal"]["modified"]["val"] = copy.copy(S)
                result["internal"]["modified"]["cov"] = copy.copy(S_cov)

            # actual gating
            S_gated = S * gate_array
            S_gated_cov = (
                np.diag(gate_array) @ S_cov @ np.diag(gate_array).T
                + np.diag(S) @ gate_array_cov @ np.diag(S).T
            )
            s_gated_ri, s_gated_ri_cov = GUM_DFT(S_gated, S_gated_cov)

        # undo zero-padding
        if config["zeropad"] is not None:
            s_gated_ri = (
                trimOrPad(s_gated_ri, length=len(f), real_imag_type=True) * Nx / Nx_mod
            )
            s_gated_ri_cov = (
                trimOrPad(s_gated_ri_cov, length=len(f), real_imag_type=True)
                * Nx
                / Nx_mod
            )

        # undo windowing
        if config["window"] is not None:
            if config["window"]["val"] is not None:
                s_gated_ri, s_gated_ri_cov = self.apply_window(
                    s_gated_ri, 1.0 / window, s_gated_ri_cov, None
                )

        # apply renormalization
        if config["renormalization"] is not None:
            if config["renormalization"] == "unitResponse":
                unit_response_gated = ri2c(self.get_gated_unit_response(data, config))
                unit_response_gated /= unit_response_gated[0]  # normalize?
                unit_response_gated_ri = c2ri(unit_response_gated)
            elif isinstance(config["renormalization"], np.ndarray):
                unit_response_gated_ri = config["renormalization"]
            else:
                raise ValueError("Renormalization of type is not supported.")
            # renormalize
            urg_inverted = c2ri(1.0 / ri2c(unit_response_gated_ri))
            s_gated_ri, s_gated_ri_cov = DFT_multiply(
                s_gated_ri,
                urg_inverted,
                s_gated_ri_cov,
                None,
            )

        # prepare output
        result["data"]["f"] = f
        result["data"]["val"] = s_gated_ri
        result["data"]["cov"] = s_gated_ri_cov

        return result

    def perform_time_gating_method_2(self, data, config, return_internal_data=False):
        # data shotcuts
        f = data["f"]
        s_ri = data["s_ri"]
        s_ri_cov = data["s_ri_cov"]

        # result data structure
        result = self.init_return_dict(return_internal_data)

        time = config["gate"]["time"]
        gate_func = config["gate"]["gate_func"]
        gate_array, gate_array_cov = gate_func(time)

        # draw gate and signal
        def draw_samples(size, x1, x1_cov, x2, x2_cov):
            SAMPLES_X1 = np.random.multivariate_normal(x1, x1_cov, size)
            SAMPLES_X2 = np.random.multivariate_normal(x2, x2_cov, size)
            return (SAMPLES_X1, SAMPLES_X2)

        # evaluate
        n_runs = 2000
        results = []
        for s_ri_mc, gate_array_mc in zip(
            *draw_samples(
                size=n_runs,
                x1=s_ri,
                x1_cov=s_ri_cov,
                x2=gate_array,
                x2_cov=gate_array_cov,
            )
        ):
            data_mc = {"f": f, "s_ri": s_ri_mc, "s_ri_cov": None}
            config_mc = copy.copy(config)
            config_mc["gate"] = {"val": gate_array_mc}

            s_gated_ri = self.perform_time_gating_method_2_core(data_mc, config)
            results.append(s_gated_ri)

        # extract mean and covariance
        s_gated_ri = np.mean(results, axis=0)
        s_gated_ri_cov = np.cov(results, rowvar=False)
        s_gated = ri2c(s_gated_ri)

        # prepare output
        result["data"]["f"] = f
        result["data"]["val"] = s_gated_ri
        result["data"]["cov"] = s_gated_ri_cov

        return result

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

            s_ri = (
                trimOrPad(s_ri, length=len(f) + pad_len // 2, real_imag_type=True)
                * Nx_mod
                / Nx
            )

        # transfer gate into FD
        if config["gate"] is not None:
            if "val" in config["gate"].keys():
                gate_array = config["gate"]["val"]  # enable Monte Carlo of this
            else:
                time = config["gate"]["time"]
                time_mod = (
                    np.arange(0, Nx_mod) * (time[1] - time[0]) * Nx / float(Nx_mod)
                    + time[0]
                )  # from scipy resample
                gate_func = config["gate"]["gate_func"]
                gate_array, _ = gate_func(
                    time_mod
                )  # the gate corresponding to a higher time resolution (to match the zero padding)

            gate_spectrum = np.fft.rfft(gate_array)

        # apply gate in the FD
        s_gated = self.complex_convolution_of_two_half_spectra(
            ri2c(s_ri), gate_spectrum
        )

        # undo zeropad signal
        if config["zeropad"] is not None:
            s_gated = (
                trimOrPad(s_gated, length=len(f), real_imag_type=False) * Nx / Nx_mod
            )

        # undo windowing
        if config["window"] is not None:
            if config["window"]["val"] is not None:
                s_gated = s_gated / window

        # apply renormalization
        if config["renormalization"] is not None:
            if config["renormalization"] == "unitResponse":
                unit_response_gated = ri2c(self.get_gated_unit_response(data, config))
                unit_response_gated /= unit_response_gated[0]  # normalize?
            elif isinstance(config["renormalization"], np.ndarray):
                unit_response_gated = config["renormalization"]
            else:
                raise ValueError("Renormalization of type is not supported.")
            # renormalize
            s_gated = s_gated / unit_response_gated

        # convert back into real-imag-representation
        s_gated_ri = c2ri(s_gated)

        return s_gated_ri

    ############################################################
    ### low level calls ########################################
    ############################################################

    def get_gated_unit_response(self, data, config):
        # recalculate method for unit response without unc.

        f = data["f"]
        unit_response_ri = c2ri(np.ones_like(f))

        data_renorm = {"f": f, "s_ri": unit_response_ri, "s_ri_cov": None}

        # remove renorm from config (otherwise leads to endless recursion)
        config_renorm = copy.deepcopy(config)
        config_renorm["renormalization"] = None

        unit_response_gated_ri = self.perform_time_gating_method_2_core(
            data_renorm, config_renorm
        )

        return unit_response_gated_ri

    def load_data(self, name="", return_mag_phase=False, return_full_cov=True):
        # check where data is (to preserve compatibility between jupyter and python call)
        if os.path.exists("data/"):
            rel_path = "data/"
        elif os.path.exists("../data/"):
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
        
        elif name == "simulated_with_unc":
            file_reflection = "S11_Sim_0_33_1000.s1p"
            df = pandas.read_csv(rel_path + file_reflection, skiprows=2, sep=" ")

            # load raw data
            f = df["!freq"]
            s_param_ri = np.r_[df["ReS11"], df["ImS11"]]
            s_param_ri_cov = np.diag(np.full_like(s_param_ri, 0.1))

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

    def gate(self, ts, t_start=0.0, t_end=1.0, kind="kaiser", kind_args=None):
        # kaiser order: 0 -> rect, +\infty -> gaussian

        mask = np.logical_and(ts >= t_start, ts <= t_end)
        gate = np.zeros(mask.size)

        width = np.sum(mask)
        if width:
            if kind in ["custom_VNA_tools_gate"]:
                span100 = t_end - t_start
                
                tdelta = span100 / 4
                if isinstance(kind_args, dict):
                    if "tdelta" in kind_args.keys():
                        tdelta = kind_args["tdelta"]
                        
                tdelta = kind_args["tdelta"] if "tdelta" in kind_args.keys() else span100 / 4

                mask1 = np.logical_and(ts >= t_start, ts <= t_start + 2*tdelta)
                width1 = np.sum(mask1)
                
                mask2 = np.logical_and(ts >= t_end - 2*tdelta, ts <= t_end)
                width2 = np.sum(mask2)

                base_shape = signal.get_window("hann", width1+width2, fftbins=False)

                gate[mask] = 1.0                    # center is flat 1.0
                gate[mask1] = base_shape[:width1]   # rising hanning
                gate[mask2] = base_shape[width1:]   # falling hanning

            else:
                if isinstance(kind_args, list):
                    window_config = (kind, *kind_args)
                elif isinstance(kind_args, dict):
                    window_config = (kind, *kind_args.values())
                else:
                    window_config = kind
                base_shape = signal.get_window(window_config, width, fftbins=False)
            
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
            w = signal.windows.get_window(
                ("kaiser", 0.5 * np.pi), Nx=size, fftbins=False
            )
            uw = np.zeros((size, size))

        if not kind == "nowindow":
            w *= w.size / np.sum(w)
            uw *= w.size / np.sum(w)

        return w, uw

    def init_return_dict(self, return_internal_data):
        result = {}
        result["data"] = self.data_series(freq=True)
        if return_internal_data:
            result["internal"] = {
                "gate": self.data_series(time=True),
                "modified": self.data_series(time=True),
            }

        return result

    def data_series(self, freq=False, time=False):
        mdo = {}

        if time:
            mdo["t"] = None

        if freq:
            mdo["f"] = None

        mdo["val"] = None
        mdo["cov"] = None

        return mdo

    def shift_time_data(self, t, val, cov):
        t_span = t[-1] - t[0]
        t_shifted = t - t_span / 2
        val_shifted, cov_shifted = shift_uncertainty(val, cov, shift=len(t) // 2)

        return t_shifted, val_shifted, cov_shifted

    def export_timedomain_to_excel(self, plotdata):
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H.%M.%SZ")
        with pandas.ExcelWriter(f"export_td_{timestamp}.xlsx") as writer:
            for data, args in plotdata:
                (t, val, cov) = data
                label = args["l"]
                t_shifted, val_shifted, cov_shifted = self.shift_time_data(t, val, cov)

                array_export = np.c_[
                    t_shifted, val_shifted, np.sqrt(np.diag(cov_shifted))
                ]
                df_export = pandas.DataFrame(array_export)
                df_export.columns = ["time", "signal", "signal_unc"]
                df_export.to_excel(writer, sheet_name=label)

    def export_freqdomain_to_excel(self, plotdata):
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H.%M.%SZ")
        with pandas.ExcelWriter(f"export_fd_{timestamp}.xlsx") as writer:
            for data, args in plotdata:
                (f, val, cov) = data
                label = args["l"]

                # convert to mag-phase-representation
                mag, phase, mag_unc, phase_unc = self.convert_ri_cov_to_mag_phase_unc(
                    val, cov
                )

                array_export = np.c_[f, mag, mag_unc, phase, phase_unc]
                df_export = pandas.DataFrame(array_export)
                df_export.columns = ["freq", "mag", "mag_unc", "phase", "phase_unc"]
                df_export.to_excel(writer, sheet_name=label)


    ############################################################
    ### plotting stuff #########################################
    ############################################################

    def add_decoration_to_plot(
        self, ax, base_style=None, custom_style=None, use_legend=True
    ):
        # the default look
        if isinstance(base_style, dict):
            if use_legend:
                ax[0].legend()
            for i_axis, style_adjustments in base_style.items():
                plt.setp(ax[i_axis], **style_adjustments)

        # some custom look
        if isinstance(custom_style, dict):
            for i_axis, style_adjustments in custom_style.items():
                plt.setp(ax[i_axis], **style_adjustments)

        return ax

    ###

    def mag_phase_plot(self, plotdata, use_base_style=True, custom_style=None):
        fig, ax = plt.subplots(nrows=4, figsize=(8, 8), sharex=True, tight_layout=True)

        for data, args in plotdata:
            self.add_data_to_mag_phase_plot(ax, *data, **args)

        ax = self.add_description_mag_phase_plot(
            ax, use_base_style=use_base_style, custom_style=custom_style
        )
        plt.show()

    def add_data_to_mag_phase_plot(
        self, ax, f, s_ri, s_ri_cov=None, l=None, c=None, lw=1
    ):
        # convert to mag-phase-representation
        mag, phase, mag_unc, phase_unc = self.convert_ri_cov_to_mag_phase_unc(
            s_ri, s_ri_cov
        )

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

    def add_description_mag_phase_plot(
        self, ax, use_base_style=True, custom_style=None
    ):
        if use_base_style:
            mag_phase_plot_style = {
                0: {
                    "title": "Frequency Domain",
                    "ylabel": "magnitude [-]",
                },
                1: {
                    "ylabel": "magnitude unc [-]",
                },
                2: {
                    "ylabel": "phase [°]",
                },
                3: {
                    "ylabel": "phase unc [°]",
                    "xlabel": "f [GHz]",
                },
            }
        else:
            mag_phase_plot_style = None

        ax = self.add_decoration_to_plot(ax, mag_phase_plot_style, custom_style)

        return ax

    ###

    def calculate_cnorm_ri_plot(self, plotdata):
        min_abs_cov = [
            1e-11,
        ]
        max_abs_cov = []

        for data, args in plotdata:
            f, s_ri, s_ri_cov = data

            min_abs_cov.append(np.abs(s_ri_cov).min())
            max_abs_cov.append(np.abs(s_ri_cov).max())

        linthresh = max(min_abs_cov)
        vmax = min(max_abs_cov)

        cnorm = colors.SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh)
        return cnorm

    def real_imag_covariance_plot(
        self, plotdata, use_base_style=True, custom_style=None
    ):
        fig, ax = plt.subplots(nrows=1, figsize=(6, 6), tight_layout=True)

        # extract relevant data
        (data, args) = plotdata[0]
        (f, s_ri, s_ri_cov) = data

        cnorm = self.calculate_cnorm_ri_plot(plotdata)

        img = ax.imshow(s_ri_cov, cmap="PuOr", norm=cnorm)
        fig.colorbar(img, ax=ax)

        ax = self.annotate_real_imag_plot(ax, f)

        if use_base_style:
            base_style = {0: {"title": f"Covariance of {args['l']} spectrum"}}
        else:
            base_style = None

        ax = self.add_decoration_to_plot(
            [ax],
            base_style,
            custom_style,
            use_legend=False,
        )

        plt.show()

    def annotate_real_imag_plot(self, ax, f, annotate_x=True, annotate_y=True):
        # define new tick positions
        labels_re = [0, 10, 20, 30]  # GHz
        labels_re = [0, 8, 16, 24]  # GHz
        labels_im = labels_re
        labels = labels_re + labels_im

        # define new labels for these positions
        ticks_re = [np.flatnonzero(f == l).item() for l in labels_re]
        ticks_im = [f.size + 1 + k for k in ticks_re]
        ticks = ticks_re + ticks_im

        # define colors for the labels to distinguish real and imag parts
        colors_re = ["k"] * len(ticks_re)
        colors_im = ["r"] * len(ticks_im)
        tick_colors = colors_re + colors_im

        if annotate_x:
            # xticks (label, position, color)
            ax.set_xticks(ticks=ticks, labels=labels)
            for ticklabel, c in zip(ax.get_xticklabels(), tick_colors):
                ticklabel.set_color(c)

            # axis label
            ax.set_xlabel("frequency [GHz]")

            # nice brace real part
            ax.annotate(
                "real",
                xy=(0.25, -0.01),
                xytext=(0.25, -0.10),
                fontsize=14,
                ha="center",
                va="bottom",
                xycoords="axes fraction",
                color="black",
                # arrowprops=dict(arrowstyle="-[, widthB=6.0, lengthB=.5"),
            )

            # nice brace imag part
            ax.annotate(
                "imag",
                xy=(0.75, -0.01),
                xytext=(0.75, -0.10),
                fontsize=14,
                ha="center",
                va="bottom",
                xycoords="axes fraction",
                color="red",
                # arrowprops=dict(arrowstyle="-[, widthB=6.0, lengthB=.5"),
            )

        if annotate_y:
            # yticks (label, position, color)
            ax.set_yticks(ticks=ticks, labels=labels)
            for ticklabel, c in zip(ax.get_yticklabels(), tick_colors):
                ticklabel.set_color(c)

            # axis label
            ax.set_ylabel("frequency [GHz]")

            # nice brace real part
            ax.annotate(
                "real",
                xy=(-0.01, 0.75),
                xytext=(-0.10, 0.75),
                fontsize=14,
                ha="left",
                va="center",
                xycoords="axes fraction",
                rotation=90,
                color="black",
                # arrowprops=dict(arrowstyle="-[, widthB=6.0, lengthB=.5"),
            )

            # nice brace imag part
            ax.annotate(
                "imag",
                xy=(-0.01, 0.25),
                xytext=(-0.10, 0.25),
                fontsize=14,
                ha="left",
                va="center",
                xycoords="axes fraction",
                rotation=90,
                color="red",
                # arrowprops=dict(arrowstyle="-[, widthB=6.0, lengthB=.5"),
            )

        return ax

    ###

    def time_domain_plot(
        self,
        plotdata,
        use_base_style=True,
        custom_style=None,
        last_dataset_has_own_axis=False,
    ):
        fig, ax = plt.subplots(nrows=2, figsize=(8, 8), sharex=True, tight_layout=True)

        if last_dataset_has_own_axis:
            ax2 = [None, None]
            ax2[0] = ax[0].twinx()
            ax2[1] = ax[1].twinx()

        for i, (data, args) in enumerate(plotdata):
            last_iteration = i == len(plotdata) - 1
            if last_dataset_has_own_axis and last_iteration:  # last data set
                self.add_data_to_time_domain_plot(ax2, *data, **args)
            else:
                self.add_data_to_time_domain_plot(ax, *data, **args)

        if last_dataset_has_own_axis:
            ax2[0].legend(loc="upper center")

        ax = self.add_description_time_domain_plot(
            ax, use_base_style=use_base_style, custom_style=custom_style
        )
        plt.show()

    def add_data_to_time_domain_plot(self, ax, t, val, cov=None, l=None, c=None, lw=1):
        # shift time series so zero is at the centers
        t_shifted, val_shifted, cov_shifted = self.shift_time_data(t, val, cov)

        # plotting arguments
        kwargs = {"label": l, "color": c, "linewidth": lw}

        # plot mag, mag_unc (if available)
        ax[0].plot(t_shifted, np.abs(val_shifted), **kwargs)
        if isinstance(cov_shifted, (list, np.ndarray)):
            unc_shifted = np.sqrt(np.diag(cov_shifted))
            ax[1].semilogy(t_shifted, unc_shifted, **kwargs)

    def add_description_time_domain_plot(
        self, ax, use_base_style=True, custom_style=None
    ):
        if use_base_style:
            time_domain_plot_style = {
                0: {
                    "title": "Time Domain",
                    "ylabel": "signal magnitude [-]",
                },
                1: {
                    "ylabel": "signal unc [-]",
                    "xlabel": "t [ns]",
                },
            }
        else:
            time_domain_plot_style = None

        ax = self.add_decoration_to_plot(ax, time_domain_plot_style, custom_style)

        return ax

    ###

    def time_domain_covariance_plot(
        self, plotdata, use_base_style=True, custom_style=None
    ):
        # note: only first element of plotdata is plotted

        fig, ax = plt.subplots(nrows=1, figsize=(8, 8), tight_layout=True)

        # extract relevant data
        (data, args) = plotdata[0]
        (t, val, cov) = data

        # shift time series so zero is at the centers
        t_shifted, val_shifted, cov_shifted = self.shift_time_data(t, val, cov)

        maxi = np.max(np.abs(cov_shifted))
        cnorm = colors.SymLogNorm(vmin=-maxi, vmax=maxi, linthresh=1e-14)
        extend = (t_shifted.min(), t_shifted.max(), t_shifted.min(), t_shifted.max())

        img0 = ax.imshow(cov_shifted, extent=extend, cmap="PuOr", norm=cnorm)
        fig.colorbar(img0, ax=ax)

        ax = self.add_description_time_domain_covariance_plot(
            ax, use_base_style=use_base_style, custom_style=custom_style
        )
        plt.show()

    def add_description_time_domain_covariance_plot(
        self, ax, use_base_style=True, custom_style=None
    ):
        if use_base_style:
            time_domain_covariance_plot_style = {
                0: {
                    "title": "Covariance of time signal",
                    "xlabel": "t [ns]",
                    "ylabel": "t [ns]",
                },
            }
        else:
            time_domain_covariance_plot_style = None

        ax = self.add_decoration_to_plot(
            [ax],
            time_domain_covariance_plot_style,
            custom_style,
            use_legend=False,
        )

        return ax

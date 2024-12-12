import copy
import numpy as np

import interactive_gating_with_unc_utils as utils
from PyDynamic.misc import complex_2_real_imag as c2ri
from PyDynamic.misc import real_imag_2_complex as ri2c

base = utils.BaseMethods()

# Compare different available datasets
base.compare_different_datasets()

# Load Dataset
data_raw = f, s11_ri, s11_ri_cov = base.load_data("empirical_cov")

# get corresponding time
Nx = len(s11_ri) - 1
t_span = 1 / np.mean(np.diff(f))  # original f, not f_mod
t = np.linspace(0, t_span, num=Nx)

# Time Gating Process Settings
# define window and gate
w, uw = base.window(size=len(f), kind="neutral")
def gate(t):
    return base.gate(t, t_start=0.0, t_end=0.18, kind="kaiser", kind_args={"order": 2.5 * np.pi})

# store settings in dicts
data = {"f": f, "s_ri": s11_ri, "s_ri_cov": s11_ri_cov}
config = {
    "window": {"val": w, "cov": uw},
    "zeropad": {"pad_len": 2500, "Nx": Nx},
    "gate": {"gate_func": gate, "time": t},
    "renormalization": "unitResponse",
}

# Perform the Gating Using Two Different Approaches
# Method 1: Direct Evaluation and Elementwise Multiplication in Time-Domain
# Method 2: Monte Carlo and Complex Convolution in Frequency-Domain
result_m1 = base.perform_time_gating_method_1(data, config, return_internal_data=True)
result_m2 = base.perform_time_gating_method_2(data, config)

# Visualiziations
# Visualize S-Parameter and Gate in the Time-Domain
# settings to obtain raw signal in the time domain
config_nomod = copy.deepcopy(config)
config_nomod["window"] = None
config_nomod["zeropad"] = None
config_nomod["renormalization"] = None
result_nomod = base.perform_time_gating_method_1(data, config_nomod, return_internal_data=True)

args_raw = {"l": "raw", "c": "tab:gray"}
args_mod = {"l": "modified", "c": "tab:green", "lw": 2}
args_gate = {"l": "gate", "c": "red"}

plotdata_timedomain = [
    [tuple(result_nomod["internal"]["modified"].values()), args_raw],
    [tuple(result_m1["internal"]["modified"].values()), args_mod],
    [tuple(result_m1["internal"]["gate"].values()), args_gate],
]

cs_timedomain = {0: {"xlim": (-0.1, 0.65)}, 1: {"yscale": "linear"}}

base.time_domain_plot(plotdata_timedomain[:2], custom_style=None)
base.time_domain_plot(
    plotdata_timedomain, custom_style=cs_timedomain, last_dataset_has_own_axis=True
)
base.time_domain_covariance_plot(
    [plotdata_timedomain[0]],
    custom_style={0: {"title": "Covariance of raw time signal"}},
)
base.time_domain_covariance_plot(
    [plotdata_timedomain[1]],
    custom_style={0: {"title": "Covariance of modified time signal"}},
)

# output results to file to inspect with other tools
if True:
    base.export_timedomain_to_excel(plotdata_timedomain)

# Visualize Results of Different Methods in Frequency Domain
args_raw = {"l": "raw", "c": "tab:gray"}
args_m1 = {"l": "gated (method 1)", "c": "tab:blue", "lw": 5}
args_m2 = {"l": "gated (method 2)", "c": "tab:orange"}

plotdata_comparison = [
    [data_raw, args_raw],
    [tuple(result_m1["data"].values()), args_m1],
    [tuple(result_m2["data"].values()), args_m2],
]

base.mag_phase_plot(
    plotdata_comparison,
    custom_style={3: {"ylim": (1e-3, 1.5e1)}},
)

for pdata in plotdata_comparison:
    base.real_imag_covariance_plot([pdata])

# compare covariance matrices
U1 = plotdata_comparison[1][0][2]
U2 = plotdata_comparison[2][0][2]
data_diff_m1_m2 = [data_raw[0], np.zeros_like(data_raw[1]), U2 / U1 - 1]
args_diff_m1_m2 = {"l": "comparison (U2 / U1 - 1)", "c": "tab:orange"}

base.real_imag_covariance_plot([[data_diff_m1_m2, args_diff_m1_m2]])
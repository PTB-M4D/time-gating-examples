import copy
import numpy as np
import interactive_gating_with_unc_utils as utils
from PyDynamic.misc import complex_2_real_imag as c2ri
from PyDynamic.misc import real_imag_2_complex as ri2c

base = utils.BaseMethods()

# data to be used for further processing
data_raw = f, s11_ri, s11_ri_cov = base.load_data("empirical_cov")

# get corresponding time
Nx = len(s11_ri) - 1
t_span = 1 / np.mean(np.diff(f))  # original f, not f_mod
t = np.linspace(0, t_span, num=Nx)

# define window and gate
w, uw = base.window(size=len(f), kind="neutral")
gate = lambda t: base.gate(t, t_start=0.0, t_end=0.18, kind="kaiser", order=2.5 * np.pi)


## RA: unit step only

# store settings in dicts
unit_response_ri = c2ri(np.ones_like(f))
unit_response_cov_ri = np.zeros((len(unit_response_ri), len(unit_response_ri)))
data_unitresp = {"f": f, "s_ri": unit_response_ri, "s_ri_cov": unit_response_cov_ri}
config_unitresp = {
    "window": None,
    "zeropad": {"pad_len": 0, "Nx": Nx},
    "gate": {"gate_func": gate, "time": t},
    "renormalization": None,
}

# plot settings
args_raw = {"l": "unit step", "c": "tab:gray"}
args_gate = {"l": "gate", "c": "tab:red"}
args_gated = {"l": "gated unit step", "c": "tab:red"}

# call different time gating implementations on same data+config
result = base.perform_time_gating_method_1(
    data_unitresp, config_unitresp, return_internal_data=True
)

# plot unit response in time domain
plotdata_td = [
    [tuple(result["internal"]["modified"].values()), args_raw],
    [tuple(result["internal"]["gate"].values()), args_gate],
]
cs_td = {0: {"xlim": (-0.1, 0.65)}, 1: {"yscale": "linear"}}
base.time_domain_plot(plotdata_td, custom_style=cs_td)

# plot unit response in frequency domain
plotdata_fd = [
    [data_unitresp.values(), args_raw],
    [tuple(result["data"].values()), args_gated],
]
cs_fd = {0: {"yscale": "log"}}
base.mag_phase_plot(plotdata_fd, custom_style=cs_fd)

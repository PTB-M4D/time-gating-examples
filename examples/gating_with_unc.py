import copy
import numpy as np
import interactive_gating_with_unc_utils as utils

base = utils.BaseMethods()

#
# base.compare_different_datasets()

# data to be used for further processing
data_raw = f, s11_ri, s11_ri_cov = base.load_data("empirical_cov")

# get corresponding time
Nx = len(s11_ri) - 1
t_span = 1 / np.mean(np.diff(f))  # original f, not f_mod
t = np.linspace(0, t_span, num=Nx)

# define window and gate
w, uw = base.window(size=len(f), kind="neutral")
gate = lambda t: base.gate(t, t_start=0.0, t_end=0.18, kind="kaiser", order=2.5 * np.pi)

# store settings in dicts
data = {"f": f, "s_ri": s11_ri, "s_ri_cov": s11_ri_cov}
config = {
    "window": {"val": w, "cov": uw},
    "zeropad": {"pad_len": 200, "Nx": Nx},
    "gate": {"gate_func": gate, "time": t},
    "renormalization": None,
}
config_nomod = copy.deepcopy(config)
config_nomod["window"] = None
config_nomod["zeropad"] = None

# call different time gating implementations on same data+config
result_m1 = base.perform_time_gating_method_1(data, config, return_internal_data=True)
result_m2 = base.perform_time_gating_method_2(data, config)
result_nomod = base.perform_time_gating_method_1(data, config_nomod, return_internal_data=True)

####################

args_raw = {"l": "raw", "c": "tab:gray"}
args_mod = {"l": "modified", "c": "tab:green", "lw": 2}
args_gate = {"l": "gate", "c": "red"}

plotdata_timedomain = [
    [tuple(result_nomod["internal"]["modified"].values()), args_raw],
    [tuple(result_m1["internal"]["modified"].values()), args_mod],
    [tuple(result_m1["internal"]["gate"].values()), args_gate],
]

cs_timedomain = {0: {"xlim": (-0.1, 0.65)}, 1: {"yscale": "linear"}}

base.export_to_excel(plotdata_timedomain)
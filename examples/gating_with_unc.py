import numpy as np
import interactive_gating_with_unc_utils as utils

base = utils.BaseMethods()

# base.compare_different_datasets()


# load data
f, s11_ri, s11_ri_cov = base.load_data("empirical_cov")

# get corresponding time
Nx = len(s11_ri) - 1
t_span = 1 / np.mean(np.diff(f))  # original f, not f_mod
t = np.linspace(0, t_span, num=Nx)

# define window and gate
w, uw = base.window(size=len(f), kind="neutral")
gate = lambda t: base.gate(t, t_start=0.0, t_end=0.18, kind="kaiser", order=2.5*np.pi)

# prepare calls to time gating methods
data = {"f": f, "s_ri": s11_ri, "s_ri_cov": s11_ri_cov}
config = {
    "window": {"val": w, "cov": uw},
    "zeropad": {"pad_len": 1000, "Nx": Nx},
    "gate": {"gate_func": gate, "time": t},
    "renormalization": None,
}

# call different time gating implementations on same data+config
a = base.perform_time_gating_method_1(data, config)
b = base.perform_time_gating_method_2_core(data, config)

# compare quickly
print(a)
print(b)
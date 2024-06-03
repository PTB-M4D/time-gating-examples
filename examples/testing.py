from interactive_gating_with_unc_utils import BaseMethods
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-10, 10, 1000)
t_start = 0
t_end = 4

base = BaseMethods()

gate, gate_unc = base.gate(
    ts=t,
    t_start=t_start,
    t_end=t_end,
    kind="custom_VNA_tools_gate",
    kind_args={
        "tdelta": 0.75,
    },
)

plt.plot(t, gate)
plt.show()

print("abc")

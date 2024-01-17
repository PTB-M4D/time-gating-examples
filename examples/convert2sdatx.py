import os
import numpy as np

from lxml import etree as ET
import metas_unclib as mu

from PyDynamic.misc import real_imag_2_complex as ri2c
from PyDynamic.misc import complex_2_real_imag as c2ri

import interactive_gating_with_unc_utils as utils

base = utils.BaseMethods()

class UNCLIB_helper:

    def __init__(self) -> None:
        pass

    def convert_pydy_to_unclib_cov(self, array):
        # convert results by METAS unclib to the structure used by PyDynamic
        # easier comparison
        # also strips negative frequencies, as 
        N = (array.shape[0])//2

        uarray = np.empty_like(array)
        uarray[::2,::2] = array[:N, :N]
        uarray[::2,1::2] = array[:N, N:]
        uarray[1::2,::2] = array[N:, :N]
        uarray[1::2,1::2] = array[N:, N:]
        
        return uarray


    def convert_metas_to_GUM_DFT_format(self, uarray):
        # convert results by METAS unclib to the structure used by PyDynamic
        # easier comparison
        # also strips negative frequencies, as 
        N = (uarray.size+1)//2
        
        val = mu.get_value(uarray)
        cov = mu.get_covariance(uarray)

        cov_RR = cov[::2,::2][:N, :N]
        cov_RI = cov[::2,1::2][:N, :N]
        cov_II = cov[1::2,1::2][:N, :N]

        uarray_ri = c2ri(val[:N])
        uarray_cov_ri = np.block([[cov_RR, cov_RI],[cov_RI.T, cov_II]])

        return uarray_ri, uarray_cov_ri


class SDATX_helper:
    # https://www.metas.ch/dam/metas/en/data/Fachbereiche/Hochfrequenz/vna-tools/vna_tools_data_data_format_v2.4.3.pdf.download.pdf/vna_tools_data_data_format_v2.4.3.pdf

    def __init__(
        self,
        output_path="default.sdatx",
        n_ports=1,
        port_type="",
        impedances=[50.0, 0.0],
    ):
        self.output_path = output_path

        data_cols, impedance_cols = self.get_column_names(n_ports)

        self.header = {
            "port_assignment": str(n_ports) + port_type,
            "reference_impedance": impedances,
            "data_column_description": data_cols,
            "impedance_column_description": impedance_cols,
        }

    def get_column_names(self, n_ports, cv_mask_type="real_imag_only"):
        data_column_names = ["Freq"]
        impedance_column_names = []

        # generate S-parameter value columns
        indices_s_param = np.arange(n_ports) + 1
        II_S, JJ_S = np.meshgrid(indices_s_param, indices_s_param)
        for i, j in zip(II_S.flatten(), JJ_S.flatten()):
            data_column_names.append(f"S[{i},{j}]re")
            data_column_names.append(f"S[{i},{j}]im")

        # generate covariance value columns
        cv_size = 2 * II_S.size
        cv_mask = self.get_mask(cv_size, cv_mask_type)
        indices_cv = np.arange(cv_size) + 1
        II_CV, JJ_CV = np.meshgrid(indices_cv, indices_cv)
        for i, j, m in zip(II_CV.flatten(), JJ_CV.flatten(), cv_mask.flatten()):
            if m:
                data_column_names.append(f"CV[{i},{j}]")

        # generate impedance column names
        for i in indices_s_param:
            impedance_column_names.append(f"Zr[{i}]re")
            impedance_column_names.append(f"Zr[{i}]im")

        return data_column_names, impedance_column_names

    def get_mask(self, size, kind):
        if kind == "full_cov":
            mask = np.full((size, size), fill_value=True)

        elif kind == "no_corr_between_s_param":
            half_size = size // 2
            mask = np.full((size, size), fill_value=True)
            mask[half_size:, :half_size] = False
            mask[:half_size, half_size:] = False

        elif kind == "real_imag_only":
            half_size = size // 2
            mask = np.full((size, size), fill_value=False)
            for i in range(half_size):
                mask[2 * i : 2 * (i + 1), 2 * i : 2 * (i + 1)] = True

        else:
            raise ValueError("Unsupported mask type")

        return mask

    def write_header(self):
        f = open(self.output_path, "w")
        f.write("")
        #
        f.write("SDATX\n")

        # ports
        ports = [f"{p:15s}" for p in self.header["port_assignment"]]
        f.write("Ports\n")
        f.write(" ".join(ports))
        f.write("\n")

        # impedances
        impedance_column_names = [
            f"{col:15s}" for col in self.header["impedance_column_description"]
        ]
        impedances = [f"{r:<15e}" for r in self.header["reference_impedance"]]
        f.write("Ports\n")
        f.write(" ".join(impedance_column_names))
        f.write("\n")
        f.write(" ".join(impedances))
        f.write("\n")

        # data column names
        impedance_column_names = [
            f"{col:15s}" for col in self.header["data_column_description"]
        ]
        f.write(" ".join(impedance_column_names))
        f.write("\n")

        # finish writing header
        f.close()

    def write_data(self, data):
        f = open(self.output_path, "a")

        out = data.array2string()
        f.write(out)
        f.close()


if __name__ == "__main__":
    sh = SDATX_helper(n_ports=1)

    sh.write_header()

    # load data
    f, s11_ri, s11_ri_cov = base.load_data("empirical_cov")
    n_cols = len(sh.header["data_column_description"])
    n_rows = f.size

    # init array
    data = np.empty((n_cols, n_rows))

    data[0,:] = f
    data[1,:] = s11_ri[:n_rows]
    data[2,:] = s11_ri[n_rows:]

    #data = np.hstack((f, ), axis=1)

    sh.write_data(data)
    print("")


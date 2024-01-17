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

    def add_frequencies(self):
        pass

    def add_ports(self):
        pass

    def add_port_impedances(self):
        pass

    def add_frequency_conversion(self):
        pass

    def add_data(self):
        pass

    def add_frequency_list(self):
        pass

if __name__ == "__main__":
    sh = SDATX_helper(n_ports=1)

    print("")


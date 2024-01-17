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
        N = (array.shape[0]) // 2

        uarray = np.empty_like(array)
        uarray[::2, ::2] = array[:N, :N]
        uarray[::2, 1::2] = array[:N, N:]
        uarray[1::2, ::2] = array[N:, :N]
        uarray[1::2, 1::2] = array[N:, N:]

        return uarray

    def convert_metas_to_GUM_DFT_format(self, uarray):
        # convert results by METAS unclib to the structure used by PyDynamic
        # easier comparison
        # also strips negative frequencies, as
        N = (uarray.size + 1) // 2

        val = mu.get_value(uarray)
        cov = mu.get_covariance(uarray)

        cov_RR = cov[::2, ::2][:N, :N]
        cov_RI = cov[::2, 1::2][:N, :N]
        cov_II = cov[1::2, 1::2][:N, :N]

        uarray_ri = c2ri(val[:N])
        uarray_cov_ri = np.block([[cov_RR, cov_RI], [cov_RI.T, cov_II]])

        return uarray_ri, uarray_cov_ri


class SDATX_helper:
    # https://www.metas.ch/dam/metas/en/data/Fachbereiche/Hochfrequenz/vna-tools/vna_tools_data_data_format_v2.4.3.pdf.download.pdf/vna_tools_data_data_format_v2.4.3.pdf

    def __init__(self):
        pass

    def create_xml_object(self, config):
        # load data
        freqs, uarray = self.load_data("empirical_cov")
        print("-- data loaded ---")

        # init xml
        main_node = self._init_xml_object()

        # header
        self._add_frequencies(main_node, freqs)
        self._add_ports(main_node, config["n_ports"])
        self._add_port_impedances(main_node, config["impedances"])
        self._add_frequency_conversion(main_node, config["n_ports"])
        print("-- metadata inserted ---")

        # data
        self._add_data(main_node, uarray)
        print("-- data inserted ---")

        return main_node

    def _init_xml_object(self):
        nsmap = {
            "xsd": "http://www.w3.org/2001/XMLSchema",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
        main_node = ET.Element("SParamData", nsmap=nsmap)

        return main_node

    def _add_frequencies(self, main_node, freqs):
        node = ET.SubElement(main_node, "FrequencyList")
        for f in freqs:
            subnode = ET.SubElement(node, "Frequency")
            subnode.text = str(f * 1e9)  # f in GHz

    def _add_ports(self, main_node, n_ports):
        node = ET.SubElement(main_node, "PortList")

        for i in range(n_ports):
            subnode = ET.SubElement(node, "Port")
            subnode.text = str(i + 1)

    def _add_port_impedances(self, main_node, impedances):
        node = ET.SubElement(main_node, "PortZrList")

        for item in impedances:
            imp_re = item[0]
            imp_im = item[1]
            subnode = ET.SubElement(node, "PortZr")

            sn_real = ET.SubElement(subnode, "Real")
            sn_real_val = ET.SubElement(sn_real, "Value")
            sn_real_dep = ET.SubElement(sn_real, "Dependencies")
            sn_real_val.text = str(imp_re)

            sn_imag = ET.SubElement(subnode, "Imag")
            sn_imag_val = ET.SubElement(sn_imag, "Value")
            sn_imag_dep = ET.SubElement(sn_imag, "Dependencies")
            sn_imag_val.text = str(imp_im)

    def _add_frequency_conversion(self, main_node, n_ports):
        node = ET.SubElement(main_node, "FrequencyConversionList")

        for i in range(n_ports):
            subnode = ET.SubElement(node, "FrequencyConversion")

            tst = ET.SubElement(subnode, "TestReceiver")
            ref = ET.SubElement(subnode, "ReferenceReceiver")
            src = ET.SubElement(subnode, "Source")

            for n in [tst, ref, src]:
                num = ET.SubElement(n, "Numerator")
                num.text = "1"

                den = ET.SubElement(n, "Denominator")
                den.text = "1"

                off = ET.SubElement(n, "Offset")
                off.text = "0"

    def _add_data(self, main_node, uarray):
        node = ET.SubElement(main_node, "Data")

        # convert uarray to xml structure
        uarray_xml_string = mu.ustorage.to_xml_string(uarray)
        data_root = ET.fromstring(uarray_xml_string.encode("utf16"))

        # for every element of uarray, create corresponding entry in main_node
        data_items = data_root.find("Data").getchildren()
        for datapoint in data_items:

            subnode = ET.SubElement(node, "Frequency")
            rec = ET.SubElement(subnode, "ReceiverPort")
            src = ET.SubElement(rec, "SourcePort")
            for child in datapoint.getchildren():
                src.append(child)


    def write_xml_object(self, main_node, filepath):
        s = ET.tostring(main_node, pretty_print=True, xml_declaration=True)
        f = open(filepath, "wb")
        f.write(s)
        f.close()

    def write_compressed_xml_object(self, obj, filepath):
        pass

    def load_data(self, kind):
        uh = UNCLIB_helper()

        f, val_ri, cov_ri = base.load_data(kind)

        val = ri2c(val_ri)
        cov = uh.convert_pydy_to_unclib_cov(cov_ri)

        uarray = mu.ucomplexarray(val, cov)

        return f, uarray


if __name__ == "__main__":
    sh = SDATX_helper()

    config = {
        "n_ports": 1,
        "impedances": [[50.0, 0.0]],
    }

    xml = sh.create_xml_object(config)
    sh.write_xml_object(xml, "test_ET.sdatx")

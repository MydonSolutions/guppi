import unittest

from guppi import header as guppi_header


class TestGuppiHeader(unittest.TestCase):
    def test_auto_init_GuppiRawProperties(self):
        PROPERTY_FSET_CLASS_MAP = {
            guppi_header.GuppiRawHeader.telescope.fset: {
                cls.telescope.fget(): cls
                for cls in [
                    guppi_header.GuppiRawAtaHeader,
                    guppi_header.GuppiRawCosmicHeader,
                    guppi_header.GuppiRawMeerkatHeader,
                    guppi_header.GuppiRawHeader
                ]
            }
        }

        for fset, value_class_map in PROPERTY_FSET_CLASS_MAP.items():
            for value, cls in value_class_map.items():
                header_dict = {}
                fset(header_dict, value)
                assert isinstance(
                    guppi_header.auto_init_GuppiRawHeader(header_dict),
                    cls
                ), f"Header {header_dict} did not invoke {cls}."


if __name__ == '__main__':
    unittest.main()

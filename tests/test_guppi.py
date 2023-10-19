import unittest

import numpy

from guppi import GuppiHandler
from guppi.header import GuppiRawHeader


class TestGuppi(unittest.TestCase):
    def test_write_and_read_integers(
        self,
        shape = (4, 16, 8, 2),
        seed = int(3141592635**3),
        blocks = 7,
        sdtype = numpy.int8
    ):
        get_integers = lambda rng: rng.integers(numpy.iinfo(sdtype).min, numpy.iinfo(sdtype).max, size=shape, dtype=sdtype)
        gr_header = GuppiRawHeader(
            TELESCOP="SoftwareUnitTest",
        )
        cdtype = GuppiHandler.NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP[sdtype]
        gr_data = numpy.zeros(shape, cdtype)

        rng = numpy.random.default_rng(seed)

        for i in range(blocks):
            gr_data[:]['re'] = get_integers(rng)
            gr_data[:]['im'] = get_integers(rng)
            
            GuppiHandler.write_to_file(
                f"test.{i//5:04d}.raw",
                gr_header,
                gr_data,
                file_open_mode = 'wb' if i%5 == 0 else 'ab'
            )
        
        rng = numpy.random.default_rng(seed)
        blocks_validated = 0
        for i, block in enumerate(
            GuppiHandler("test.").blocks(astype=sdtype, viewtype=cdtype)
        ):
            assert i < blocks
            gr_header, gr_data = block
            assert gr_header.blockshape == shape
            assert (gr_data[:]['re'] == get_integers(rng)).all(), f"real mismatch in block #{i+1}"
            assert (gr_data[:]['im'] == get_integers(rng)).all(), f"imag mismatch in block #{i+1}"
            blocks_validated += 1
        
        assert blocks_validated == blocks


if __name__ == '__main__':
    unittest.main()

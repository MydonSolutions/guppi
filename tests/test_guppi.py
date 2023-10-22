import unittest

import numpy

from guppi import GuppiRawHandler
from guppi.header import GuppiRawHeader


class TestGuppi(unittest.TestCase):
    @staticmethod
    def _get_guppi_integers(rng, shape, dtype):
        return rng.integers(
            numpy.iinfo(dtype).min,
            numpy.iinfo(dtype).max,
            size=shape,
            dtype=dtype
        )

    def test_write_and_read_integers(
        self,
        shape=(4, 16, 8, 2),
        seed=int(3141592635**3),
        blocks=7,
        sdtype=numpy.int8
    ):
        gr_header = GuppiRawHeader(
            TELESCOP="SoftwareUnitTest",
        )
        cdtype = GuppiRawHandler.NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP[sdtype]
        gr_data = numpy.zeros(shape, cdtype)

        rng = numpy.random.default_rng(seed)

        for i in range(blocks):
            gr_data[:]['re'] = self._get_guppi_integers(
                rng,
                shape,
                sdtype
            )
            gr_data[:]['im'] = self._get_guppi_integers(
                rng,
                shape,
                sdtype
            )

            GuppiRawHandler.write_to_file(
                f"test.{i//5:04d}.raw",
                gr_header,
                gr_data,
                file_open_mode=('wb' if i%5 == 0 else 'ab')
            )

        rng = numpy.random.default_rng(seed)
        blocks_validated = 0
        for i, block in enumerate(
            GuppiRawHandler("test.").blocks(astype=sdtype, viewtype=cdtype)
        ):
            assert i < blocks
            gr_header, gr_data = block
            assert gr_header.blockshape == shape
            assert (gr_data[:]['re'] == self._get_guppi_integers(
                rng,
                shape,
                sdtype
            )).all(), f"real mismatch in block #{i+1}"
            assert (gr_data[:]['im'] == self._get_guppi_integers(
                rng,
                shape,
                sdtype
            )).all(), f"imag mismatch in block #{i+1}"
            blocks_validated += 1

        assert blocks_validated == blocks

    def test_write_and_read_4bit(
        self,
        shape=(4, 16, 8, 2),
        seed=int(3141592635**3),
        blocks=3
    ):
        get_integers = lambda rng: rng.integers(
            -8,
            7,
            size=shape,
            dtype=numpy.int8
        )
        gr_header = GuppiRawHeader(
            TELESCOP="SoftwareUnitTest",
        )
        cdtype = GuppiRawHandler.NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP[numpy.int8]

        rng = numpy.random.default_rng(seed)

        for i in range(blocks):
            data_real = get_integers(rng).flatten()
            data_imag = get_integers(rng).flatten()

            gr_data = numpy.array(
                [
                    (real << 4) + (data_imag[i] & 0xf)
                    for i, real in enumerate(data_real)
                ],
                dtype=numpy.int8
            ).reshape(shape)

            GuppiRawHandler.write_to_file(
                "test_4bit.0000.raw",
                gr_header,
                gr_data,
                file_open_mode=('wb' if i == 0 else 'ab')
            )

        rng = numpy.random.default_rng(seed)
        blocks_validated = 0
        for i, block in enumerate(
            GuppiRawHandler("test_4bit.0000.raw").blocks(
                astype=numpy.int8,
                viewtype=cdtype
            )
        ):
            assert i < blocks
            gr_header, gr_data = block
            assert gr_header.nof_bits == 4
            assert gr_header.blockshape == shape
            assert (
                gr_data[:]['re'] == get_integers(rng)
            ).all(), f"real mismatch in block #{i+1}"
            assert (
                gr_data[:]['im'] == get_integers(rng)
            ).all(), f"imag mismatch in block #{i+1}"
            blocks_validated += 1

        assert blocks_validated == blocks


if __name__ == '__main__':
    unittest.main()

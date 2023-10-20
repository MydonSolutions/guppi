import os
import glob
import logging
from typing import List, Tuple, Union

import numpy

from guppi.header import GuppiRawHeader, auto_init_GuppiRawHeader


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class GuppiHandler:
    """
    GUPPI RAW file reader
    """

    NBITS_NUMPY_INTTYPE_MAP = {
        4: numpy.int8,
        8: numpy.int8
    }

    NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP = {
        numpy.int8: numpy.dtype([('re', numpy.int8), ('im', numpy.int8)]),
        numpy.int16: numpy.dtype([('re', numpy.int16), ('im', numpy.int16)]),
        numpy.int32: numpy.dtype([('re', numpy.int32), ('im', numpy.int32)]),
        numpy.float32: numpy.complex64
    }

    def __init__(self, guppi_filepaths: Union[str, List[str]]):
        if isinstance(guppi_filepaths, str):
            if not os.path.exists(guppi_filepaths):
                logger.warning(f"Given GUPPI RAW filepath does not exist, assuming it is the stem.")
                guppi_filepaths = glob.glob(f"{guppi_filepaths}*.raw")
                logger.debug(f"Found {guppi_filepaths}.")
            else:
                guppi_filepaths = [guppi_filepaths]

        self._guppi_filepaths = guppi_filepaths
        self._guppi_file_index = 0
        self._guppi_file_handle = None

    def _read_header_entry(self) -> str:
        try:
            header_entry = self._guppi_file_handle.read(80).decode()
            if len(header_entry) == 0:
                raise EOFError()
            if len(header_entry) < 80:
                raise ValueError("Could not read 80 characters for a header entry.")
            return header_entry
        except UnicodeDecodeError as err:
            pos = self._guppi_file_handle.tell()
            self._guppi_file_handle.seek(pos - 321)
            preceeding_bytes = self._guppi_file_handle.read(240)
            next_bytes = self._guppi_file_handle.read(240)

            logger.error(f"UnicodeDecodeError in {self._guppi_filepaths[self._guppi_file_index]} at position: {pos}")
            logger.error(f"Preceeding bytes: {preceeding_bytes}")
            logger.error(f"Proceeding bytes: {next_bytes}")
            # logger.error(f"Block #{raw_blocks} starting at {block_header_start}")

            raise RuntimeError("Failed to read GUPPI header entry.") from err

    def _header_entries(self) -> Tuple[str, Union[int, float, str]]:
        while True:
            header_entry = self._read_header_entry()
            if header_entry == "END" + " "*77:
                break

            key, value = map(lambda s: s.strip(), header_entry.split("=", 1))
            try:
                try:
                    value = int(value)
                except:
                    value = float(value)
            except:
                # must be a str value, drop enclosing single-quotes
                assert value[0] == value[-1] == "'"
                value = value[1:-1].strip()

            yield key, value

    def _seek_align_directio(self):
        pos = self._guppi_file_handle.tell()
        remainder = pos % 512
        if remainder == 0:
            return
        self._guppi_file_handle.seek(512 - remainder, 1)

    def open_next_file(self):
        if self._guppi_file_handle is not None:
            self._guppi_file_handle.close()
            self._guppi_file_index += 1

        self._guppi_file_handle = open(self._guppi_filepaths[self._guppi_file_index], 'rb')

    def read_next_header(self) -> GuppiRawHeader:
        entry_iter = self._header_entries()
        header_entries = {}
        try:
            key, value = next(entry_iter)
            header_entries[key] = value
        except EOFError:
            self.open_next_file()
            entry_iter = self._header_entries()

        for key, value in entry_iter:
            header_entries[key] = value

        gr_header: GuppiRawHeader = auto_init_GuppiRawHeader(header_entries)

        if gr_header.directio:
            self._seek_align_directio()

        return gr_header

    def read_block(self, gr_header, astype=None, viewtype=None):
        if astype is None:
            astype = numpy.float32
        if viewtype is None:
            viewtype = GuppiHandler.NUMPY_INTTYPE_COMPLEXVIEWTYPE_MAP[astype]

        gr_block = numpy.fromfile(
            self._guppi_file_handle,
            dtype=GuppiHandler.NBITS_NUMPY_INTTYPE_MAP[gr_header.nof_bits],
            count=gr_header.blocksize
        )
        if gr_header.directio:
            self._seek_align_directio()

        if gr_header.nof_bits == 4:
            # every 1 sample is a complex number (8bit) => (4bit + 4bit)
            gr_block = gr_block.repeat(2)
            gr_block[0::2] = gr_block[0::2] >> 4
            gr_block[1::2] = gr_block[1::2] << 4 >> 4

        # cast up to astype, then construct view (typically as complex)
        gr_block = gr_block.astype(astype).view(viewtype)

        return gr_block.reshape(gr_header.blockshape)

    def seek_past_block(self, gr_header):
        self._guppi_file_handle.seek(gr_header.blocksize, 1)
        if gr_header.directio:
            self._seek_align_directio()

    def blocks(self, astype=None, viewtype=None):
        self._guppi_file_index = 0
        self._guppi_file_handle = None
        self.open_next_file()

        block_index = 0
        reference_block_shape = None
        while True:
            try:
                header = self.read_next_header()
            except IndexError:
                break
            block_index += 1

            if reference_block_shape is None:
                self.validate_header(header)
                reference_block_shape = header.blockshape
            elif reference_block_shape != header.blockshape:
                raise RuntimeError(f"The header indicates a different blockshape at block #{block_index} ")

            yield (
                header,
                self.read_block(
                    header,
                    astype=astype,
                    viewtype=viewtype,
                )
            )

    def __del__(self):
        if self._guppi_file_handle is not None:
            self._guppi_file_handle.close()

    def validate_header(self, header):
        if header.nof_bits not in GuppiHandler.NBITS_NUMPY_INTTYPE_MAP:
            raise NotImplementedError(f"Only {list(GuppiHandler.NBITS_NUMPY_INTTYPE_MAP.keys())}-bit data are implemented, cannot handle {header.nof_bits}.")
        if header.nof_polarizations not in [1,2]:
            raise NotImplementedError(f"Only 1 and 2 polarizations are expected, not {header.nof_polarizations}.")

    @staticmethod
    def write_to_file(
        filepath: str,
        header: GuppiRawHeader,
        datablock: numpy.ndarray,
        file_open_mode: str = "ab"
    ):
        A, F, T, P = datablock.shape
        header.observed_nof_channels = A*F
        header.nof_antennas = A
        header.nof_polarizations = P
        datablock_bytes = datablock.tobytes()
        header.blocksize = len(datablock_bytes)
        header.nof_bits = (len(datablock_bytes)*8)//(numpy.prod(datablock.shape)*2)

        header_str = header.to_fits()

        with open(filepath, file_open_mode) as fio:
            fio.write(header_str.encode())
            if header.directio:
                header_len = len(header_str)
                padded_len = ((header_len + 511) // 512) * 512
                fio.write(b"*"*(padded_len - header_len))

            bytes_written = fio.write(datablock_bytes)
            if header.directio:
                padded_len = ((bytes_written + 511) // 512) * 512
                fio.write(b" "*(padded_len - bytes_written))

"""Read the same NPZ semantic vectors with an optional faster DEFLATE codec.

Each call reads and checks the original file. There is no retained vector cache,
format conversion, normalization, or precomputation outside the caller's timer.
"""
import io
import math
import struct
import zipfile

import numpy as np

try:
    import isal
    from isal import isal_zlib
except ImportError:
    isal = None
    isal_zlib = None


def _npy_from_bytes(raw):
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if dtype.hasobject:
        raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
    array = np.frombuffer(raw, dtype=dtype, offset=stream.tell())
    if array.size != math.prod(shape):
        raise ValueError("NPY array size does not match its header")
    return array.reshape(shape, order="F" if fortran else "C")


def _read_array(archive, name):
    info = archive.getinfo(name)
    if info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 1:
        with archive.open(info) as member:
            return np.lib.format.read_array(member, allow_pickle=False)
    handle = archive.fp
    handle.seek(info.header_offset)
    header = handle.read(30)
    if len(header) != 30:
        raise zipfile.BadZipFile("Truncated local ZIP header")
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04" or fields[3] != info.compress_type:
        raise zipfile.BadZipFile("Invalid local ZIP header")
    filename_size, extra_size = fields[-2:]
    filename = handle.read(filename_size)
    codec = "utf-8" if fields[2] & 0x800 else "cp437"
    if filename.decode(codec) != info.orig_filename:
        raise zipfile.BadZipFile("Local and central ZIP filenames differ")
    if len(handle.read(extra_size)) != extra_size:
        raise zipfile.BadZipFile("Truncated local ZIP extra data")
    compressed = handle.read(info.compress_size)
    if len(compressed) != info.compress_size:
        raise zipfile.BadZipFile("Truncated ZIP member")
    # Known output size avoids repeated decompressor buffer growth/copies.
    raw = isal_zlib.decompress(compressed, -15, max(info.file_size, 1))
    if len(raw) != info.file_size or isal_zlib.crc32(raw) != info.CRC:
        raise zipfile.BadZipFile("ZIP member size or CRC mismatch")
    return _npy_from_bytes(raw)


def load_semantic_arrays(path):
    if isal_zlib is None:
        with np.load(path, allow_pickle=False) as payload:
            return payload["graph_ids"], payload["embeddings"], "numpy_zipfile_zlib"
    with zipfile.ZipFile(path) as archive:
        return (_read_array(archive, "graph_ids.npy"),
                _read_array(archive, "embeddings.npy"), f"isal_{isal.__version__}_deflate_crc_checked")

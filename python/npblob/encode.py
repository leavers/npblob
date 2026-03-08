import sys
import typing as t
from struct import pack, unpack

import numpy as np

try:
    from orjson import dumps, loads

except (ImportError, ModuleNotFoundError):
    try:
        from ujson import dumps, loads  # type: ignore
    except (ImportError, ModuleNotFoundError):
        from json import dumps, loads  # type: ignore


def json_loadb(blob: bytes) -> t.Any:
    return loads(blob)


def json_dumpb(data: t.Any) -> bytes:
    return dumps(data)


EndianType = t.Literal["<", ">", "="]
ExtraEncodingType = t.Literal["bytes", "json", "msgpack"]
MatExtraPair = t.Tuple[np.ndarray, t.Any]

# Version and format constants
VERSION_V1 = 0x10  # 0001 0000 - version 1, may have extra
VERSION_V1_MUST_EXTRA = 0x11  # 0001 0001 - version 1, must have extra

# Extra flag format: high 4 bits = 0000, low 4 bits = type
EXTRA_FLAG_BYTES = 0x01  # 0000 0001
EXTRA_FLAG_JSON = 0x02  # 0000 0010
EXTRA_FLAG_MSGPACK = 0x03  # 0000 0011

_DTYPE_FLAGS_ENCODING = {
    "u1": 1,
    "u2": 2,
    "u4": 3,
    "u8": 4,
    "i1": 5,
    "i2": 6,
    "i4": 7,
    "i8": 8,
    "f2": 9,
    "f4": 10,
    "f8": 11,
    "f16": 12,
}
_DTYPE_FLAGS_DECODING = {
    1: "u1",
    2: "u2",
    3: "u4",
    4: "u8",
    5: "i1",
    6: "i2",
    7: "i4",
    8: "i8",
    9: "f2",
    10: "f4",
    11: "f8",
    12: "f16",
}
_EXTRA_ENCODING_FLAGS = {
    "bytes": EXTRA_FLAG_BYTES,
    "json": EXTRA_FLAG_JSON,
    "msgpack": EXTRA_FLAG_MSGPACK,
}
_EXTRA_DECODING_FLAGS = {
    EXTRA_FLAG_BYTES: "bytes",
    EXTRA_FLAG_JSON: "json",
    EXTRA_FLAG_MSGPACK: "msgpack",
}


class MsgpackBytes(bytes): ...


def _encode(
    arr: np.ndarray,
    extra: t.Optional[t.Union[t.Dict[str, t.Any], MsgpackBytes, bytes]] = None,
    endian: EndianType = "=",
) -> bytes:
    sys_endian = "<" if sys.byteorder == "little" else ">"
    target_endian = endian if endian != "=" else sys_endian

    dtype = arr.dtype
    byteorder = dtype.byteorder
    if (byteorder == "=" and sys_endian != target_endian) or (
        byteorder != "=" and byteorder != target_endian
    ):
        arr = arr.astype(dtype.newbyteorder())
        dtype = arr.dtype
        byteorder = dtype.byteorder

    dtype_flag = _DTYPE_FLAGS_ENCODING[f"{dtype.kind}{dtype.alignment}"]
    if target_endian == ">":
        dtype_flag = -dtype_flag

    shape = arr.shape
    shape_size = len(shape)
    if shape_size >= 128:
        raise ValueError("Number of array dimensions too large")
    max_dim = np.max(shape)
    if max_dim < 65536:
        shape_dtype = f"{target_endian}u2"
        big_shape = False
    elif max_dim < 4294967296:
        shape_dtype = f"{target_endian}u4"
        big_shape = True
    else:
        raise ValueError("Array dimension too large")

    bfmt = "<b"  # for single byte, "<b" and ">b" are the same
    blob = bytearray()

    # Version byte: high 4 bits = 0001 (v1), low 4 bits = extra mode
    # 0000 = may have extra, 0001 = must have extra
    if extra is not None:
        version_byte = VERSION_V1_MUST_EXTRA
    else:
        version_byte = VERSION_V1
    blob += pack(bfmt, version_byte)

    blob += pack(bfmt, dtype_flag)
    blob += pack(bfmt, -shape_size if big_shape else shape_size)
    blob += np.array(shape, dtype=shape_dtype).tobytes()
    blob += arr.tobytes()

    if extra is not None:
        if isinstance(extra, dict):
            extra = json_dumpb(extra)
            extra_flag = _EXTRA_ENCODING_FLAGS["json"]
        elif isinstance(extra, bytes):
            extra_flag = (
                _EXTRA_ENCODING_FLAGS["msgpack"]
                if isinstance(extra, MsgpackBytes)
                else _EXTRA_ENCODING_FLAGS["bytes"]
            )
        else:
            raise TypeError(
                f"Argument `extra` must be bytes, got {type(extra)} instead"
            )
        blob += pack(bfmt, extra_flag)
        blob += pack(f"{target_endian}L", len(extra))
        blob += extra

    return bytes(blob)


@t.overload
def encode(
    mat: np.ndarray,
    extra: t.Any = None,
    *more_mat_extra_pairs: t.Union[np.ndarray, t.Any],
    endian: EndianType = "=",
) -> bytes: ...


@t.overload
def encode(
    mat_extra_pair: MatExtraPair,
    *more_mat_extra_pairs: MatExtraPair,
    endian: EndianType = "=",
) -> bytes: ...


def encode(*args, endian: EndianType = "=", **kwargs) -> bytes:
    mat_extra_pairs: t.List[MatExtraPair]
    len_args = len(args)
    if len_args == 0:
        if "mat" not in kwargs or "extra" not in kwargs:
            raise ValueError(
                "If no positional arguments are given, "
                "`mat` and `extra` must be given as keyword arguments"
            )
        elif len(kwargs) > 2:
            for k in kwargs:
                if k in ("mat", "extra"):
                    continue
                raise KeyError(f"Invalid keyword argument: {k}")
        mat_extra_pairs = [(kwargs["mat"], kwargs["extra"])]
    else:
        if "mat" in kwargs:
            raise ValueError(
                "If one positional argument is given, argument `mat` is not allowed"
            )
        if isinstance(args[0], tuple):
            if kwargs:
                raise ValueError(
                    "If tuple positional argument is given, "
                    "no keyword arguments are allowed"
                )
            mat_extra_pairs = [args[0]]  # type: ignore
            for pair in args[1:]:
                if not isinstance(pair, tuple):
                    raise TypeError(f"Expected tuple, got {type(pair)} instead")
                mat_extra_pairs.append(pair)  # type: ignore
        elif isinstance(args[0], np.ndarray):
            if len_args == 1:
                mat_extra_pairs = [(args[0], kwargs.get("extra"))]
            else:
                if "extra" in kwargs:
                    raise ValueError(
                        "Argument `extra` must not be given if more than one "
                        "positional argument is given"
                    )
                mat_extra_pairs = []
                for i in range(0, len_args, 2):
                    if not isinstance(args[i], np.ndarray):
                        raise TypeError(
                            f"Expected ndarray, got {type(args[i])} instead"
                        )
                    ei = i + 1
                    extra = args[ei] if ei < len_args else None
                    mat_extra_pairs.append((args[i], extra))
        else:
            raise TypeError(f"Expected ndarray or tuple, got {type(args[0])} instead")

    return b"\x00".join([_encode(*pair, endian=endian) for pair in mat_extra_pairs])


def decode(blob: bytes) -> t.List[MatExtraPair]:
    """Parse numpy array blob with version header"""
    result: t.List[t.Tuple[np.ndarray, t.Any]] = []
    bfmt = "<b"  # for single byte, "<b" and ">b" are the same
    while blob:
        # Parse version byte
        version_byte = unpack(bfmt, blob[0:1])[0]
        version = (version_byte >> 4) & 0x0F
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        extra_mode = version_byte & 0x0F  # 0 = may have, 1 = must have

        i0 = unpack(bfmt, blob[1:2])[0]
        if i0 > 0:
            endian = "<"
            dtype_flag = i0
        elif i0 < 0:
            endian = ">"
            dtype_flag = -i0
        else:
            raise ValueError("Invalid dtype flag")
        dtype = np.dtype(f"{endian}{_DTYPE_FLAGS_DECODING[dtype_flag]}")

        i1 = unpack(bfmt, blob[2:3])[0]
        if i1 > 0:
            shape_dtype = f"{endian}u2"
            shape_size = i1
            d0 = 3 + (shape_size * 2)
        elif i1 < 0:
            shape_dtype = f"{endian}u4"
            shape_size = -i1
            d0 = 3 + (shape_size * 4)
        else:
            raise ValueError("Invalid shape flag")

        shape = np.frombuffer(blob[3:d0], dtype=shape_dtype)
        e0 = d0 + dtype.alignment * np.prod(shape, dtype=int)
        arr = np.frombuffer(blob[d0:e0], dtype=dtype).reshape(shape)

        if e0 >= len(blob):
            # No more data
            if extra_mode == 0x01:
                raise ValueError("Expected extra data but found none")
            result.append((arr, None))
            break

        # Check if next byte is extra flag or new array
        next_byte = unpack(bfmt, blob[e0 : e0 + 1])[0]
        next_byte_high = (next_byte >> 4) & 0x0F

        if next_byte_high == 0x00:
            # This is an extra flag (format: 0000 ffff)
            extra_flag = next_byte
            e1 = e0 + 1
            extra_encoding = _EXTRA_DECODING_FLAGS.get(extra_flag)
            if extra_encoding is None:
                raise ValueError(f"Unknown extra flag: {extra_flag}")
            e2 = e1 + 4
            elen = unpack(f"{endian}L", blob[e1:e2])[0]
            e3 = e2 + elen
            if extra_encoding == "bytes":
                extra = blob[e2:e3]
            elif extra_encoding == "json":
                extra = json_loadb(blob[e2:e3])
            elif extra_encoding == "msgpack":
                raise NotImplementedError("msgpack not implemented")

            result.append((arr, extra))

            # Check for separator (null byte)
            if e3 < len(blob) and blob[e3] == 0:
                blob = blob[e3 + 1 :]
                continue
            break
        else:
            # No extra data, next byte starts a new array
            if extra_mode == 0x01:
                raise ValueError("Expected extra data but found new array")
            result.append((arr, None))
            # next_byte should be a version byte (0x10 or 0x11)
            if next_byte_high != 0x01:
                raise ValueError(f"Expected version byte, got: {next_byte}")
            blob = blob[e0:]
            continue

    return result

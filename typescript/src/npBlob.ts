import { castFloat16 } from "./conversion";
import { MatData, MatAndExtra, MatDataType } from "./interfaces";

const getSystemEndianness = (): "<" | ">" => {
  const arr0 = new Uint16Array([0xff00]);
  const arr1 = new Uint8Array(arr0.buffer);
  return arr1[0] === 0x00 ? "<" : ">";
};
const __SYS_ENDIAN = getSystemEndianness();

/**
 * Parses an ArrayBuffer containing 16-bit floating point numbers and returns a
 * Float32Array.
 * @param buffer - The ArrayBuffer containing 16-bit floating point numbers.
 * @returns A Float32Array containing the parsed floating point numbers.
 */
export const parseFloat16ArrayBuffer = (
  buffer: ArrayBuffer,
  littleEndian?: boolean
): Float32Array => {
  if (buffer.byteLength % 2 !== 0) {
    throw new Error();
  } else if (
    littleEndian === undefined ||
    (littleEndian === true && __SYS_ENDIAN === "<")
  ) {
    const raw = new Uint16Array(buffer);
    const arr = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; ++i) {
      arr[i] = castFloat16(raw[i]);
    }
    return arr;
  } else {
    const raw = new Uint16Array(buffer);
    const arr = new Float32Array(raw.length);
    const view = new DataView(buffer);
    const getUint16 = view.getUint16;
    for (let i = 0; i < raw.length; ++i) {
      arr[i] = castFloat16(getUint16(i * 2, littleEndian));
    }
    return arr;
  }
};

const __DTYPE_FLAGS_DECODING: { [key: number]: [object, number, string] } = {
  1: [Uint8Array, 1, "uint8"],
  2: [Uint16Array, 2, "uint16"],
  3: [Uint32Array, 4, "uint32"],
  // 4: [BigUint64Array, 8, "uint64"],  // not supported
  5: [Int8Array, 1, "int8"],
  6: [Int16Array, 2, "int16"],
  7: [Int32Array, 4, "int32"],
  // 8: [BigInt64Array, 8, "int64"],  // not supported
  9: [parseFloat16ArrayBuffer, 2, "float32"], // mock 16-bit float
  10: [Float32Array, 4, "float32"],
  11: [Float64Array, 8, "float64"],
};
const __DTYPE_FLAGS_ENCODING: { [key: string]: number } = {
  uint8: 1,
  uint16: 2,
  uint32: 3,
  // uint64: 4,  // not supported
  int8: 5,
  int16: 6,
  int32: 7,
  // int64: 8,  // not supported
  float16: 9, // mock 16-bit float
  float32: 10,
  float64: 11,
};

/**
 * Decodes a MatBlob buffer into an array of Mat objects and their corresponding extra
 * data.
 * @param buffer - The binary buffer to decode.
 * @returns An array of tuples, where each tuple contains a Mat object and its
 * corresponding extra data (if any).
 * @throws An error if the binary buffer is invalid or cannot be decoded.
 */
export const decodeMatBlob = (buffer: Uint8Array): Array<MatAndExtra> => {
  const result: Array<MatAndExtra> = [];
  while (buffer.length > 0) {
    const [i0, i1] = buffer.slice(0, 2);
    let endian, littleEndian;
    let dType;
    if (i0 > 0) {
      endian = "<";
      littleEndian = true;
      dType = i0;
    } else if (i0 < 0) {
      endian = ">";
      littleEndian = false;
      dType = -i0;
    } else {
      throw new Error();
    }
    if (dType === 0 || dType >= 12) {
      throw new Error();
    }

    const [ArrayType, dSize, type] = __DTYPE_FLAGS_DECODING[dType];
    const sameEndian = endian === __SYS_ENDIAN;
    let shape;
    let d0;
    let e0;
    if (i1 > 0) {
      d0 = 2 + i1 * 2;
      if (sameEndian) {
        shape = new Uint16Array(buffer.slice(2, d0).buffer);
      } else {
        shape = new Uint16Array(i1);
        const view = new DataView(buffer.slice(2, d0).buffer);
        const getUint16 = view.getUint16;
        for (let i = 0; i < i1; ++i) {
          shape[i] = getUint16(i * 2, littleEndian);
        }
      }
      e0 = d0 + shape.reduce((a, b) => a * b, 1) * dSize;
    } else {
      const s = -i1;
      d0 = 2 + s * 4;
      if (sameEndian) {
        shape = new Uint32Array(buffer.slice(2, d0).buffer);
      } else {
        shape = new Uint32Array(s);
        const view = new DataView(buffer.slice(2, d0).buffer);
        const getUint32 = view.getUint32;
        for (let i = 0; i < s; ++i) {
          shape[i] = getUint32(i * 4, littleEndian);
        }
      }
      e0 = d0 + shape.reduce((a, b) => a * b, 1) * dSize;
    }

    const arr =
      typeof ArrayType === "function"
        ? (ArrayType as { (buffer: ArrayBuffer): number[] })(
            buffer.slice(d0, e0).buffer
          )
        : new (ArrayType as { new (buffer: ArrayBuffer): number[] })(
            buffer.slice(d0, e0).buffer
          );

    if (e0 >= buffer.length) {
      result.push([
        {
          shape: Array.from(shape),
          type: type as MatDataType,
          data: arr as unknown as MatData,
        },
        null,
      ]);
      break;
    }

    const e1 = e0 + 1;
    const eFlag = buffer[e0];
    if (eFlag === 0) {
      result.push([
        {
          shape: Array.from(shape),
          type: type as MatDataType,
          data: arr as unknown as MatData,
        },
        null,
      ]);
      buffer = buffer.slice(e1);
      continue;
    }

    const e2 = e1 + 4;
    const eLen = new DataView(buffer.slice(e1, e2).buffer).getUint32(
      0,
      littleEndian
    );
    const e3 = e2 + eLen;
    let extra: object;
    switch (eFlag) {
      case 1:
        extra = buffer.slice(e2, e3).buffer;
        break;
      case 2:
        extra = JSON.parse(new TextDecoder().decode(buffer.slice(e2, e3)));
        break;
      case 3: // placeholder for msgpack
      default:
        throw new Error();
    }

    result.push([
      {
        shape: Array.from(shape),
        type: type as MatDataType,
        data: arr as unknown as MatData,
      },
      extra,
    ]);

    if (e3 < buffer.length && buffer[e3] === 0) {
      buffer = buffer.slice(e3 + 1);
      continue;
    }
    break;
  }
  return result;
};

/**
 * Encodes an array of Mats and their associated metadata into a binary blob.
 * @param mats - An array of tuples, where each tuple contains a Mat and an optional
 * object of extra metadata.
 * @returns A Uint8Array containing the encoded binary blob.
 * @throws An error if the number of array dimensions is too large or if an array
 * dimension is too large.
 */
export function encodeMatBlob(
  mats: Array<MatAndExtra>,
  littleEndian?: boolean
): Uint8Array {
  const textEncoder = new TextEncoder();
  let blobLength = 0;
  let sameEndian = false;
  if (littleEndian === undefined) {
    sameEndian = true;
    littleEndian = __SYS_ENDIAN === "<";
  } else if (littleEndian === (__SYS_ENDIAN === "<")) {
    sameEndian = true;
  }
  const blobComponents: Array<
    [
      number,
      number,
      Uint16Array | Uint32Array,
      MatData,
      number,
      number,
      Uint8Array | null,
    ]
  > = [];

  for (const [mat, extra] of mats) {
    const { shape, data } = mat;

    const dType = __DTYPE_FLAGS_ENCODING[mat.type];
    if (!dType) {
      throw new Error();
    }

    const i0 = littleEndian ? dType : -dType;
    const i1 = shape.length;
    if (i1 > 128) {
      throw new Error();
    }
    blobLength += 2;

    let bigShape: boolean;
    let shapeArray: Uint16Array | Uint32Array;
    const maxDim = Math.max(...shape);
    if (maxDim < 65536) {
      shapeArray = new Uint16Array(
        shape instanceof Array ? shape : Array.from(shape)
      );
      bigShape = false;
    } else if (maxDim < 4294967296) {
      shapeArray = new Uint32Array(
        shape instanceof Array ? shape : Array.from(shape)
      );
      bigShape = true;
    } else {
      throw new Error();
    }
    blobLength += shapeArray.byteLength;
    blobLength += data.byteLength;

    if (extra) {
      let extraArray: Uint8Array;
      let extraEncoding: number;
      if (extra instanceof ArrayBuffer) {
        extraArray = new Uint8Array(extra);
        extraEncoding = 1;
      } else if (extra instanceof Uint8Array) {
        extraArray = extra;
        extraEncoding = 1;
      } else {
        const extraString = JSON.stringify(extra);
        extraArray = textEncoder.encode(extraString);
        extraEncoding = 2;
      }
      blobLength += 5 + extraArray.byteLength;
      blobComponents.push([
        i0,
        bigShape ? -i1 : i1,
        shapeArray,
        data,
        extraEncoding,
        extraArray ? extraArray.byteLength : 0,
        extraArray,
      ]);
    } else {
      blobComponents.push([
        i0,
        bigShape ? -i1 : i1,
        shapeArray,
        data,
        0,
        0,
        null,
      ]);
    }
  }

  const numBlobComponents = blobComponents.length;
  const blob = new Uint8Array(blobLength + numBlobComponents - 1);
  let offset = 0;
  for (let i = 0; i < numBlobComponents; ++i) {
    const [i0, i1, shapeArray, data, extraEncoding, extraLength, extraArray] =
      blobComponents[i];

    if (i > 0) {
      blob[offset] = 0;
      offset += 1;
    }

    blob[offset] = i0;
    blob[offset + 1] = i1;
    offset += 2;

    if (sameEndian) {
      blob.set(new Uint8Array(shapeArray.buffer), offset);
      offset += shapeArray.byteLength;

      blob.set(new Uint8Array(data.buffer), offset);
      offset += data.byteLength;
    } else {
      const shapeArrayBuffer = new Uint8Array(shapeArray.buffer);
      const shapeStep = shapeArray.BYTES_PER_ELEMENT;
      const shapeArrayLength = shapeArray.length;
      for (let i = 0; i < shapeArrayLength; ++i) {
        for (let j = 0; j < shapeStep; ++j) {
          blob[offset + j] = shapeArrayBuffer[offset + shapeStep - j - 1];
        }
        offset += shapeStep;
      }

      const dataBuffer = new Uint8Array(data.buffer);
      const dataStep = data.BYTES_PER_ELEMENT;
      const dataLength = data.length;
      for (let i = 0; i < dataLength; ++i) {
        for (let j = 0; j < dataStep; ++j) {
          blob[offset + j] = dataBuffer[offset + dataStep - j - 1];
        }
        offset += dataStep;
      }
    }

    if (extraEncoding > 0) {
      if (!extraArray) {
        throw new Error();
      }

      blob[offset] = extraEncoding;
      offset += 1;
      const extraLengthArray = new DataView(
        blob.buffer,
        blob.byteOffset + offset,
        4
      );
      extraLengthArray.setUint32(0, extraLength, littleEndian);
      offset += 4;
      blob.set(extraArray, offset);
      offset += extraArray.byteLength;
    }
  }

  return blob;
}

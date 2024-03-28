/**
 * A map of data types to their corresponding typed arrays.
 */
interface DTypeMap {
  uint8: Uint8Array;
  uint16: Uint16Array;
  uint32: Uint32Array;
  // uint64: BigUint64Array;
  int8: Int8Array;
  int16: Int16Array;
  int32: Int32Array;
  // int64: BigInt64Array;
  float16: Float32Array;
  float32: Float32Array;
  float64: Float64Array;
}

/**
 * A union type of all the keys in the `DTypeMap`.
 */
type DType = keyof DTypeMap;

/**
 * A type alias for the data type of a `NDArrayData`.
 */
type NDArrayData = DTypeMap[DType];

/**
 * An interface representing a NDArray.
 */
interface NDArray {
  /** The shape of the array. */
  readonly shape: Array<number>;
  /** The data type of the array. */
  readonly dType: DType;
  /** The data of the array. */
  readonly data: NDArrayData;
}

/**
 * A tuple representing a `NDArray` object and an optional extra object.
 */
type NDArrayAndExtra = [NDArray, object | null];

/**
 * Converts a 16-bit integer to a floating point number.
 *
 * @param int16 The 16-bit integer to convert.
 * @returns The converted floating point number.
 */
const castFloat16 = (int16: number): number => {
  const exponent = (int16 & 0x7c00) >> 10,
    fraction = int16 & 0x03ff;
  return (
    (int16 >> 15 ? -1 : 1) *
    (exponent
      ? exponent === 0x1f
        ? fraction
          ? NaN
          : Infinity
        : Math.pow(2, exponent - 15) * (1 + fraction / 0x400)
      : 6.103515625e-5 * (fraction / 0x400))
  );
};

/**
 * Retrieves the system endianness by creating an array buffer and checking the byte
 * order.
 *
 * @return { "<" | ">" } The system endianness ("<" for little endian, ">" for big
 * endian)
 */
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
  littleEndian?: boolean,
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

const TypedArray = Object.getPrototypeOf(Uint8Array);
const __DTYPE_FLAGS_DECODING: Record<number, [object, number, string]> = {
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
const __DTYPE_FLAGS_ENCODING: Record<string, number> = {
  uint8: 1,
  uint16: 2,
  uint32: 3,
  // uint64: 4,
  int8: 5,
  int16: 6,
  int32: 7,
  // int64: 8,
  float16: 9,
  float32: 10,
  float64: 11,
};

class IncompleteChunksError extends Error {}

/**
 * Splits an array of Uint8Array chunks into a single Uint8Array based on a specified
 * length. Note that parameter chunks is splitted and modified in-place.
 *
 * @param {Array<Uint8Array>} chunks - The array of Uint8Array chunks to split.
 * @param {number} length - The length at which to split the chunks.
 * @return {Uint8Array} The concatenated Uint8Array based on the length.
 */
const splitChunks = (chunks: Array<Uint8Array>, length: number): Uint8Array => {
  const poppedChunks: Array<Uint8Array> = [];
  const chunksSize = chunks.length;
  let accumelatedLength = 0;
  let splitChunksAt = 0;
  for (let i = 0; i < chunksSize; ++i) {
    const chunk = chunks[i];
    const chunkLength = chunk.length;
    if (accumelatedLength + chunkLength <= length) {
      poppedChunks.push(chunk);
      accumelatedLength += chunkLength;
      splitChunksAt = i + 1;
    } else {
      const splitChunkAt = length - accumelatedLength;
      if (splitChunkAt > 0) {
        poppedChunks.push(chunk.slice(0, splitChunkAt));
        chunks[i] = chunk.slice(splitChunkAt);
        accumelatedLength = length;
      }
      break;
    }
  }
  if (accumelatedLength < length) {
    throw new IncompleteChunksError("Unexpected end of stream");
  }
  if (splitChunksAt > 0) {
    chunks.splice(0, splitChunksAt);
  }

  const result = new Uint8Array(length);
  const poppedChunksSize = poppedChunks.length;
  let offset = 0;
  for (let i = 0; i < poppedChunksSize; ++i) {
    const chunk = poppedChunks[i];
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
};

/**
 * Parses an array of Uint8Array chunks to extract `NDArrayAndExtra` objects.
 *
 * @param {Array<Uint8Array>} chunks - The array of Uint8Array chunks to be parsed.
 * @return {Array<NDArrayAndExtra>} The parsed array of `NDArrayAndExtra` objects.
 * @throws An error if the chunks cannot be parsed correctly.
 */
const parseFromUint8ArrayChunks = (
  chunks: Array<Uint8Array>,
): Array<NDArrayAndExtra> => {
  const result: Array<NDArrayAndExtra> = [];
  if (chunks.length === 0) {
    return result;
  }

  const tempChunks: Array<Uint8Array> = [];
  let chunk: Uint8Array;
  try {
    while (chunks.length > 0) {
      chunk = splitChunks(chunks, 2);
      tempChunks.push(chunk);
      const [i0, i1] = chunk;
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

      const [ArrayType, dSize, typ] = __DTYPE_FLAGS_DECODING[dType];
      const sameEndian = endian === __SYS_ENDIAN;
      let shape;
      let d0;
      let e0;
      if (i1 > 0) {
        d0 = i1 * 2;
        chunk = splitChunks(chunks, d0);
        tempChunks.push(chunk);
        if (sameEndian) {
          shape = new Uint16Array(chunk.buffer);
        } else {
          shape = new Uint16Array(i1);
          const view = new DataView(chunk.buffer);
          const getUint16 = view.getUint16;
          for (let i = 0; i < i1; ++i) {
            shape[i] = getUint16(i * 2, littleEndian);
          }
        }
        e0 = shape.reduce((a, b) => a * b, 1) * dSize;
      } else {
        const s = -i1;
        d0 = s * 4;
        chunk = splitChunks(chunks, d0);
        tempChunks.push(chunk);
        if (sameEndian) {
          shape = new Uint32Array(chunk.buffer);
        } else {
          shape = new Uint32Array(s);
          const view = new DataView(chunk.buffer);
          const getUint32 = view.getUint32;
          for (let i = 0; i < s; ++i) {
            shape[i] = getUint32(i * 4, littleEndian);
          }
        }
        e0 = shape.reduce((a, b) => a * b, 1) * dSize;
      }

      chunk = splitChunks(chunks, e0);
      tempChunks.push(chunk);
      const arr =
        Object.getPrototypeOf(ArrayType) === TypedArray
          ? new (ArrayType as { new (buffer: ArrayBuffer): number[] })(
              chunk.buffer,
            )
          : (ArrayType as { (buffer: ArrayBuffer): number[] })(chunk.buffer);

      if (chunks.length === 0) {
        result.push([
          {
            shape: Array.from(shape),
            dType: typ as DType,
            data: arr as unknown as NDArrayData,
          },
          null,
        ]);
        break;
      }

      chunk = splitChunks(chunks, 1);
      tempChunks.push(chunk);
      const eFlag = chunk[0];
      if (eFlag === 0) {
        result.push([
          {
            shape: Array.from(shape),
            dType: typ as DType,
            data: arr as unknown as NDArrayData,
          },
          null,
        ]);
        continue;
      }

      chunk = splitChunks(chunks, 4);
      tempChunks.push(chunk);

      const eLen = new DataView(chunk.buffer).getUint32(0, littleEndian);
      chunk = splitChunks(chunks, eLen);
      tempChunks.push(chunk);
      let extra: object;
      switch (eFlag) {
        case 1:
          extra = chunk.buffer;
          break;
        case 2:
          extra = JSON.parse(new TextDecoder().decode(chunk));
          break;
        case 3: // placeholder for msgpack
        default:
          throw new Error();
      }

      result.push([
        {
          shape: Array.from(shape),
          dType: typ as DType,
          data: arr as unknown as NDArrayData,
        },
        extra,
      ]);
      tempChunks.length = 0;

      if (chunks.length > 0 && chunks[0][0] === 0) {
        splitChunks(chunks, 1);
        continue;
      } else {
        break;
      }
    }
  } catch (e) {
    if (e instanceof IncompleteChunksError) {
      chunks.unshift(...tempChunks);
    } else {
      throw e;
    }
  }
  return result;
};

/**
 * Generates a ReadableStream of NDArrayAndExtra objects from a given ReadableStream of
 * Uint8Array chunks.
 *
 * @param {ReadableStream<Uint8Array>} stream - The input ReadableStream of Uint8Array
 * chunks.
 * @return {ReadableStream<NDArrayAndExtra>} A ReadableStream of NDArrayAndExtra objects.
 */
const streamFromReadableStream = (
  stream: ReadableStream<Uint8Array>,
): ReadableStream<NDArrayAndExtra> => {
  const chunks: Array<Uint8Array> = [];
  const nes: Array<NDArrayAndExtra> = [];
  const streamReader = stream.getReader();
  let streamDone = false;

  return new ReadableStream({
    pull(controller) {
      const yieldArrays = async () => {
        if (streamDone && nes.length === 0) {
          controller.close();
          return;
        }
        if (nes.length > 0) {
          while (nes.length > 0) {
            controller.enqueue(nes.shift()!);
          }
          return;
        }
        if (chunks.length > 0) {
          for (const ne of parseFromUint8ArrayChunks(chunks)) {
            nes.push(ne);
          }
          if (nes.length > 0) {
            while (nes.length > 0) {
              controller.enqueue(nes.shift()!);
            }
            return;
          }
        }
        if (!streamDone) {
          while (nes.length === 0) {
            const part = await streamReader.read();
            if (part.done) {
              streamDone = true;
              break;
            }
            const chunk = part.value;
            if (chunk === undefined) {
              throw new Error("Unexpected end of stream");
            } else if (!(chunk instanceof Uint8Array)) {
              throw new TypeError(`Expected Uint8Array, got ${chunk}`);
            }
            chunks.push(chunk);
            for (const ne of parseFromUint8ArrayChunks(chunks)) {
              nes.push(ne);
            }
          }
          if (nes.length > 0) {
            while (nes.length > 0) {
              controller.enqueue(nes.shift()!);
            }
            return;
          }
        } else {
          for (const ne of parseFromUint8ArrayChunks(chunks)) {
            nes.push(ne);
          }
          if (nes.length > 0) {
            while (nes.length > 0) {
              controller.enqueue(nes.shift()!);
            }
            return;
          }
          if (nes.length === 0) {
            controller.close();
          }
          if (chunks.length > 0) {
            throw new Error("There are still chunks left");
          }
        }
      };

      return yieldArrays();
    },
  });
};

function stream(raw: ArrayBuffer): ReadableStream<NDArrayAndExtra>;
function stream(
  raw: ReadableStream<Uint8Array>,
): ReadableStream<NDArrayAndExtra>;
function stream(raw: Response): ReadableStream<NDArrayAndExtra>;
function stream(raw: Uint8Array): ReadableStream<NDArrayAndExtra>;
/**
 * Generates a ReadableStream<NDArrayAndExtra> based on the type of input provided.
 *
 * @param {ArrayBuffer | ReadableStream<Uint8Array> | Response | Uint8Array} raw - input
 * data to generate the stream from
 * @return {ReadableStream<NDArrayAndExtra>} the generated
 * ReadableStream<NDArrayAndExtra>
 */
function stream(
  raw: ArrayBuffer | ReadableStream<Uint8Array> | Response | Uint8Array,
): ReadableStream<NDArrayAndExtra> {
  if (raw instanceof Response) {
    if (!raw.body) {
      throw new Error("Response has no body");
    }
    return streamFromReadableStream(raw.body);
  } else if (raw instanceof ReadableStream) {
    return streamFromReadableStream(raw);
  } else if (raw instanceof Uint8Array) {
    return streamFromReadableStream(
      new ReadableStream({
        pull(controller) {
          controller.enqueue(raw);
          controller.close();
        },
      }),
    );
  } else if (raw instanceof ArrayBuffer) {
    return streamFromReadableStream(
      new ReadableStream({
        pull(controller) {
          controller.enqueue(new Uint8Array(raw));
          controller.close();
        },
      }),
    );
  } else {
    throw new TypeError(`Unexpected type of input: ${raw}`);
  }
}

/**
 * Decodes a npblob buffer into an array of NDArray objects and their corresponding
 * extra data.
 *
 * @param buffer - The binary buffer to decode.
 * @returns An array of tuples, where each tuple contains a `NDArray` object and its
 * corresponding extra data (if any).
 * @throws An error if the binary buffer is invalid or cannot be decoded.
 */
const decode = (buffer: Uint8Array): Array<NDArrayAndExtra> => {
  return parseFromUint8ArrayChunks([buffer]);
};

/**
 * Encodes an array of NDArray and their associated metadata into a binary blob.
 *
 * @param items - An array of tuples, where each tuple contains a `NDArray` and an
 * optional object of extra metadata.
 * @param littleEndian - Whether to encode the binary blob in little-endian format.
 * @returns A Uint8Array containing the encoded binary blob.
 * @throws An error if the number of array dimensions is too large or if an array
 * dimension is too large.
 */
function encode(
  items: Array<NDArrayAndExtra>,
  littleEndian?: boolean,
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
      NDArrayData,
      number,
      number,
      Uint8Array | null,
    ]
  > = [];

  for (const [arr, extra] of items) {
    const { shape, data } = arr;

    const dType = __DTYPE_FLAGS_ENCODING[arr.dType];
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
        shape instanceof Array ? shape : Array.from(shape),
      );
      bigShape = false;
    } else if (maxDim < 4294967296) {
      shapeArray = new Uint32Array(
        shape instanceof Array ? shape : Array.from(shape),
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
        4,
      );
      extraLengthArray.setUint32(0, extraLength, littleEndian);
      offset += 4;
      blob.set(extraArray, offset);
      offset += extraArray.byteLength;
    }
  }

  return blob;
}

export {
  DType,
  DTypeMap,
  NDArray,
  NDArrayAndExtra,
  NDArrayData,
  decode,
  encode,
  stream,
};

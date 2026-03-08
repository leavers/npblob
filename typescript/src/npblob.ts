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

// Version and format constants
const VERSION_V1 = 0x10; // 0001 0000 - version 1, may have extra
const VERSION_V1_MUST_EXTRA = 0x11; // 0001 0001 - version 1, must have extra

// Extra flag format: high 4 bits = 0000, low 4 bits = type
const EXTRA_FLAG_BYTES = 0x01; // 0000 0001
const EXTRA_FLAG_JSON = 0x02; // 0000 0010
const EXTRA_FLAG_MSGPACK = 0x03; // 0000 0011

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

const __EXTRA_DECODING_FLAGS: Record<number, string> = {
  [EXTRA_FLAG_BYTES]: "bytes",
  [EXTRA_FLAG_JSON]: "json",
  [EXTRA_FLAG_MSGPACK]: "msgpack",
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
  let accumulatedLength = 0;
  let splitChunksAt = 0;
  for (let i = 0; i < chunksSize; ++i) {
    const chunk = chunks[i];
    const chunkLength = chunk.length;
    if (accumulatedLength + chunkLength <= length) {
      poppedChunks.push(chunk);
      accumulatedLength += chunkLength;
      splitChunksAt = i + 1;
    } else {
      const splitChunkAt = length - accumulatedLength;
      if (splitChunkAt > 0) {
        poppedChunks.push(chunk.slice(0, splitChunkAt));
        chunks[i] = chunk.slice(splitChunkAt);
        accumulatedLength = length;
      }
      break;
    }
  }
  if (accumulatedLength < length) {
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
      // Read version byte + dtype flag + shape flag (3 bytes)
      chunk = splitChunks(chunks, 3);
      tempChunks.push(chunk);
      const [versionByte, dtypeFlag, shapeFlag] = chunk;
      
      // Parse version byte
      const version = (versionByte >> 4) & 0x0F;
      if (version !== 1) {
        throw new Error(`Unsupported version: ${version}`);
      }
      const extraMode = versionByte & 0x0F; // 0 = may have, 1 = must have
      
      // Parse dtype flag
      let endian: "<" | ">", littleEndian: boolean;
      let dType: number;
      if (dtypeFlag > 0) {
        endian = "<";
        littleEndian = true;
        dType = dtypeFlag;
      } else if (dtypeFlag < 0) {
        endian = ">";
        littleEndian = false;
        dType = -dtypeFlag;
      } else {
        throw new Error("Invalid dtype flag");
      }
      if (dType === 0 || dType > 12) {
        throw new Error(`Invalid dtype: ${dType}`);
      }

      const [ArrayType, dSize, typ] = __DTYPE_FLAGS_DECODING[dType];
      const sameEndian = endian === __SYS_ENDIAN;
      
      // Parse shape
      let shape: Uint16Array | Uint32Array;
      let dataSize: number;
      const shapeLen = shapeFlag > 0 ? shapeFlag : -shapeFlag;
      
      if (shapeFlag > 0) {
        // uint16 shape
        const shapeBytes = shapeLen * 2;
        chunk = splitChunks(chunks, shapeBytes);
        tempChunks.push(chunk);
        if (sameEndian) {
          shape = new Uint16Array(chunk.buffer, chunk.byteOffset, shapeLen);
        } else {
          shape = new Uint16Array(shapeLen);
          const view = new DataView(chunk.buffer, chunk.byteOffset);
          for (let i = 0; i < shapeLen; ++i) {
            shape[i] = view.getUint16(i * 2, littleEndian);
          }
        }
      } else {
        // uint32 shape
        const shapeBytes = shapeLen * 4;
        chunk = splitChunks(chunks, shapeBytes);
        tempChunks.push(chunk);
        if (sameEndian) {
          shape = new Uint32Array(chunk.buffer, chunk.byteOffset, shapeLen);
        } else {
          shape = new Uint32Array(shapeLen);
          const view = new DataView(chunk.buffer, chunk.byteOffset);
          for (let i = 0; i < shapeLen; ++i) {
            shape[i] = view.getUint32(i * 4, littleEndian);
          }
        }
      }
      
      dataSize = Array.from(shape).reduce((a, b) => a * b, 1) * dSize;

      // Read array data
      chunk = splitChunks(chunks, dataSize);
      tempChunks.push(chunk);
      const arr =
        Object.getPrototypeOf(ArrayType) === TypedArray
          ? new (ArrayType as { new (buffer: ArrayBufferLike): number[] })(
              chunk.buffer,
              chunk.byteOffset,
              dataSize / dSize,
            )
          : (ArrayType as { (buffer: ArrayBufferLike, littleEndian?: boolean): number[] })(
              chunk.slice().buffer,
              littleEndian,
            );

      // Check if we have more data for extra or next array
      if (chunks.length === 0) {
        // No more data - validate extraMode
        if (extraMode === 0x01) {
          throw new IncompleteChunksError("Expected extra data but stream ended");
        }
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

      // Peek at next byte to determine if it's extra flag or new array
      // We need to look without consuming yet
      const nextByte = chunks[0][0];
      const nextByteHigh = (nextByte >> 4) & 0x0F;
      
      if (nextByte === 0x00) {
        // This is a separator between arrays
        splitChunks(chunks, 1); // consume the separator
        result.push([
          {
            shape: Array.from(shape),
            dType: typ as DType,
            data: arr as unknown as NDArrayData,
          },
          null,
        ]);
        tempChunks.length = 0;
        continue;
      } else if (nextByteHigh === 0x00 && nextByte !== 0x00) {
        // This is an extra flag: 0000 ffff (where ffff != 0000)
        chunk = splitChunks(chunks, 1);
        tempChunks.push(chunk);
        const eFlag = chunk[0];
        
        const extraType = eFlag & 0x0F;
        const extraEncoding = __EXTRA_DECODING_FLAGS[extraType];
        if (!extraEncoding) {
          throw new Error(`Unknown extra flag: ${eFlag}`);
        }

        // Read extra length (4 bytes)
        chunk = splitChunks(chunks, 4);
        tempChunks.push(chunk);
        const eLen = new DataView(chunk.buffer, chunk.byteOffset).getUint32(0, littleEndian);
        
        // Read extra data
        chunk = splitChunks(chunks, eLen);
        tempChunks.push(chunk);
        
        let extra: object;
        switch (extraEncoding) {
          case "bytes":
            extra = chunk.slice().buffer;
            break;
          case "json":
            extra = JSON.parse(new TextDecoder().decode(chunk));
            break;
          case "msgpack":
            throw new Error("msgpack not implemented");
          default:
            throw new Error(`Unknown extra encoding: ${extraEncoding}`);
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

        // Check for separator (null byte) indicating more arrays
        if (chunks.length > 0 && chunks[0][0] === 0) {
          splitChunks(chunks, 1);
          continue;
        } else {
          break;
        }
      } else if (nextByteHigh === 0x01) {
        // Next byte is a version byte (new array starts)
        if (extraMode === 0x01) {
          throw new Error("Expected extra data but found new array");
        }
        result.push([
          {
            shape: Array.from(shape),
            dType: typ as DType,
            data: arr as unknown as NDArrayData,
          },
          null,
        ]);
        tempChunks.length = 0;
        // Don't consume the version byte - it will be processed in next iteration
        continue;
      } else {
        // Unknown byte - could be incomplete data or corruption
        if (extraMode === 0x01) {
          throw new IncompleteChunksError("Expected extra data");
        }
        // For extraMode === 0, we might have incomplete data
        // Push back temp chunks and wait for more data
        throw new IncompleteChunksError("Incomplete data or unknown format");
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
          // Stream ended but no arrays parsed - try one more time to parse remaining chunks
          if (streamDone) {
            for (const ne of parseFromUint8ArrayChunks(chunks)) {
              nes.push(ne);
            }
            if (nes.length > 0) {
              while (nes.length > 0) {
                controller.enqueue(nes.shift()!);
              }
              return;
            }
            // No more arrays and stream ended - close
            controller.close();
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
      number, // version byte
      number, // dtype flag
      number, // shape flag
      Uint16Array | Uint32Array, // shape array
      NDArrayData, // data
      number, // extra encoding (0 = none)
      number, // extra length
      Uint8Array | null, // extra array
    ]
  > = [];

  for (const [arr, extra] of items) {
    const { shape, data } = arr;

    const dType = __DTYPE_FLAGS_ENCODING[arr.dType];
    if (!dType) {
      throw new Error();
    }

    // Version byte: 0001 vvvv where vvvv is extra mode
    // 0000 = may have extra, 0001 = must have extra
    const versionByte = extra ? VERSION_V1_MUST_EXTRA : VERSION_V1;
    
    const dtypeFlag = littleEndian ? dType : -dType;
    const shapeLen = shape.length;
    if (shapeLen > 128) {
      throw new Error();
    }
    blobLength += 3; // version + dtype + shape_len

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
        extraEncoding = EXTRA_FLAG_BYTES;
      } else if (extra instanceof Uint8Array) {
        extraArray = extra;
        extraEncoding = EXTRA_FLAG_BYTES;
      } else {
        const extraString = JSON.stringify(extra);
        extraArray = textEncoder.encode(extraString);
        extraEncoding = EXTRA_FLAG_JSON;
      }
      blobLength += 5 + extraArray.byteLength; // flag(1) + len(4) + data
      blobComponents.push([
        versionByte,
        dtypeFlag,
        bigShape ? -shapeLen : shapeLen,
        shapeArray,
        data,
        extraEncoding,
        extraArray.byteLength,
        extraArray,
      ]);
    } else {
      blobComponents.push([
        versionByte,
        dtypeFlag,
        bigShape ? -shapeLen : shapeLen,
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
    const [versionByte, dtypeFlag, shapeFlag, shapeArray, data, extraEncoding, extraLength, extraArray] =
      blobComponents[i];

    if (i > 0) {
      blob[offset] = 0; // separator
      offset += 1;
    }

    // Write version byte
    blob[offset] = versionByte;
    offset += 1;
    
    // Write dtype flag
    blob[offset] = dtypeFlag;
    offset += 1;
    
    // Write shape flag
    blob[offset] = shapeFlag;
    offset += 1;

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

      // Extra flag: high 4 bits = 0000, low 4 bits = type
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

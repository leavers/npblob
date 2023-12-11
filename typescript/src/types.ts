/**
 * A map of data types to their corresponding typed arrays.
 */
interface DTypeMap {
  uint8: Uint8Array;
  uint16: Uint16Array;
  uint32: Uint32Array;
  // uint64: BigUint64Array;  // not supported
  int8: Int8Array;
  int16: Int16Array;
  int32: Int32Array;
  // int64: BigInt64Array;  // not supported
  float16: Float32Array;
  float32: Float32Array;
  float64: Float64Array;
}

/**
 * A union type of all the keys in the `DTypeMap`.
 */
type NDArrayDataType = keyof DTypeMap;

/**
 * A type alias for the data type of a `NDArray`.
 */
type NDArrayData = DTypeMap[NDArrayDataType];

/**
 * An interface representing a matrix of data.
 */
interface NDArray {
  /** The shape of the matrix. */
  readonly shape: number[];
  /** The data type of the matrix. */
  readonly type: NDArrayDataType;
  /** The data of the matrix. */
  readonly data: NDArrayData;
}

/**
 * A tuple representing a `NDArray` object and an optional extra object.
 */
type NDArrayAndExtra = [NDArray, object | null];

export { NDArray, NDArrayAndExtra, NDArrayData, NDArrayDataType, DTypeMap };

/**
 * Converts a 16-bit integer to a floating point number.
 * @param int16 The 16-bit integer to convert.
 * @returns The converted floating point number.
 */
export const castFloat16 = (int16: number): number => {
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

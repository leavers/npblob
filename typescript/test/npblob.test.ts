import { describe, expect, it } from "vitest";
import { NDArrayAndExtra, decode, encode, stream } from "../src/npblob";
import { ReadableStream } from "stream/web";

/**
 * A Numpy array bytes which represents the following array:
 * array([[1.1, 2.2, 3.3],
 *        [4.4, 5.5, 6.6]], dtype=float32)
 */
const _hex = "\xcd\xcc\x8c?\xcd\xcc\x0c@33S@\xcd\xcc\x8c@\x00\x00\xb0@33\xd3@";
const hex = new Uint8Array(new ArrayBuffer(_hex.length));
for (let i = 0; i < _hex.length; ++i) {
  hex[i] = _hex.charCodeAt(i);
}
/* its npblob */
const _arrHex =
  "\n\x02\x02\x00\x03\x00\xcd\xcc\x8c?\xcd\xcc\x0c@33S@\xcd\xcc\x8c@\x00\x00\xb0@33\xd3@";
const arrHex = new Uint8Array(new ArrayBuffer(_arrHex.length));
for (let i = 0; i < _arrHex.length; ++i) {
  arrHex[i] = _arrHex.charCodeAt(i);
}
/* with extra data: {"min": 1.1, "max": 6.6} */
const _matAndExtraHex =
  '\n\x02\x02\x00\x03\x00\xcd\xcc\x8c?\xcd\xcc\x0c@33S@\xcd\xcc\x8c@\x00\x00\xb0@33\xd3@\x02\x15\x00\x00\x00{"min":1.1,"max":6.6}';
const matAndExtraHex = new Uint8Array(new ArrayBuffer(_matAndExtraHex.length));
for (let i = 0; i < _matAndExtraHex.length; ++i) {
  matAndExtraHex[i] = _matAndExtraHex.charCodeAt(i);
}
/* multiple mat blobs */
const _multipleMatAndExtraHex = [
  _arrHex,
  _matAndExtraHex,
  _arrHex,
  _matAndExtraHex,
  _arrHex,
  _matAndExtraHex,
  _arrHex,
  _matAndExtraHex,
  _arrHex,
  _matAndExtraHex,
].join("\x00");
const multipleMatWithExtraHex = new Uint8Array(
  new ArrayBuffer(_multipleMatAndExtraHex.length),
);
for (let i = 0; i < _multipleMatAndExtraHex.length; ++i) {
  multipleMatWithExtraHex[i] = _multipleMatAndExtraHex.charCodeAt(i);
}

describe("matBlob", () => {
  const mat: NDArrayAndExtra = [
    {
      shape: [2, 3],
      dType: "float32",
      data: new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]),
    },
    null,
  ];
  const matAndExtra: NDArrayAndExtra = [
    {
      shape: [2, 3],
      dType: "float32",
      data: new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]),
    },
    {
      min: 1.1,
      max: 6.6,
    },
  ];

  it("decode", () => {
    expect(decode(arrHex)).toEqual([mat]);
    expect(decode(matAndExtraHex)).toEqual([matAndExtra]);

    const mats = decode(multipleMatWithExtraHex);
    expect(mats.length).toEqual(10);
    expect(mats[0]).toEqual(mat);
    expect(mats[1]).toEqual(matAndExtra);
    expect(mats[8]).toEqual(mat);
    expect(mats[9]).toEqual(matAndExtra);
  });

  it("encode", () => {
    expect(encode([mat])).toEqual(arrHex);
    expect(encode([matAndExtra])).toEqual(matAndExtraHex);
    expect(
      encode([
        mat,
        matAndExtra,
        mat,
        matAndExtra,
        mat,
        matAndExtra,
        mat,
        matAndExtra,
        mat,
        matAndExtra,
      ]),
    ).toEqual(multipleMatWithExtraHex);
  });

  it("stream", async () => {
    expect(await stream(arrHex).getReader().read()).toEqual({
      done: false,
      value: mat,
    });

    expect(await stream(matAndExtraHex).getReader().read()).toEqual({
      done: false,
      value: matAndExtra,
    });

    let reader = stream(multipleMatWithExtraHex).getReader();
    for (let i = 0; i < 10; ++i) {
      if (i % 2 === 0) {
        expect(await reader.read()).toEqual({
          done: false,
          value: mat,
        });
      } else {
        expect(await reader.read()).toEqual({
          done: false,
          value: matAndExtra,
        });
      }
    }

    const createMockResponse = () => {
      let count = 0;
      const stream: ReadableStream<Uint8Array> = new ReadableStream({
        pull(controller) {
          if (count < 10) {
            if (count % 2 === 0) {
              controller.enqueue(arrHex);
            } else {
              controller.enqueue(matAndExtraHex);
            }
            ++count;
          } else {
            controller.close();
          }
        },
      });
      return new Response(stream);
    };

    reader = stream(createMockResponse()).getReader();
    for (let i = 0; i < 10; ++i) {
      if (i % 2 === 0) {
        expect(await reader.read()).toEqual({
          done: false,
          value: mat,
        });
      } else {
        expect(await reader.read()).toEqual({
          done: false,
          value: matAndExtra,
        });
      }
    }
  });
});

import { describe, expect, it } from "vitest";
import { NDArrayAndExtra, decode, encode, stream } from "../src/npblob";
import { ReadableStream } from "stream/web";

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

  it("encode and decode roundtrip", () => {
    // Test single array without extra
    const encodedMat = encode([mat]);
    const decodedMat = decode(encodedMat);
    expect(decodedMat.length).toEqual(1);
    expect(decodedMat[0][0].shape).toEqual(mat[0].shape);
    expect(decodedMat[0][0].dType).toEqual(mat[0].dType);
    expect(Array.from(decodedMat[0][0].data as Float32Array)).toEqual(
      Array.from(mat[0].data as Float32Array),
    );
    expect(decodedMat[0][1]).toBeNull();

    // Test single array with extra
    const encodedMatExtra = encode([matAndExtra]);
    const decodedMatExtra = decode(encodedMatExtra);
    expect(decodedMatExtra.length).toEqual(1);
    expect(decodedMatExtra[0][0].shape).toEqual(matAndExtra[0].shape);
    expect(decodedMatExtra[0][0].dType).toEqual(matAndExtra[0].dType);
    expect(Array.from(decodedMatExtra[0][0].data as Float32Array)).toEqual(
      Array.from(matAndExtra[0].data as Float32Array),
    );
    expect(decodedMatExtra[0][1]).toEqual(matAndExtra[1]);
  });

  it("encode and decode multiple arrays", () => {
    const multiple = [mat, matAndExtra, mat, matAndExtra, mat];
    const encoded = encode(multiple);
    const decoded = decode(encoded);
    
    expect(decoded.length).toEqual(5);
    
    // Check alternating pattern
    for (let i = 0; i < 5; ++i) {
      expect(decoded[i][0].shape).toEqual([2, 3]);
      expect(decoded[i][0].dType).toEqual("float32");
      if (i % 2 === 0) {
        expect(decoded[i][1]).toBeNull();
      } else {
        expect(decoded[i][1]).toEqual({ min: 1.1, max: 6.6 });
      }
    }
  });

  it("stream single array without extra", async () => {
    const encoded = encode([mat]);
    const reader = stream(encoded).getReader();
    const result = await reader.read();
    expect(result.done).toBe(false);
    expect(result.value[0].shape).toEqual(mat[0].shape);
    expect(result.value[0].dType).toEqual(mat[0].dType);
    expect(Array.from(result.value[0].data as Float32Array)).toEqual(
      Array.from(mat[0].data as Float32Array),
    );
    expect(result.value[1]).toBeNull();
    
    const end = await reader.read();
    expect(end.done).toBe(true);
  });

  it("stream single array with extra", async () => {
    const encoded = encode([matAndExtra]);
    const reader = stream(encoded).getReader();
    const result = await reader.read();
    expect(result.done).toBe(false);
    expect(result.value[0].shape).toEqual(matAndExtra[0].shape);
    expect(result.value[0].dType).toEqual(matAndExtra[0].dType);
    expect(result.value[1]).toEqual(matAndExtra[1]);
    
    const end = await reader.read();
    expect(end.done).toBe(true);
  });

  it("stream multiple arrays", async () => {
    const multiple = [mat, matAndExtra, mat, matAndExtra];
    const encoded = encode(multiple);
    const reader = stream(encoded).getReader();
    
    for (let i = 0; i < 4; ++i) {
      const result = await reader.read();
      expect(result.done).toBe(false);
      expect(result.value[0].shape).toEqual([2, 3]);
      if (i % 2 === 0) {
        expect(result.value[1]).toBeNull();
      } else {
        expect(result.value[1]).toEqual({ min: 1.1, max: 6.6 });
      }
    }
    
    const end = await reader.read();
    expect(end.done).toBe(true);
  });

  it("stream from Response", async () => {
    const multiple = [mat, matAndExtra];
    const encoded = encode(multiple);
    
    let count = 0;
    const mockStream: ReadableStream<Uint8Array> = new ReadableStream({
      pull(controller) {
        if (count === 0) {
          // Split the data into chunks to test streaming
          controller.enqueue(encoded.slice(0, 10));
          controller.enqueue(encoded.slice(10));
          ++count;
        } else {
          controller.close();
        }
      },
    });
    
    const reader = stream(new Response(mockStream)).getReader();
    
    const result1 = await reader.read();
    expect(result1.done).toBe(false);
    expect(result1.value[0].shape).toEqual([2, 3]);
    
    const result2 = await reader.read();
    expect(result2.done).toBe(false);
    expect(result2.value[0].shape).toEqual([2, 3]);
    expect(result2.value[1]).toEqual({ min: 1.1, max: 6.6 });
    
    const end = await reader.read();
    expect(end.done).toBe(true);
  });

  it("supports different data types", () => {
    const testCases: Array<{ shape: number[]; dType: any; data: any }> = [
      { shape: [3], dType: "uint8", data: new Uint8Array([1, 2, 3]) },
      { shape: [3], dType: "uint16", data: new Uint16Array([100, 200, 300]) },
      { shape: [3], dType: "uint32", data: new Uint32Array([1000, 2000, 3000]) },
      { shape: [3], dType: "int8", data: new Int8Array([-1, 0, 1]) },
      { shape: [3], dType: "int16", data: new Int16Array([-100, 0, 100]) },
      { shape: [3], dType: "int32", data: new Int32Array([-1000, 0, 1000]) },
      { shape: [3], dType: "float32", data: new Float32Array([1.5, 2.5, 3.5]) },
      { shape: [3], dType: "float64", data: new Float64Array([1.1, 2.2, 3.3]) },
    ];

    for (const tc of testCases) {
      const input: NDArrayAndExtra = [
        { shape: tc.shape, dType: tc.dType, data: tc.data },
        null,
      ];
      const encoded = encode([input]);
      const decoded = decode(encoded);
      
      expect(decoded[0][0].shape).toEqual(tc.shape);
      expect(decoded[0][0].dType).toEqual(tc.dType);
      expect(Array.from(decoded[0][0].data as any)).toEqual(Array.from(tc.data));
    }
  });

  it("supports multi-dimensional arrays", () => {
    const testCases: Array<{ shape: number[]; data: number[] }> = [
      { shape: [2, 2, 2], data: [1, 2, 3, 4, 5, 6, 7, 8] },
      { shape: [2, 4], data: [1, 2, 3, 4, 5, 6, 7, 8] },
      { shape: [4, 2], data: [1, 2, 3, 4, 5, 6, 7, 8] },
      { shape: [8], data: [1, 2, 3, 4, 5, 6, 7, 8] },
    ];

    for (const tc of testCases) {
      const input: NDArrayAndExtra = [
        { shape: tc.shape, dType: "float32", data: new Float32Array(tc.data) },
        null,
      ];
      const encoded = encode([input]);
      const decoded = decode(encoded);
      
      expect(decoded[0][0].shape).toEqual(tc.shape);
      expect(Array.from(decoded[0][0].data as Float32Array)).toEqual(tc.data);
    }
  });
});

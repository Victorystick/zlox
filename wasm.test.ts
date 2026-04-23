import { expect, test } from "vitest";
import init from "./zig-out/bin/zlox.wasm?init";

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

test("print", async () => {
  let calls = 0;

  const instance = await init({
    js: {
      jsPrint(ptr: number, len: number) {
        console.log(++calls);
        const bytes = new Uint8Array(instance.exports.memory.buffer, ptr, len);
        console.log(decoder.decode(bytes));
      },
    },
  });

  /** Decodes an error code into a human-readable string. */
  function getErrorName(code: number): string {
    const start = instance.exports.errorName(code);
    const buffer = new Uint8Array(instance.exports.memory.buffer);
    // Find the length by looking for the null terminator.
    const bytes = buffer.slice(start, buffer.indexOf(0, start));
    return decoder.decode(bytes);
  }

  function interpret(source: string) {
    const bytes = encoder.encode(source);
    const ptr = instance.exports.alloc(bytes.length);
    new Uint8Array(instance.exports.memory.buffer).set(bytes, ptr);
    const err = instance.exports.interpret(ptr, bytes.length);
    if (err) {
      console.log("Error", err, getErrorName(err));
    }
    instance.exports.dealloc(ptr, bytes.length);
  }

  // No code has been executed yet, so bytes used should be 0.
  expect(instance.exports.bytesUsed()).toBe(0);

  interpret('var a = "foo"; fun foo() { print a; } foo();');
  const afterFirst = instance.exports.bytesUsed();
  console.log("Bytes used:", afterFirst);

  interpret('print "a" + "b"; a = "bar"; foo();');
  const afterSecond = instance.exports.bytesUsed();
  console.log("Bytes used:", afterSecond);

  // More bytes should be in use after the second interpretation.
  expect(afterSecond).toBeGreaterThan(afterFirst);

  interpret("a +");

  // The parse error should have caused the bytecode to be freed, so no
  // additional bytes should be used.
  expect(instance.exports.bytesUsed()).toBe(afterSecond);
});

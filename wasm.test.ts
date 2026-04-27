import { expect, test } from "vitest";
import { instantiateNode } from "./zlox";

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

test("print", async () => {
  let calls = 0;

  const decoder = new TextDecoder();

  const zlox = await instantiateNode({
    stdout(bytes) {
      console.log(++calls);
      console.log(decoder.decode(bytes));
    },
  });

  function interpret(source: string) {
    const err = zlox.interpret(source);
    if (err) {
      console.log("Error", err, zlox.error(err));
    }
  }

  // No code has been executed yet, so bytes used should be 0.
  expect(zlox.bytesUsed()).toBe(0);

  interpret('var a = "foo"; fun foo() { print a; } foo();');
  const afterFirst = zlox.bytesUsed();
  console.log("Bytes used:", afterFirst);

  interpret('print "a" + "b"; a = "bar"; foo();');
  const afterSecond = zlox.bytesUsed();
  console.log("Bytes used:", afterSecond);

  // More bytes should be in use after the second interpretation.
  expect(afterSecond).toBeGreaterThan(afterFirst);

  interpret("a +");

  // The parse error should have caused the bytecode to be freed, so no
  // additional bytes should be used.
  expect(zlox.bytesUsed()).toBe(afterSecond);
});

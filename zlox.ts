import wasmUrl from "./zig-out/bin/zlox.wasm?url";

export interface ZloxOptions {
  stdout(bytes: Uint8Array): void;
}

export interface ZloxExports {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  dealloc(ptr: number, size: number): void;
  interpret(ptr: number, size: number): number;
  errorName(code: number): number;
  bytesUsed(): number;
}

export interface Zlox {
  interpret(source: string): number;
  error(code: number): string;
  bytesUsed(): number;
}

/** Instantiates the Zlox WebAssembly module in a web environment. */
export async function instantiate(options: ZloxOptions): Promise<Zlox> {
  return createZlox(fetch(wasmUrl), options);
}

/** Instantiates the Zlox WebAssembly module in a Node.js environment. */
export async function instantiateNode(options: ZloxOptions): Promise<Zlox> {
  const fs = await import("node:fs/promises");
  const response = new Response(await fs.readFile("./zig-out/bin/zlox.wasm"), {
    headers: { "Content-Type": "application/wasm" },
  });
  return createZlox(response, options);
}

export async function createZlox(
  response: Response | PromiseLike<Response>,
  options: ZloxOptions,
): Promise<Zlox> {
  const wasm = await WebAssembly.instantiateStreaming(response, {
    js: {
      jsPrint(ptr: number, len: number) {
        const bytes = new Uint8Array(exports.memory.buffer, ptr, len);
        options.stdout(bytes);
      },
    },
  });

  const exports: ZloxExports = wasm.instance.exports as unknown as ZloxExports;

  function interpret(source: string): number {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(source);
    const ptr = exports.alloc(bytes.length);
    new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
    const res = exports.interpret(ptr, bytes.length);
    exports.dealloc(ptr, bytes.length);
    return res;
  }

  function error(code: number): string {
    const start = exports.errorName(code);
    const buffer = new Uint8Array(exports.memory.buffer);
    // Find the length by looking for the null terminator.
    const bytes = buffer.slice(start, buffer.indexOf(0, start));
    const decoder = new TextDecoder();
    return decoder.decode(bytes);
  }

  function bytesUsed(): number {
    return exports.bytesUsed();
  }

  return { interpret, error, bytesUsed };
}

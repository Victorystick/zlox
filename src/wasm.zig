const std = @import("std");
const zlox = @import("zlox");

const Writer = std.Io.Writer;

extern "js" fn jsPrint(ptr: [*]const u8, len: usize) void;

fn write(data: []const u8) void {
    jsPrint(data.ptr, data.len);
}

fn drain(w: *Writer, data: []const []const u8, splat: usize) Writer.Error!usize {
    if (w.end > 0) {
        write(w.buffered());
        w.end = 0;
    }

    var n: usize = 0;
    for (data[0 .. data.len - 1]) |slice| {
        write(slice);
        n += slice.len;
    }

    const last = data[data.len - 1];
    if (last.len > 0) {
        for (0..splat) |_| {
            write(last);
        }
        n += splat * last.len;
    }

    return n;
}

const gpa = std.heap.wasm_allocator;

/// The number of bytes currently allocated by this module. Used for tracking
/// memory usage from JavaScript.
var usedMemory: usize = 0;

fn sizeAlloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
    if (gpa.vtable.alloc(undefined, len, alignment, ret_addr)) |a| {
        usedMemory += len;
        return a;
    } else {
        return null;
    }
}

fn sizeFree(_: *anyopaque, ptr: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
    gpa.vtable.free(undefined, ptr, alignment, ret_addr);
    usedMemory -= ptr.len;
}

const sized: std.mem.Allocator = .{
    .ptr = undefined,
    .vtable = &.{
        .alloc = sizeAlloc,
        // TODO: Implement `resize` and `remap`.
        .resize = gpa.vtable.resize,
        .remap = gpa.vtable.remap,
        .free = sizeFree,
    },
};

export fn bytesUsed() usize {
    return usedMemory;
}

export fn errorName(num: usize) usize {
    const err = @errorFromInt(@as(std.meta.Int(.unsigned, @bitSizeOf(anyerror)), @intCast(num)));
    return @intFromPtr(@errorName(err).ptr);
}

/// Allocates bytes and returns a pointer to the allocated memory. Used to pass
/// stings from JavaScript. Returns `null` on failure.
export fn alloc(size: usize) ?[*]u8 {
    const arr = sized.alloc(u8, size) catch return @ptrFromInt(0);
    return arr.ptr;
}

/// Deallocates memory previously allocated with `alloc`. The `ptr` and `size`
/// parameters must match the values used in the corresponding call to `alloc`.
export fn dealloc(ptr: [*]u8, size: usize) void {
    sized.free(ptr[0..size]);
}

// Actual implementation.

var buffer: [1024]u8 = undefined;
var writer: Writer = .{ .buffer = &buffer, .vtable = &.{ .drain = drain } };
var stack: [64]zlox.Value = undefined;

var vm = zlox.VM.init(sized, &writer, &stack);

/// Interprets a Lox script. The `ptr` and `len` parameters specify the
/// location of the script in memory. Returns 0 on success, or a non-zero error
/// code on failure. The error code can be passed to `errorName` to get a
/// string describing the error.
export fn interpret(ptr: [*]const u8, len: usize) i32 {
    // Reconstruct the slice from the pointer and length
    const script = ptr[0..len];

    // Catch errors and return a simple status code (e.g., 0 for success)
    _ = vm.interpret(script) catch |err| {
        // Log the error or return a specific code
        return @intFromError(err);
    };

    return 0;
}

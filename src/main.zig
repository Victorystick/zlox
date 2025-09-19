const std = @import("std");
const zlox = @import("zlox");

pub fn main() !void {
    if (std.os.argv.len > 2) {
        std.debug.print("Usage: zlox [path]\n", .{});
        std.process.exit(64);
    }

    var args = std.process.args();

    // Skip binary name.
    _ = args.skip();

    if (args.next()) |file| {
        try runFile(file);
    } else {
        var buffer: [1024]u8 = undefined;
        var outBuf: [1024]u8 = undefined;
        var in = std.fs.File.stdin().reader(&buffer);
        var out = std.fs.File.stdout().writer(&outBuf);

        while (true) {
            std.debug.print("> ", .{});
            const res = try in.interface.takeSentinel('\n');
            std.debug.print("\n", .{});

            try zlox.interpret(res, &out.interface);
        }
    }
}

fn runFile(name: []const u8) !void {
    const alloc = std.heap.page_allocator;

    const buf = try readFileToBuffer(alloc, name);
    defer alloc.free(buf);

    var writer = zlox.ChunkWriter.init(alloc);
    defer writer.deinit();

    if (!try zlox.compile(buf, &writer)) {
        std.debug.print("Compile error.", .{});
        std.process.exit(64);
    }

    var buffer: [128]u8 = undefined;
    const file = std.fs.File.stdout();
    var stdout = std.fs.File.writer(file, &buffer);
    _ = try zlox.interpretChunk(&writer.chunk(), &stdout.interface);
}

fn readFileToBuffer(allocator: std.mem.Allocator, fname: []const u8) ![]u8 {
    const f = try std.fs.cwd().openFile(fname, .{});
    defer f.close(); // The file closes before we exit the function which happens before we work with the buffer.

    const f_len = try f.getEndPos();
    const buf = try allocator.alloc(u8, f_len);
    errdefer allocator.free(buf); // In case an error happens while reading

    const read_bytes = try f.readAll(buf);
    std.debug.print("Read {} bytes\n", .{read_bytes});
    return buf;
}

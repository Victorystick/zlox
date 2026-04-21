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

    var outBuf: [1024]u8 = undefined;
    var out = std.fs.File.stdout().writer(&outBuf);

    var stack: [128]zlox.Value = undefined;
    var vm = zlox.VM.init(std.heap.page_allocator, &out.interface, &stack);
    defer vm.deinit();

    if (args.next()) |file| {
        const buf = try readFileToBuffer(std.heap.page_allocator, file);
        defer std.heap.page_allocator.free(buf);

        _ = vm.interpret(buf) catch |err| {
            std.debug.print("Error: {}\n", .{err});
            std.process.exit(64);
        };
    } else {
        var buffer: [1024]u8 = undefined;
        var in = std.fs.File.stdin().reader(&buffer);

        while (true) {
            std.debug.print("> ", .{});
            const res = try in.interface.takeSentinel('\n');
            std.debug.print("\n", .{});

            _ = try vm.interpret(res);
        }
    }
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

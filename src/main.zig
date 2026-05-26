const std = @import("std");
const zlox = @import("zlox");

pub fn main(init: std.process.Init) !void {
    var args = init.minimal.args.iterate();

    // Skip binary name.
    _ = args.skip();

    var outBuf: [1024]u8 = undefined;

    var out = std.Io.File.stdout().writer(init.io, &outBuf);

    var stack: [128]zlox.Value = undefined;
    var vm = zlox.VM.init(std.heap.page_allocator, &out.interface, &stack);
    defer vm.deinit();

    if (args.next()) |file| {
        const buf = try std.Io.Dir.cwd().readFileAlloc(init.io, file, std.heap.page_allocator, .unlimited);
        defer std.heap.page_allocator.free(buf);

        _ = vm.interpret(buf) catch |err| {
            std.debug.print("Error: {}\n", .{err});
            std.process.exit(64);
        };
    } else {
        var buffer: [1024]u8 = undefined;
        var in = std.Io.File.stdin().reader(init.io, &buffer);

        while (true) {
            std.debug.print("> ", .{});
            const res = try in.interface.takeSentinel('\n');
            std.debug.print("\n", .{});

            _ = try vm.interpret(res);
        }
    }
}

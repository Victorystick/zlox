//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const debug = .{
    .PrintBytecode = false,
    .TraceExecution = false,
    // Must be false in Wasm since it relies on std.debug.print, which is not
    // available in that target.
    .StressGC = false,
    .LogGC = false,
};

const ValueType = enum {
    nil,
    bool,
    float,
    obj,
};
pub const Value = union(ValueType) {
    nil,
    bool: bool,
    float: f64,
    obj: *Object,

    pub fn deinit(self: Value, alloc: std.mem.Allocator) void {
        switch (self) {
            .obj => |obj| {
                obj.deinit(alloc);
                alloc.destroy(obj);
            },
            else => {},
        }
    }

    pub fn format(self: Value, writer: *std.Io.Writer) !void {
        try switch (self) {
            .nil => writer.writeAll("nil"),
            .float => |fl| writer.print("{d}", .{fl}),
            .bool => |b| writer.writeAll(if (b) "true" else "false"),
            .obj => |o| o.format(writer),
        };
    }

    fn equals(self: Value, other: Value) bool {
        if (std.meta.activeTag(self) != std.meta.activeTag(other)) return false;

        // The values share the same tag. Let's check their contents.
        return switch (self) {
            .nil => true,
            .float => |fl| fl == other.float,
            .bool => |b| b == other.bool,
            .obj => |o| o.equals(other.obj.*),
        };
    }

    fn isFalsey(self: Value) bool {
        return switch (self) {
            .nil => true,
            .bool => |b| !b,
            else => false,
        };
    }

    fn isNumber(self: Value) bool {
        return switch (self) {
            .float => true,
            else => false,
        };
    }

    fn isString(self: Value) bool {
        return switch (self) {
            .obj => switch (self.obj.*) {
                .string => true,
                else => false,
            },
            else => false,
        };
    }

    fn asString(self: Value) ?*String {
        return switch (self) {
            .obj => self.obj.asString(),
            else => null,
        };
    }

    fn asFunction(self: Value) ?*const Function {
        return switch (self) {
            .obj => switch (self.obj.*) {
                .function => |*f| f,
                else => null,
            },
            else => null,
        };
    }

    // Cool, but too cryptic.
    // fn as(self: Value, comptime ty: ObjectType) ?*std.meta.fields(Object)[@intFromEnum(ty)].type {
    //     return switch (self) {
    //         .obj => switch (self.obj.*) {
    //             ty => |*val| val,
    //             else => null,
    //         },
    //         else => null,
    //     };
    // }

    fn asInstance(self: Value) ?*Instance {
        return switch (self) {
            .obj => switch (self.obj.*) {
                .instance => |*i| i,
                else => null,
            },
            else => null,
        };
    }

    fn mark(self: Value) void {
        return switch (self) {
            .obj => |o| o.mark(),
            else => {},
        };
    }
};

const String = struct {
    chars: []const u8,
    hash: u64,

    // Creates a new string by copying the given one. The returned string owns
    // its memory and must be freed with `deinit`.
    fn init(alloc: std.mem.Allocator, str: []const u8) !String {
        const chars = try alloc.alloc(u8, str.len);
        @memcpy(chars, str);
        return String{
            .chars = chars,
            .hash = std.hash_map.hashString(str),
        };
    }

    fn deinit(self: String, alloc: std.mem.Allocator) void {
        alloc.free(self.chars);
    }

    // Creates a new string that borrows the given one. The returned string
    // does not own its memory and must not be freed.
    fn unowned(str: []const u8) String {
        return String{
            .chars = str,
            .hash = std.hash_map.hashString(str),
        };
    }
};
const StringHashing = struct {
    pub fn hash(self: @This(), s: *const String) u64 {
        _ = self;
        return s.hash;
    }
    pub fn eql(self: @This(), a: *const String, b: *const String) bool {
        _ = self;
        return std.mem.eql(u8, a.chars, b.chars);
    }
};

const FunctionType = enum {
    Function,
    Script,
};

const Function = struct {
    name: ?String = null,
    arity: u8 = 0,
    upvalueCount: u8 = 0,
    chunk: Chunk,
    typ: FunctionType,

    fn deinit(self: Function, alloc: std.mem.Allocator) void {
        if (self.name) |name| {
            name.deinit(alloc);
        }
        self.chunk.deinit(alloc);
    }

    pub fn format(self: Function, writer: *std.Io.Writer) !void {
        if (self.name) |name| {
            try writer.print("<fn {s}/{d}>", .{ name.chars, self.arity });
        } else {
            try writer.writeAll("<script>");
        }
    }

    fn disassemble(self: *const Function, io: *std.Io.Writer) !void {
        try io.print("== {f} ==\n", .{self});
        try self.chunk.disassemble(io);
    }

    fn debugDisassemble(self: *const Function) !void {
        var buf: [1024]u8 = undefined;
        const stderr = std.debug.lockStderr(&buf);
        defer std.debug.unlockStderr();
        var io = &stderr.file_writer.interface;
        try self.disassemble(io);
        try io.flush();
    }
};

const Upvalue = struct {
    // Unowned.
    location: *Value,
    closed: Value = .nil,
    // Unowned.
    next: ?*Upvalue = null,
};

const Closure = struct {
    // Unowned.
    function: *const Function,
    // Owned.
    upvalues: []*Upvalue,

    fn init(alloc: std.mem.Allocator, function: *const Function) error{OutOfMemory}!Closure {
        return Closure{
            .function = function,
            .upvalues = try alloc.alloc(*Upvalue, function.upvalueCount),
        };
    }

    fn deinit(self: Closure, alloc: std.mem.Allocator) void {
        alloc.free(self.upvalues);
    }

    pub fn format(self: Closure, writer: *std.Io.Writer) !void {
        return self.function.format(writer);
    }
};

const Class = struct {
    // Unowned.
    name: *String,

    pub fn format(self: Class, writer: *std.Io.Writer) !void {
        // Note: In the book, a class just prints its name.
        try writer.print("<class {s}>", .{self.name.chars});
    }
};

const Instance = struct {
    // Unowned.
    class: *Class,
    fields: ValueMap = .empty,

    pub fn deinit(self: *Instance, alloc: std.mem.Allocator) void {
        self.fields.deinit(alloc);
    }

    pub fn format(self: Instance, writer: *std.Io.Writer) !void {
        // Note: In the book, an instance prints the class name
        // followed by " instance".
        try writer.print("<instance {s}>", .{self.class.name.chars});
    }
};

const NativeErr = error{
    OutOfMemory,
    RuntimeError,
    WriteFailed,
};

const NativeFn = *const fn (*VM, []Value) NativeErr!Value;

const Native = struct {
    function: NativeFn,

    pub fn format(_: Native, writer: *std.Io.Writer) !void {
        try writer.writeAll("<native fn>");
    }
};

const ObjectType = enum {
    string,
    function,
    native,
    closure,
    upvalue,
    class,
    instance,
};
const Object = union(ObjectType) {
    string: String,
    function: Function,
    native: Native,
    closure: Closure,
    upvalue: Upvalue,
    class: Class,
    instance: Instance,

    pub fn deinit(self: *Object, alloc: std.mem.Allocator) void {
        if (comptime debug.LogGC) {
            std.debug.print("  Free    {s: <10}: {f}\n", .{ @tagName(@as(ObjectType, self.*)), self });
        }
        switch (self.*) {
            .string => |string| string.deinit(alloc),
            .function => |function| function.deinit(alloc),
            .closure => |closure| closure.deinit(alloc),
            .instance => |*instance| instance.deinit(alloc),
            .class => {}, // |class| class.deinit(alloc),
            .upvalue => {},
            .native => {},
        }
    }

    fn asString(self: *Object) ?*String {
        return switch (self.*) {
            .string => |*s| s,
            else => null,
        };
    }

    fn mark(self: *Object) void {
        const node: *VM.ObjectNode = @fieldParentPtr("data", self);
        node.isMarked = true;
    }

    pub fn format(self: Object, writer: *std.Io.Writer) !void {
        try switch (self) {
            .string => |string| writer.writeAll(string.chars),
            .function => |function| function.format(writer),
            .native => |native| native.format(writer),
            .closure => |closure| closure.format(writer),
            .upvalue => writer.writeAll("<upvalue>"),
            .class => |class| class.format(writer),
            .instance => |instance| instance.format(writer),
        };
    }

    fn equals(self: Object, other: Object) bool {
        return std.meta.eql(self, other);
    }
};

const OpCode = enum(u8) {
    Return,
    Constant,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Nil,
    True,
    False,
    Not,
    Equal,
    Greater,
    Less,
    Print,
    Pop,
    DefineGlobal,
    SetGlobal,
    GetGlobal,
    SetLocal,
    GetLocal,
    SetUpvalue,
    GetUpvalue,
    CloseUpvalue,
    JumpIfFalse,
    Jump,
    /// A backwards jump, used for loops.
    Loop,
    Call,
    Closure,
    Class,
    GetProperty,
    SetProperty,
    DelProperty,
    _,
};

const Chunk = struct {
    code: []u8 = undefined,
    constants: []Value = undefined,
    /// When true, this chunk owns its object constants and will free them on deinit.
    /// Set to false for GC-managed chunks where the GC owns the object constants.
    ownsObjects: bool = true,

    pub fn deinit(self: Chunk, alloc: std.mem.Allocator) void {
        alloc.free(self.code);
        if (self.ownsObjects) {
            for (self.constants) |constant| {
                constant.deinit(alloc);
            }
        }
        alloc.free(self.constants);
    }

    pub fn disassemble(chunk: *const Chunk, io: *std.Io.Writer) !void {
        var offset: usize = 0;
        while (offset < chunk.code.len) {
            offset = try disassembleInstruction(chunk, offset, io);
        }

        try io.flush();

        // for (0..chunk.constants.len) |i| {
        //     try io.print("{} {f}\n", .{ i, chunk.constants[i] });
        // }

        // try io.flush();
    }

    pub fn readShort(self: *const Chunk, offset: usize) usize {
        var res: usize = 0;
        res |= @as(usize, self.code[offset]) << 8;
        res |= @as(usize, self.code[offset + 1]);
        return res;
    }
};

fn disassembleInstruction(chunk: *const Chunk, offset: usize, io: *std.Io.Writer) !usize {
    try io.print("{d:0>4} ", .{offset});

    return switch (@as(OpCode, @enumFromInt(chunk.code[offset]))) {
        .Constant => {
            const index = chunk.code[offset + 1];
            const constant = chunk.constants[index];
            try io.print("{s} {d: >3} {f}\n", .{ "Constant    ", index, constant });
            return offset + 2;
        },
        .DefineGlobal => {
            const index = chunk.code[offset + 1];
            const constant = chunk.constants[index];
            try io.print("{s} {d: >3} {f}\n", .{ "DefineGlobal", index, constant });
            return offset + 2;
        },
        .GetGlobal => {
            const index = chunk.code[offset + 1];
            const constant = chunk.constants[index];
            try io.print("{s} {d: >3} {f}\n", .{ "GetGlobal   ", index, constant });
            return offset + 2;
        },
        .SetGlobal => {
            const index = chunk.code[offset + 1];
            const constant = chunk.constants[index];
            try io.print("{s} {d: >3} {f}\n", .{ "SetGlobal   ", index, constant });
            return offset + 2;
        },
        .GetLocal => byteInstruction("GetLocal", chunk, offset, io),
        .SetLocal => byteInstruction("SetLocal", chunk, offset, io),
        .GetUpvalue => byteInstruction("GetUpvalue", chunk, offset, io),
        .SetUpvalue => byteInstruction("SetUpvalue", chunk, offset, io),
        .Class => byteInstruction("Class", chunk, offset, io),
        .CloseUpvalue => simpleInstruction("CloseUpvalue", offset, io),
        .Return => simpleInstruction("Return", offset, io),
        .Negate => simpleInstruction("Negate", offset, io),
        .Add => simpleInstruction("Add", offset, io),
        .Subtract => simpleInstruction("Subtract", offset, io),
        .Multiply => simpleInstruction("Multiply", offset, io),
        .Divide => simpleInstruction("Divide", offset, io),
        .False => simpleInstruction("False", offset, io),
        .True => simpleInstruction("True", offset, io),
        .Nil => simpleInstruction("Nil", offset, io),
        .Not => simpleInstruction("Not", offset, io),
        .Equal => simpleInstruction("Equal", offset, io),
        .Greater => simpleInstruction("Greater", offset, io),
        .Less => simpleInstruction("Less", offset, io),
        .Print => simpleInstruction("Print", offset, io),
        .Pop => simpleInstruction("Pop", offset, io),
        .JumpIfFalse => {
            const toSkip = chunk.readShort(offset + 1);
            try io.print("{s} {d: <3} \n", .{ "JumpIfFalse   ", @as(isize, @intCast(toSkip + 2)) });
            return offset + 3;
        },
        .Jump => {
            const toSkip = chunk.readShort(offset + 1);
            try io.print("{s} {d: <3} \n", .{ "Jump          ", @as(isize, @intCast(toSkip + 2)) });
            return offset + 3;
        },
        .Loop => {
            const toSkip = chunk.readShort(offset + 1);
            try io.print("{s} {d: <3} \n", .{ "Loop          ", -@as(isize, @intCast(toSkip + 2)) });
            return offset + 3;
        },
        .Call => byteInstruction("Call", chunk, offset, io),
        .Closure => {
            // Read the following constant.
            const constantIndex = chunk.code[offset + 1];
            const constant = chunk.constants[constantIndex];
            try io.print("{s: <12} {d: >3} {f}\n", .{ "Closure", constantIndex, constant });

            if (constant.asFunction()) |function| {
                // Read the upvalue count.
                var upvalueOffset = offset + 2;
                for (0..function.upvalueCount) |_| {
                    const kind = if (chunk.code[upvalueOffset] == 1) "local" else "upvalue";
                    const index = chunk.code[upvalueOffset + 1];
                    try io.print("{d:0>4} |              {s: <8} {d: >3}\n", .{ upvalueOffset, kind, index });
                    upvalueOffset += 2;
                }

                return upvalueOffset;
            } else {
                try io.writeAll("Closure with non-function constant!\n");
            }

            return offset + 2;
        },
        .GetProperty => byteInstruction("GetProperty", chunk, offset, io),
        .SetProperty => byteInstruction("SetProperty", chunk, offset, io),
        .DelProperty => byteInstruction("DelProperty", chunk, offset, io),
        _ => {
            try io.print("Unknown opcode {d}\n", .{chunk.code[offset]});
            return offset + 1;
        },
    };
}

fn simpleInstruction(name: []const u8, offset: usize, io: *std.Io.Writer) !usize {
    try io.print("{s}\n", .{name});
    return offset + 1;
}

fn byteInstruction(name: []const u8, chunk: *const Chunk, offset: usize, io: *std.Io.Writer) !usize {
    const index = chunk.code[offset + 1];
    try io.print("{s: <12} {d: >3}\n", .{ name, index });
    return offset + 2;
}

pub const ChunkWriter = struct {
    gpa: std.mem.Allocator,
    code: std.ArrayList(u8) = .empty,
    constants: std.ArrayList(Value) = .empty,
    upvalueCount: u8 = 0,
    fnType: FunctionType,

    /// Deinitialize with `deinit` or use `toOwnedSlice`.
    pub fn init(gpa: std.mem.Allocator, fnType: FunctionType) ChunkWriter {
        return ChunkWriter{
            .gpa = gpa,
            .fnType = fnType,
        };
    }

    /// Release all allocated memory.
    pub fn deinit(self: *ChunkWriter) void {
        self.code.deinit(self.gpa);
        for (self.constants.items) |val| {
            val.deinit(self.gpa);
        }
        self.constants.deinit(self.gpa);
    }

    fn func(self: *ChunkWriter) !Function {
        return Function{
            .chunk = try self.chunk(),
            .typ = self.fnType,
            .upvalueCount = self.upvalueCount,
        };
    }

    /// Returns a chunk that owns its memory.
    pub fn chunk(self: *ChunkWriter) !Chunk {
        const code = try self.code.toOwnedSlice(self.gpa);
        errdefer self.gpa.free(code);
        return Chunk{
            .code = code,
            .constants = try self.constants.toOwnedSlice(self.gpa),
        };
    }

    pub fn emit(self: *ChunkWriter, byte: u8) !void {
        try self.code.append(self.gpa, byte);
    }

    pub fn emitOp(self: *ChunkWriter, op: OpCode) !void {
        try self.emit(@intFromEnum(op));
    }

    pub fn emitLoop(self: *ChunkWriter, offset: usize) !void {
        try self.emitOp(.Loop);

        // Account for the two bytes of the jump offset itself.
        const loop = self.code.items.len - offset + 2;
        if (loop > 0xFFFF) {
            return error.JumpTooLong;
        }

        try self.emit(@intCast((loop >> 8) & 0xFF));
        try self.emit(@intCast(loop & 0xFF));
    }

    pub fn emitJump(self: *ChunkWriter, op: OpCode) !usize {
        try self.emitOp(op);
        // Placeholders for the jump offset, which we'll patch later.
        try self.emit(0xFF);
        try self.emit(0xFF);
        return self.code.items.len - 2;
    }

    pub fn patchJump(self: *ChunkWriter, offset: usize) !void {
        const jump = self.code.items.len - offset - 2;
        if (jump > 0xFFFF) {
            return error.JumpTooLong;
        }

        self.code.items[offset] = @intCast((jump >> 8) & 0xFF);
        self.code.items[offset + 1] = @intCast(jump & 0xFF);
    }

    pub fn emitOps(self: *ChunkWriter, op1: OpCode, op2: OpCode) !void {
        try self.emitOp(op1);
        try self.emitOp(op2);
    }

    pub fn makeString(self: *ChunkWriter, str: []const u8) !u8 {
        // Don't allocate the same string twice.
        for (0..self.constants.items.len) |i| {
            if (self.constants.items[i].asString()) |other| {
                if (std.mem.eql(u8, str, other.chars)) {
                    return @intCast(i);
                }
            }
        }

        const obj = try self.gpa.create(Object);
        errdefer self.gpa.destroy(obj);
        obj.* = Object{ .string = try String.init(self.gpa, str) };
        return try self.makeConstant(Value{ .obj = obj });
    }

    pub fn emitString(self: *ChunkWriter, str: []const u8) !void {
        const index = try self.makeString(str);
        try self.emitConstant(index);
    }

    pub fn makeConstant(self: *ChunkWriter, val: Value) !u8 {
        const index: u8 = @intCast(self.constants.items.len);
        try self.constants.append(self.gpa, val);
        return index;
    }

    pub fn emitConstant(self: *ChunkWriter, index: u8) !void {
        try self.emitOp(.Constant);
        try self.emit(index);
    }
};

fn StringMap(comptime V: type) type {
    return std.hash_map.HashMapUnmanaged(*const String, V, StringHashing, 80);
}

const ValueMap = StringMap(Value);

// A call frame represents a single function invocation. It keeps track of the
// function being called, where we are in that function's bytecode, and where
// the function's local variables are on the stack.
// It does not own any of its data, that is all owned by the VM.
const CallFrame = struct {
    const Self = @This();

    closure: *const Closure,
    ip: usize,
    slots: []Value,

    fn runnable(frame: Self) bool {
        return frame.ip < frame.closure.function.chunk.code.len;
    }

    /// Returns the next opcode.
    fn op(vm: *Self) OpCode {
        return @enumFromInt(vm.byte());
    }

    fn constant(vm: *Self) Value {
        return vm.closure.function.chunk.constants[vm.byte()];
    }

    fn slot(vm: *Self) *Value {
        return &vm.slots[vm.byte()];
    }

    fn byte(vm: *Self) u8 {
        const val = vm.closure.function.chunk.code[vm.ip];
        vm.ip += 1;
        return val;
    }

    fn offset(vm: *Self) usize {
        const addr = vm.closure.function.chunk.readShort(vm.ip);
        vm.ip += 2;
        return addr;
    }
};

fn clockNative(_: *VM, _: []Value) NativeErr!Value {
    return Value{ .float = @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0 };
}

pub const VM = struct {
    alloc: std.mem.Allocator,
    io: *std.Io.Writer,

    frames: [64]CallFrame = undefined,
    frameCount: usize = 0,

    stack: []Value,
    stackTop: usize = 0,

    // Interned strings.
    strings: StringMap(*Object) = .empty,
    globals: ValueMap = .empty,

    // The head of a linked list of open upvalues.
    openUpvalues: ?*Upvalue = null,

    // A linked list of all allocated objects.
    objects: ?*ObjectNode = null,
    grayStack: std.ArrayList(*ObjectNode) = .empty,
    isCollecting: bool = false,

    const ObjectNode = struct {
        next: ?*ObjectNode,
        isMarked: bool = false,
        data: Object,
    };

    const Result = enum { OK, RuntimeError };

    pub fn init(alloc: std.mem.Allocator, io: *std.Io.Writer, stack: []Value) VM {
        return VM{
            .alloc = alloc,
            .io = io,
            .stack = stack,
        };
    }

    pub fn deinit(vm: *VM) void {
        vm.strings.deinit(vm.alloc);
        vm.globals.deinit(vm.alloc);
        vm.grayStack.deinit(vm.alloc);

        while (vm.objects) |n| {
            n.data.deinit(vm.alloc);
            vm.objects = n.next;
            vm.alloc.destroy(n);
        }
    }

    fn gc(vm: *VM) !void {
        if (vm.isCollecting) {
            // This can happen if GC is triggered during GC, for example by a native function. In that case, we just skip the nested GC call.
            if (comptime debug.LogGC) {
                std.debug.print("-- GC already in progress\n", .{});
            }
            return;
        }
        vm.isCollecting = true;
        defer vm.isCollecting = false;

        if (comptime debug.LogGC) {
            std.debug.print("-- GC begin\n", .{});
        }

        try vm.markRoots();
        try vm.traceReferences();
        // try vm.removeUnreachableStrings();
        vm.sweep();

        if (comptime debug.LogGC) {
            std.debug.print("-- GC end\n", .{});
        }
    }

    fn markRoots(vm: *VM) !void {
        for (vm.stack[0..vm.stackTop]) |*val| {
            try vm.markValue(val);
        }

        // The book says to also mark the strings, but I store them
        // differently. Let's see if this causes any issues.
        try vm.markValueMap(&vm.globals);

        for (vm.frames[0..vm.frameCount]) |frame| {
            const o: *Object = @constCast(@fieldParentPtr("closure", frame.closure));
            try vm.mark(o);
        }

        var nextUpvalue = vm.openUpvalues;
        while (nextUpvalue) |upvalue| {
            const o: *Object = @constCast(@fieldParentPtr("upvalue", upvalue));
            try vm.mark(o);
            nextUpvalue = upvalue.next;
        }

        // TODO: Do we need to cover compiler roots? In the book, the compiler
        // is closer connected to the VM than it is here.
    }

    fn markValueMap(vm: *VM, map: *ValueMap) !void {
        var iter = map.iterator();
        while (iter.next()) |entry| {
            // The key_ptr points to the ObjectNode in the global map.
            // It must be cast to *Object to use vm.mark
            try vm.mark(@constCast(@fieldParentPtr("string", entry.key_ptr.*)));
            try vm.markValue(entry.value_ptr);
        }
    }

    fn markValue(vm: *VM, val: *Value) !void {
        switch (val.*) {
            .obj => |o| try vm.mark(o),
            else => {},
        }
    }

    fn mark(vm: *VM, o: *Object) !void {
        const n: *ObjectNode = @fieldParentPtr("data", o);
        if (n.isMarked) return;
        if (comptime debug.LogGC) {
            std.debug.print("  Marking {s: <10}: {f}\n", .{ @tagName(@as(ObjectType, o.*)), o });
        }
        n.isMarked = true;
        switch (o.*) {
            .string, .native => {},
            else => try vm.grayStack.append(vm.alloc, n),
        }
    }

    /// Traces all gray stack (in progress) objects and marks reachable references.
    fn traceReferences(vm: *VM) !void {
        while (vm.grayStack.items.len > 0) {
            const node = vm.grayStack.items[vm.grayStack.items.len - 1];
            vm.grayStack.items.len -= 1;

            switch (node.data) {
                .string => {},
                .function => |function| {
                    for (function.chunk.constants) |*constant| {
                        try vm.markValue(constant);
                    }
                },
                .native => {},
                .closure => |closure| {
                    try vm.mark(@constCast(@fieldParentPtr("function", closure.function)));
                    for (closure.upvalues) |upvalue| {
                        try vm.mark(@fieldParentPtr("upvalue", upvalue));
                    }
                },
                .upvalue => |upvalue| {
                    try vm.markValue(upvalue.location);
                },
                .instance => |*i| {
                    try vm.mark(@fieldParentPtr("class", i.class));
                    try vm.markValueMap(&i.fields);
                },
                .class => {},
            }
        }
    }

    test "remove unreachable strings" {
        var disc: std.Io.Writer.Discarding = .init(&.{});
        var stack: [50]Value = undefined;
        var vm = VM.init(std.testing.allocator, &disc.writer, &stack);
        defer vm.deinit();

        _ = try vm.intern(String.unowned("str1"));
        _ = try vm.intern(String.unowned("str2"));

        try vm.gc();

        try std.testing.expectEqual(2, vm.strings.size);

        _ = vm.pop();
        _ = vm.pop();

        try vm.gc();

        try std.testing.expectEqual(0, vm.strings.size);
    }

    /// Walks the linked list of object nodes and deallocates any that aren't marked.
    fn sweep(vm: *VM) void {
        var previous: ?*VM.ObjectNode = null;
        var current = vm.objects;
        while (current) |node| {
            if (node.isMarked) {
                node.isMarked = false;
                previous = current;
                current = node.next;
            } else {
                current = node.next;
                if (previous) |prev| {
                    prev.next = current;
                } else {
                    vm.objects = current;
                }

                if (node.data.asString()) |string| {
                    _ = vm.strings.remove(string);
                }
                node.data.deinit(vm.alloc);
                vm.alloc.destroy(node);
            }
        }
    }

    pub fn interpret(vm: *VM, source: []const u8) !Result {
        // TODO: Differentiate between stdout and stderr. This should be stderr.
        var parser = try Parser.init(vm.alloc, vm.io, source);
        defer parser.deinit();

        const func = try parser.compile();
        {
            errdefer func.deinit(vm.alloc);

            if (comptime debug.PrintBytecode) {
                try func.debugDisassemble();
            }
        }

        return vm.run(func);
    }

    fn createObject(vm: *VM, data: Object) !*Object {
        const node = try vm.alloc.create(ObjectNode);
        if (comptime debug.LogGC) {
            std.debug.print("  Free    {s: <10}: {f}\n", .{ @tagName(@as(ObjectType, data)), data });
        }
        node.* = ObjectNode{ .next = vm.objects, .data = data };
        vm.objects = node;
        return &node.data;
    }

    fn defineNative(vm: *VM, name: []const u8, function: NativeFn) !void {
        // Both the string and the native function are stored on the stack
        // so as not to be garbage collected.
        const strObj = try vm.intern(String.unowned(name));
        const nativeObj = try vm.pushObj(.{ .native = .{ .function = function } });

        try vm.globals.put(vm.alloc, &strObj.string, .{ .obj = nativeObj });

        _ = vm.pop();
        _ = vm.pop();
    }

    // Takes ownership of the function.
    fn run(vm: *VM, func: Function) !Result {
        // Implicit assumtion: The function has arity 0.
        const fnObj = try blk: {
            defer func.deinit(vm.alloc);

            // The function that gets passed in does itself owns all its values.
            // This is a problem since the VM may need to keep (at least parts
            // of) the function alive after the run() function returns. To solve
            // this, the VM clones the function.
            break :blk vm.cloneFn(func);
        };

        const o = try blk: {
            const closure = try Closure.init(vm.alloc, &fnObj.function);
            errdefer closure.deinit(vm.alloc);
            break :blk vm.createObject(.{ .closure = closure });
        };
        _ = vm.pop();
        vm.push(.{ .obj = o });

        _ = try vm.call(&o.closure, 0);

        if (comptime debug.LogGC) {
            std.debug.print("Starting execution\n", .{});
        }
        defer if (comptime debug.LogGC) std.debug.print("Finished execution\n", .{});

        return vm.runLoop() catch |err| switch (err) {
            error.RuntimeError => {
                var i: isize = @intCast(vm.frameCount - 1);
                while (i >= 0) : (i -= 1) {
                    const frame = &vm.frames[@intCast(i)];
                    try vm.io.print("[line ??] in {f}\n", .{frame.closure.function});
                }
                return error.RuntimeError;
            },
            else => err,
        };
    }

    fn cloneChunk(vm: *VM, chunk: *const Chunk) ParseError!Chunk {
        const code = try vm.alloc.alloc(u8, chunk.code.len);
        errdefer vm.alloc.free(code);
        @memcpy(code, chunk.code);

        const constants = try vm.alloc.alloc(Value, chunk.constants.len);
        errdefer vm.alloc.free(constants);

        // While non-object constants can be copied as-is, objects need to be
        // cloned so they don't get freed when the original function gets
        // freed. To ensure they don't get GC:ed during cloning, we push them
        // onto the stack and pop them off after we're done.
        var objectCount: usize = 0;
        for (0..chunk.constants.len) |i| {
            switch (chunk.constants[i]) {
                .obj => |o| {
                    objectCount += 1;
                    switch (o.*) {
                        .string => |string| {
                            const obj = try vm.intern(string);
                            constants[i] = Value{ .obj = obj };
                        },
                        .function => |function| {
                            const obj = try vm.cloneFn(function);
                            constants[i] = Value{ .obj = obj };
                        },
                        // All other types only exist at runtime.
                        else => @panic("Found unexpected object type in constants!"),
                    }
                },
                else => {
                    constants[i] = chunk.constants[i];
                },
            }
        }

        // Pop the objects we pushed onto the stack to keep them alive during cloning.
        for (0..objectCount) |_| {
            _ = vm.pop();
        }

        return Chunk{
            .code = code,
            .constants = constants,
            .ownsObjects = false,
        };
    }

    fn cloneFn(vm: *VM, func: Function) !*Object {
        // This copy is safe, as long as we replace the chunk and name with
        // empty values before any other memory allocation occurs. Otherwise,
        // we risk treating the constants of the function as ObjectNodes and
        // marking them during GC, clobbering any surrounding memory.
        var obj = try vm.pushObj(.{ .function = func });
        const copy = &obj.function;
        copy.name = null;
        copy.chunk = Chunk{ .code = &[_]u8{}, .constants = &[_]Value{} };

        copy.chunk = try vm.cloneChunk(&func.chunk);
        errdefer copy.chunk.deinit(vm.alloc);
        if (func.name) |name| {
            copy.name = try String.init(vm.alloc, name.chars);
        }
        return obj;
    }

    fn printStack(vm: *VM, frame: ?*const CallFrame) !void {
        var buf: [1024]u8 = undefined;
        const stderr = std.debug.lockStderr(&buf);
        defer std.debug.unlockStderr();
        var io = &stderr.file_writer.interface;
        try io.writeAll("[ ");
        for (vm.stack[0..vm.stackTop]) |val| {
            try io.print("{f} ", .{val});
        }
        try io.writeAll("]\n");
        if (frame) |f| {
            _ = disassembleInstruction(&f.closure.function.chunk, f.ip, io) catch {};
        }
        try io.flush();
    }

    fn runLoop(vm: *VM) !Result {
        var frame = &vm.frames[vm.frameCount - 1];

        while (frame.runnable()) {
            if (comptime debug.TraceExecution) {
                try vm.printStack(frame);
            }

            switch (frame.op()) {
                .Constant => {
                    const constant = frame.constant();

                    if (constant.asString()) |str| {
                        _ = try vm.intern(str.*);
                    } else {
                        vm.push(constant);
                    }
                },
                .DefineGlobal => {
                    const constant = frame.constant();
                    if (constant.asString()) |str| {
                        const interned = try vm.intern(str.*);
                        try vm.globals.put(vm.alloc, &interned.string, vm.peek(1));
                        _ = vm.pop();
                        _ = vm.pop();
                    } else {
                        try vm.io.print("Expected string, got {f}!\n", .{constant});
                        try vm.io.flush();
                        return .RuntimeError;
                    }
                },
                .SetGlobal => {
                    const constant = frame.constant();
                    if (constant.asString()) |str| {
                        if (vm.globals.getPtr(str)) |ptr| {
                            ptr.* = vm.peek(0);
                        } else {
                            try vm.io.print("Undefined variable '{s}'!\n", .{str.chars});
                            try vm.io.flush();
                            return .RuntimeError;
                        }
                    } else {
                        try vm.io.print("Expected string, got {f}!\n", .{constant});
                        try vm.io.flush();
                        return .RuntimeError;
                    }
                },
                .GetGlobal => {
                    const constant = frame.constant();
                    if (constant.asString()) |str| {
                        if (vm.globals.get(str)) |val| {
                            vm.push(val);
                        } else {
                            try vm.io.print("Undefined variable '{s}'!\n", .{str.chars});
                            try vm.io.flush();
                            return .RuntimeError;
                        }
                    } else {
                        try vm.io.print("Expected string, got {f}!\n", .{constant});
                        try vm.io.flush();
                        return .RuntimeError;
                    }
                },
                .SetLocal => {
                    frame.slot().* = vm.peek(0);
                },
                .GetLocal => {
                    vm.push(frame.slot().*);
                },
                .SetUpvalue => {
                    const slot = frame.byte();
                    frame.closure.upvalues[slot].location.* = vm.peek(0);
                },
                .GetUpvalue => {
                    const slot = frame.byte();
                    vm.push(frame.closure.upvalues[slot].location.*);
                },
                .CloseUpvalue => {
                    try vm.closeUpvalues(&vm.stack[vm.stackTop - 1]);
                    _ = vm.pop();
                },
                .Nil => vm.push(.nil),
                .True => vm.push(Value{ .bool = true }),
                .False => vm.push(Value{ .bool = false }),
                .Not => {
                    const val = vm.pop();
                    vm.push(Value{ .bool = val.isFalsey() });
                },
                .Return => {
                    const result = vm.pop();
                    try vm.closeUpvalues(&frame.slots[0]);
                    vm.frameCount -= 1;
                    if (vm.frameCount == 0) {
                        // Pop the script function off the stack.
                        _ = vm.pop();
                        try vm.io.flush();
                        if (comptime debug.TraceExecution) {
                            try vm.printStack(null);
                        }
                        return .OK;
                    }
                    vm.stackTop = frame.slots.ptr - vm.stack.ptr;
                    vm.push(result);
                    frame = &vm.frames[vm.frameCount - 1];
                },
                .Negate => {
                    const val = vm.pop();
                    switch (val) {
                        .float => {},
                        else => {
                            try vm.io.print("Expected float!\n", .{});
                            try vm.io.flush();
                            return .RuntimeError;
                        },
                    }
                    vm.push(Value{ .float = -val.float });
                },
                .Add => {
                    if (vm.peek(0).isNumber() and vm.peek(0).isNumber()) {
                        // Reuse logic.
                        try vm.binop(.Add);
                    } else if (vm.peek(0).isString() and vm.peek(0).isString()) {
                        try vm.concatenate();
                    } else {
                        try vm.io.print("Operands must be two numbers or two strings!\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    }
                },
                .Subtract => try vm.binop(.Subtract),
                .Multiply => try vm.binop(.Multiply),
                .Divide => try vm.binop(.Divide),
                .Equal => {
                    const b = vm.pop();
                    const a = vm.pop();
                    vm.push(Value{ .bool = a.equals(b) });
                },
                .Greater => try vm.binop(.Greater),
                .Less => try vm.binop(.Less),
                _ => {
                    try vm.io.flush();
                    return .RuntimeError;
                },
                .Print => try vm.io.print("{f}\n", .{vm.pop()}),
                .Pop => _ = vm.pop(),
                .JumpIfFalse => {
                    const toSkip = frame.offset();
                    if (vm.peek(0).isFalsey()) {
                        frame.ip += toSkip;
                    }
                },
                .Jump => {
                    const toSkip = frame.offset();
                    frame.ip += toSkip;
                },
                .Loop => {
                    const toSkip = frame.offset();
                    frame.ip -= toSkip;
                },
                .Call => {
                    const argCount = frame.byte();
                    switch (vm.peek(argCount)) {
                        .obj => switch (vm.peek(argCount).obj.*) {
                            .closure => |*closure| {
                                frame = try vm.call(closure, argCount);
                            },
                            .native => |native| {
                                const args = vm.stack[vm.stackTop - argCount .. vm.stackTop];
                                const result = try native.function(vm, args);
                                vm.stackTop -= argCount + 1;
                                vm.push(result);
                            },
                            .class => |*class| {
                                const obj = try vm.createObject(.{ .instance = .{ .class = class } });
                                vm.stackTop -= argCount + 1;
                                vm.push(.{ .obj = obj });
                            },
                            else => {
                                try vm.io.print("Can only call functions and classes!\n", .{});
                                try vm.io.flush();
                                return error.RuntimeError;
                            },
                        },
                        else => {
                            try vm.io.print("Can only call functions and classes!\n", .{});
                            try vm.io.flush();
                            return .RuntimeError;
                        },
                    }
                },
                .Closure => {
                    const constant = frame.constant();
                    if (constant.asFunction()) |function| {
                        // Hack! Create an upvalue that points to the closure,
                        // and use it for all upvalues before
                        var val: Value = .{ .nil = {} };
                        const tmp = try vm.pushObj(.{ .upvalue = .{ .location = &val } });
                        defer _ = vm.pop();

                        const closure = try Closure.init(vm.alloc, function);
                        errdefer closure.deinit(vm.alloc);
                        const self = try vm.pushObj(.{ .closure = closure });

                        // Hack! Swap the closure with the temp value.
                        vm.swap();

                        val = .{ .obj = self };

                        // Hack! Point all upvalues to the closure itself. That
                        // ensures that iterating
                        for (closure.upvalues) |*upvalue| {
                            upvalue.* = &tmp.upvalue;
                        }

                        // Capture upvalues.
                        for (closure.upvalues) |*upvalue| {
                            const isLocal = frame.byte();
                            const index = frame.byte();
                            if (isLocal == 1) {
                                upvalue.* = try vm.captureUpvalue(&frame.slots[index]);
                            } else {
                                upvalue.* = frame.closure.upvalues[index];
                            }
                        }
                    } else {
                        try vm.io.print("Expected function, got {f}!\n", .{constant});
                        try vm.io.flush();
                        return .RuntimeError;
                    }
                },
                .Class => {
                    const constant = frame.constant();

                    if (constant.asString()) |str| {
                        _ = try vm.pushObj(.{ .class = .{ .name = str } });
                    } else {
                        return error.RuntimeError;
                    }
                },
                .GetProperty => {
                    const instance = vm.peek(0).asInstance() orelse {
                        try vm.io.print("Only instances have properties.\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    const name = frame.constant().asString() orelse {
                        try vm.io.print("Tried to access a non-string property!\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    const value = instance.fields.get(name) orelse {
                        try vm.io.print("Undefined property '{s}'.\n", .{name.chars});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    _ = vm.pop(); // instance
                    vm.push(value);
                },
                .SetProperty => {
                    const instance = vm.peek(1).asInstance() orelse {
                        try vm.io.print("Only instances have properties.\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    const name = frame.constant().asString() orelse {
                        try vm.io.print("Tried to access a non-string property!\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    try instance.fields.put(vm.alloc, name, vm.peek(0));
                    const value = vm.pop();
                    _ = vm.pop();
                    vm.push(value);
                },
                .DelProperty => {
                    const instance = vm.peek(0).asInstance() orelse {
                        try vm.io.print("Only instances have properties.\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    const name = frame.constant().asString() orelse {
                        try vm.io.print("Tried to access a non-string property!\n", .{});
                        try vm.io.flush();
                        return error.RuntimeError;
                    };
                    vm.push(.{ .bool = instance.fields.remove(name) });
                },
            }
        }

        try vm.io.print("Missing a return to terminate the chunk!\n", .{});
        try vm.io.flush();
        return .RuntimeError;
    }

    test "class" {
        try runs(&.{
            \\class Brioche {}
            \\print Brioche;
        },
            \\<class Brioche>
            \\
        );
    }

    test "instance" {
        try runs(&.{
            \\class Brioche {}
            \\print Brioche();
        },
            \\<instance Brioche>
            \\
        );
    }

    test "properties" {
        try runs(&.{
            \\class Ref {}
            \\var r = Ref();
            \\r.value = 42;
            \\print r.value;
        },
            \\42
            \\
        );
    }

    // Interns a string and returns the interned version. The returned string
    // is owned by the VM and will be freed when the VM is deinitialized.
    fn intern(vm: *VM, str: String) !*Object {
        // If we've already interned the string, use that instance.
        if (vm.strings.get(&str)) |string| {
            if (comptime debug.LogGC) {
                std.debug.print("  Interned: {s}\n", .{str.chars});
            }
            vm.push(.{ .obj = string });
            return string;
        }

        // Otherwise, allocate a new one.
        var string = try String.init(vm.alloc, str.chars);
        errdefer string.deinit(vm.alloc);
        const obj = try vm.pushObj(.{ .string = string });
        try vm.strings.put(vm.alloc, &obj.string, obj);
        return obj;
    }

    fn call(vm: *VM, callee: *const Closure, argCount: usize) !*CallFrame {
        const arity = callee.function.arity;
        if (argCount != arity) {
            try vm.io.print("Expected {d} arguments but got {d}!\n", .{ arity, argCount });
            try vm.io.flush();
            return error.RuntimeError;
        }
        if (vm.frameCount == vm.frames.len) {
            try vm.io.print("Stack overflow!\n", .{});
            try vm.io.flush();
            return error.RuntimeError;
        }
        vm.frames[vm.frameCount] = CallFrame{
            .closure = callee,
            .ip = 0,
            .slots = vm.stack[vm.stackTop - argCount - 1 ..],
        };
        const frame = &vm.frames[vm.frameCount];
        vm.frameCount += 1;
        return frame;
    }

    fn captureUpvalue(vm: *VM, local: *Value) !*Upvalue {
        var prevUpvalue: ?*Upvalue = null;
        var upvalue = vm.openUpvalues;

        // We're looking for an upvalue for the `local` pointer, which is
        // guaranteed to be sorted in descending order in the linked list.
        while (upvalue) |u| {
            if (u.location == local) {
                return u;
            } else if (@intFromPtr(u.location) < @intFromPtr(local)) {
                break;
            }
            prevUpvalue = upvalue;
            upvalue = u.next;
        }

        const created = try vm.createObject(.{ .upvalue = .{ .location = local } });
        const createdUpvalue = &created.upvalue;
        createdUpvalue.next = upvalue;

        if (prevUpvalue) |prev| {
            prev.next = createdUpvalue;
        } else {
            vm.openUpvalues = createdUpvalue;
        }

        return createdUpvalue;
    }

    fn closeUpvalues(vm: *VM, last: *Value) !void {
        while (vm.openUpvalues) |upvalue| {
            if (@intFromPtr(upvalue.location) < @intFromPtr(last)) break;
            upvalue.closed = upvalue.location.*;
            upvalue.location = &upvalue.closed;
            vm.openUpvalues = upvalue.next;
        }
    }

    fn binop(vm: *VM, comptime code: OpCode) !void {
        if (!vm.peek(0).isNumber() or !vm.peek(1).isNumber()) {
            try vm.io.print("Expected float!\n", .{});
            try vm.io.flush();
            return error.RuntimeError;
        }
        const b = vm.pop().float;
        const a = vm.pop().float;
        const res = switch (code) {
            .Add => Value{ .float = a + b },
            .Subtract => Value{ .float = a - b },
            .Multiply => Value{ .float = a * b },
            .Divide => Value{ .float = a / b },
            .Greater => Value{ .bool = a > b },
            .Less => Value{ .bool = a < b },
            else => unreachable,
        };
        vm.push(res);
    }

    // Requires the top two values to be strings.
    fn concatenate(vm: *VM) !void {
        const b = vm.peek(0);
        const a = vm.peek(1);

        const first = a.obj.*.string.chars;
        const second = b.obj.*.string.chars;
        var result = try vm.alloc.alloc(u8, first.len + second.len);
        errdefer vm.alloc.free(result);

        @memcpy(result[0..first.len], first);
        @memcpy(result[first.len..], second);
        _ = vm.pop();
        _ = vm.pop();

        const string = String{
            .chars = result,
            .hash = std.hash_map.hashString(result),
        };

        // We need to allocate a new string object for the concatenated string.
        const o = try vm.pushObj(.{ .string = string });
        const entry = try vm.strings.getOrPut(vm.alloc, &o.string);
        if (entry.found_existing) {
            // Free old allocation.
            vm.alloc.free(result);
            vm.push(Value{ .obj = entry.value_ptr.* });
        } else {
            entry.value_ptr.* = o;
        }
    }

    fn peek(vm: *VM, distance: usize) Value {
        return vm.stack[vm.stackTop - (distance + 1)];
    }

    fn push(vm: *VM, val: Value) void {
        vm.stack[vm.stackTop] = val;
        vm.stackTop += 1;
    }

    fn pushObj(vm: *VM, object: Object) !*Object {
        const obj = try vm.createObject(object);
        vm.push(Value{ .obj = obj });
        return obj;
    }

    fn pop(vm: *VM) Value {
        vm.stackTop -= 1;
        return vm.stack[vm.stackTop];
    }

    /// Swaps the top two elements on the stack.
    fn swap(vm: *VM) void {
        const top = vm.peek(0);
        const second = vm.peek(1);

        vm.stack[vm.stackTop - 1] = second;
        vm.stack[vm.stackTop - 2] = top;
    }
};

const TokenType = enum {
    // Single-character tokens.
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    // One or two character tokens.
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    // Literals.
    Identifier,
    String,
    Number,
    // Keywords.
    And,
    Class,
    Else,
    False,
    For,
    Fun,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    Del,

    Special, // Used internally for the parser.
    Error,
    Eof,
};

const Token = struct {
    type: TokenType,
    source: []const u8,
    start: usize,
    line: usize,
};

const Scanner = struct {
    source: []const u8,
    start: usize = 0,
    current: usize = 0,
    line: usize = 1,

    fn advance(scanner: *Scanner) u8 {
        scanner.current += 1;
        return scanner.source[scanner.current - 1];
    }

    fn peek(scanner: *const Scanner) u8 {
        if (scanner.atEnd()) return 0;
        return scanner.source[scanner.current];
    }

    fn peekNext(scanner: *const Scanner) u8 {
        if (scanner.atEnd()) return 0;
        return scanner.source[scanner.current + 1];
    }

    fn atEnd(scanner: *const Scanner) bool {
        return scanner.source.len == scanner.current;
    }

    fn next(scanner: *Scanner) Token {
        scanner.skipWhitespace();
        scanner.start = scanner.current;

        if (scanner.atEnd()) return scanner.token(.Eof);

        const c = scanner.advance();

        if (isAlpha(c)) return scanner.identifier();
        if (isDigit(c)) return scanner.number();

        return switch (c) {
            '(' => scanner.token(.LeftParen),
            ')' => scanner.token(.RightParen),
            '{' => scanner.token(.LeftBrace),
            '}' => scanner.token(.RightBrace),
            ';' => scanner.token(.Semicolon),
            ',' => scanner.token(.Comma),
            '.' => scanner.token(.Dot),
            '-' => scanner.token(.Minus),
            '+' => scanner.token(.Plus),
            '/' => scanner.token(.Slash),
            '*' => scanner.token(.Star),
            '!' => scanner.token(if (scanner.match('=')) .BangEqual else .Bang),
            '=' => scanner.token(if (scanner.match('=')) .EqualEqual else .Equal),
            '<' => scanner.token(if (scanner.match('=')) .LessEqual else .Less),
            '>' => scanner.token(if (scanner.match('=')) .GreaterEqual else .Greater),
            '"' => scanner.string(),
            else => scanner.errorToken("Unexpectred character."),
        };
    }

    fn errorToken(scanner: *const Scanner, text: []const u8) Token {
        return Token{ .type = .Error, .source = text, .start = scanner.start, .line = scanner.line };
    }

    fn match(scanner: *Scanner, expected: u8) bool {
        if (scanner.atEnd()) return false;
        if (scanner.peek() != expected) return false;
        scanner.current += 1;
        return true;
    }

    fn string(scanner: *Scanner) Token {
        while (scanner.peek() != '"' and !scanner.atEnd()) {
            if (scanner.peek() == '\n') scanner.line += 1;
            _ = scanner.advance();
        }

        if (scanner.atEnd()) return scanner.errorToken("Unterminated string.");

        // The closing quote.
        _ = scanner.advance();
        return scanner.token(.String);
    }

    fn number(scanner: *Scanner) Token {
        while (isDigit(scanner.peek())) _ = scanner.advance();

        // Look for a fractional part.
        if (scanner.peek() == '.' and isDigit(scanner.peekNext())) {
            // Consume the ".".
            _ = scanner.advance();

            while (isDigit(scanner.peek())) _ = scanner.advance();
        }

        return scanner.token(.Number);
    }

    fn identifier(scanner: *Scanner) Token {
        while (isAlpha(scanner.peek()) or isDigit(scanner.peek())) _ = scanner.advance();
        return scanner.token(scanner.identifierType());
    }

    fn identifierType(scanner: *Scanner) TokenType {
        return switch (scanner.source[scanner.start]) {
            'a' => scanner.checkKeyword(1, "nd", .And),
            'c' => scanner.checkKeyword(1, "lass", .Class),
            'd' => scanner.checkKeyword(1, "el", .Del),
            'e' => scanner.checkKeyword(1, "lse", .Else),
            'i' => scanner.checkKeyword(1, "f", .If),
            'n' => scanner.checkKeyword(1, "il", .Nil),
            'o' => scanner.checkKeyword(1, "r", .Or),
            'p' => scanner.checkKeyword(1, "rint", .Print),
            'r' => scanner.checkKeyword(1, "eturn", .Return),
            's' => scanner.checkKeyword(1, "uper", .Super),
            'v' => scanner.checkKeyword(1, "ar", .Var),
            'w' => scanner.checkKeyword(1, "hile", .While),
            'f' => {
                if (scanner.current - scanner.start > 1) {
                    return switch (scanner.source[scanner.start + 1]) {
                        'a' => scanner.checkKeyword(2, "lse", .False),
                        'o' => scanner.checkKeyword(2, "r", .For),
                        'u' => scanner.checkKeyword(2, "n", .Fun),
                        else => .Identifier,
                    };
                }
                return .Identifier;
            },
            't' => {
                if (scanner.current - scanner.start > 1) {
                    return switch (scanner.source[scanner.start + 1]) {
                        'h' => scanner.checkKeyword(2, "is", .This),
                        'r' => scanner.checkKeyword(2, "ue", .True),
                        else => .Identifier,
                    };
                }
                return .Identifier;
            },
            else => .Identifier,
        };
    }

    fn checkKeyword(scanner: *Scanner, from: usize, rest: []const u8, typ: TokenType) TokenType {
        if (std.mem.eql(
            u8,
            scanner.source[scanner.start + from .. scanner.current],
            rest,
        )) return typ;
        return .Identifier;
    }

    fn skipWhitespace(scanner: *Scanner) void {
        while (!scanner.atEnd()) {
            switch (scanner.peek()) {
                ' ', '\r', '\t' => _ = scanner.advance(),
                '\n' => {
                    scanner.line += 1;
                    _ = scanner.advance();
                },
                '/' => {
                    if (scanner.peekNext() == '/') {
                        // A comment goes until the end of the line.
                        while (scanner.peek() != '\n' and !scanner.atEnd()) {
                            _ = scanner.advance();
                        }
                    } else {
                        return;
                    }
                },
                else => break,
            }
        }
    }

    fn token(scanner: *const Scanner, typ: TokenType) Token {
        return Token{
            .type = typ,
            .source = scanner.source[scanner.start..scanner.current],
            .start = scanner.start,
            .line = scanner.line,
        };
    }

    const Range = struct {
        start: usize,
        end: usize,
    };

    fn lineRangeFor(scanner: *const Scanner, tok: Token) Range {
        var start = tok.start;
        var end = start + tok.source.len;
        while (0 < start and scanner.source[start - 1] != '\n') start -= 1;
        while (end < scanner.source.len and scanner.source[end] != '\n') end += 1;
        return .{ .start = start, .end = end };
    }

    fn lineFor(scanner: *const Scanner, tok: Token) []const u8 {
        var start = tok.start;
        var end = start + tok.source.len;
        while (0 < start and scanner.source[start - 1] != '\n') start -= 1;
        while (end < scanner.source.len and scanner.source[end] != '\n') end += 1;
        return scanner.source[start..end];
    }
};

fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
        (c >= 'A' and c <= 'Z') or
        c == '_';
}
fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

const MAX_LOCALS = 128;

const Compiler = struct {
    const Self = @This();

    const Local = struct {
        name: Token,
        depth: i32,
        isCaptured: bool = false,
    };

    const UpvalueRef = struct {
        index: u8,
        isLocal: bool,
    };

    writer: ChunkWriter,
    upvalues: [MAX_LOCALS]UpvalueRef = undefined,
    locals: [MAX_LOCALS]Local = undefined,
    localCount: u8 = 1, // First slot is reserved for VM internal use.
    scopeDepth: i32 = 0,
    enclosing: ?*Self = null,

    fn init(alloc: std.mem.Allocator, typ: FunctionType) Compiler {
        var compiler = Compiler{ .writer = ChunkWriter.init(alloc, typ) };
        // The VM uses the first slot of the locals array internally, so we need to reserve it with a dummy value.
        const token = Token{ .type = .Special, .source = "", .start = 0, .line = 0 };
        compiler.locals[0] = Local{ .name = token, .depth = 0 };
        return compiler;
    }

    fn deinit(self: *Self) void {
        self.writer.deinit();
    }

    fn pop(self: *Self) u8 {
        self.scopeDepth -= 1;
        var count: u8 = 0;
        while (self.localCount > 0 and self.locals[self.localCount - 1].depth > self.scopeDepth) : (self.localCount -= 1) {
            count += 1;
        }
        return count;
    }

    fn addLocal(self: *Self, name: Token) !void {
        if (self.localCount == MAX_LOCALS - 1) {
            return error.TooManyLocals;
        }
        var local = &self.locals[self.localCount];
        local.name = name;
        local.depth = -1;
        self.localCount += 1;
    }

    fn markInitialized(self: *Self) void {
        if (self.scopeDepth == 0) return;
        self.locals[self.localCount - 1].depth = self.scopeDepth;
    }

    fn resolveLocal(self: *Self, name: Token) error{ReadInInitializer}!?u8 {
        if (self.localCount == 0) return null;

        var i: i32 = self.localCount - 1;
        while (i >= 0) : (i -= 1) {
            const local = self.locals[@intCast(i)];

            if (std.mem.eql(u8, name.source, local.name.source)) {
                if (local.depth == -1) {
                    return error.ReadInInitializer;
                    // Uninitialized.
                    // return null;
                }
                return @intCast(i);
            }
        }
        return null;
    }

    fn resolveUpvalue(self: *Self, name: Token) !?u8 {
        // We need an enclosing function to have an upvalue.
        if (self.enclosing) |enclosing| {
            if (try enclosing.resolveLocal(name)) |local| {
                enclosing.locals[local].isCaptured = true;
                return try self.addUpvalue(local, true);
            } else if (try enclosing.resolveUpvalue(name)) |upvalue| {
                return try self.addUpvalue(upvalue, false);
            }
        }
        return null;
    }

    fn addUpvalue(self: *Self, index: u8, isLocal: bool) !u8 {
        const upvalueCount = self.writer.upvalueCount;
        for (0..upvalueCount) |i| {
            const upvalue = self.upvalues[i];
            if (upvalue.index == index and upvalue.isLocal == isLocal) {
                return @intCast(i);
            }
        }

        if (upvalueCount == MAX_LOCALS) {
            return error.TooManyLocals;
        }

        self.upvalues[upvalueCount] = .{ .index = index, .isLocal = isLocal };
        self.writer.upvalueCount += 1;
        return @intCast(upvalueCount);
    }

    fn definesInSubscope(self: *Self, name: Token) bool {
        if (self.localCount == 0) return false;

        var i: i32 = self.localCount - 1;
        while (i >= 0) : (i -= 1) {
            const local = self.locals[@intCast(i)];
            if (local.depth != -1 and local.depth < self.scopeDepth) break;

            if (std.mem.eql(u8, name.source, local.name.source)) {
                return true;
            }
        }
        return false;
    }
};

const ParseError = error{
    OutOfMemory,
    TooManyLocals,
    InvalidCharacter,
    JumpTooLong,
    WriteFailed,
    ParseError,
};

const Parser = struct {
    alloc: std.mem.Allocator,
    scanner: Scanner,
    compiler: *Compiler,
    writer: *ChunkWriter,
    err: *std.Io.Writer,
    current: Token = Token{
        .type = .Error,
        .source = "No parse\n",
        .start = 0,
        .line = 0,
    },
    previous: Token = undefined,
    hadError: bool = false,
    panicMode: bool = false,

    fn init(alloc: std.mem.Allocator, err: *std.Io.Writer, source: []const u8) !Parser {
        const compiler = try alloc.create(Compiler);
        compiler.* = Compiler.init(alloc, .Script);
        return Parser{
            .alloc = alloc,
            .scanner = Scanner{ .source = source },
            .compiler = compiler,
            .writer = &compiler.writer,
            .err = err,
        };
    }

    fn deinit(self: *Parser) void {
        self.compiler.deinit();
        self.alloc.destroy(self.compiler);
    }

    const Precedence = enum {
        None,
        Assignment,
        Or,
        And,
        Equality,
        Comparison,
        Term,
        Factor,
        Unary,
        Call,
        Primary,

        fn lowerOrEqualTo(self: Precedence, other: Precedence) bool {
            return @intFromEnum(self) <= @intFromEnum(other);
        }

        fn next(self: Precedence) Precedence {
            return switch (self) {
                .Primary => .Primary,
                else => @enumFromInt(@intFromEnum(self) + 1),
            };
        }
    };

    fn advance(parser: *Parser) void {
        parser.previous = parser.current;

        while (true) {
            parser.current = parser.scanner.next();
            if (parser.current.type != .Error) break;

            parser.errorAt(parser.current, "BOOM!");
        }
    }

    fn errorAt(parser: *Parser, token: Token, text: []const u8) void {
        if (parser.panicMode) return;
        parser.panicMode = true;

        // TODO: Improve error handling.
        parser.err.print("Error", .{}) catch {};

        switch (token.type) {
            .Eof => parser.err.print(" at end", .{}) catch {},
            .Error => {},
            else => parser.err.print(" at '{s}'", .{token.source}) catch {},
        }

        parser.err.print(": {s}\n", .{text}) catch {};

        switch (token.type) {
            .Error, .Special => {},
            else => {
                const range = parser.scanner.lineRangeFor(token);
                parser.err.print("{d: >3}: {s}\n", .{ token.line, parser.scanner.source[range.start..range.end] }) catch {};

                const indent = token.start - range.start + 5; // for the line number, colon and space.
                for (0..indent) |_| {
                    _ = parser.err.write(" ") catch {};
                }
                _ = parser.err.write("^\n") catch {};
            },
        }

        parser.err.flush() catch {};

        parser.hadError = true;
    }

    fn check(parser: *Parser, typ: TokenType) bool {
        return parser.current.type == typ;
    }

    /// Checks for the given token type, and consumes it.
    fn match(parser: *Parser, typ: TokenType) bool {
        const res = parser.check(typ);
        if (res) parser.advance();
        return res;
    }

    fn consume(parser: *Parser, typ: TokenType, text: []const u8) void {
        if (!parser.match(typ)) {
            parser.errorAt(parser.current, text);
        }
    }

    fn call(parser: *Parser, _: bool) !void {
        const argCount = try parser.argumentList();
        try parser.writer.emitOp(.Call);
        try parser.writer.emit(argCount);
    }

    fn argumentList(parser: *Parser) !u8 {
        var argCount: u8 = 0;
        if (!parser.check(.RightParen)) {
            while (true) {
                try parser.expression();
                if (argCount == MAX_LOCALS) {
                    parser.errorAt(parser.current, "Can't have more than 128 arguments.");
                }
                argCount += 1;

                if (!parser.match(.Comma)) break;
            }
        }
        parser.consume(.RightParen, "Expect ')' after arguments.");
        return argCount;
    }

    fn dot(parser: *Parser, canAssign: bool) !void {
        parser.consume(.Identifier, "Expect property name after '.'.");
        const name = try parser.identifierConstant(parser.previous);

        if (canAssign and parser.match(.Equal)) {
            try parser.expression();
            try parser.writer.emitOp(.SetProperty);
        } else {
            try parser.writer.emitOp(.GetProperty);
        }
        try parser.writer.emit(name);
    }

    fn del(parser: *Parser, _: bool) !void {
        try parser.expression();

        // The second last prop should be a get. Replace it with del.
        if (parser.writer.code.items[parser.writer.code.items.len - 2] != @intFromEnum(OpCode.GetProperty)) {
            parser.errorAt(parser.previous, "Requires a variable access!");
            return error.ParseError;
        } else {
            parser.writer.code.items[parser.writer.code.items.len - 2] = @intFromEnum(OpCode.DelProperty);
        }
    }

    fn number(parser: *Parser, _: bool) !void {
        const val = try std.fmt.parseFloat(f64, parser.previous.source);
        const index = try parser.writer.makeConstant(Value{ .float = val });
        try parser.writer.emitConstant(index);
    }

    // Logical and.
    fn land(parser: *Parser, _: bool) !void {
        const endJump = try parser.writer.emitJump(.JumpIfFalse);
        try parser.writer.emitOp(.Pop);
        try parser.parsePrecedence(.And);
        try parser.writer.patchJump(endJump);
    }

    // Logical or.
    fn lor(parser: *Parser, _: bool) !void {
        const elseJump = try parser.writer.emitJump(.JumpIfFalse);
        const endJump = try parser.writer.emitJump(.Jump);
        try parser.writer.patchJump(elseJump);
        try parser.writer.emitOp(.Pop);
        try parser.parsePrecedence(.Or);
        try parser.writer.patchJump(endJump);
    }

    fn string(parser: *Parser, _: bool) !void {
        const source = parser.previous.source;
        try parser.writer.emitString(source[1 .. source.len - 1]);
    }

    fn literal(parser: *Parser, _: bool) !void {
        switch (parser.previous.type) {
            .False => try parser.writer.emitOp(.False),
            .Nil => try parser.writer.emitOp(.Nil),
            .True => try parser.writer.emitOp(.True),
            else => unreachable,
        }
    }

    fn grouping(parser: *Parser, _: bool) !void {
        try parser.expression();
        parser.consume(.RightParen, "Expect ')' after expression.");
    }

    fn unary(parser: *Parser, _: bool) !void {
        const typ = parser.previous.type;

        try parser.parsePrecedence(.Unary);

        switch (typ) {
            .Minus => try parser.writer.emitOp(.Negate),
            .Bang => try parser.writer.emitOp(.Not),
            else => {},
        }
    }

    fn binary(parser: *Parser, _: bool) !void {
        const typ = parser.previous.type;
        const rule = getRule(typ);
        try parser.parsePrecedence(rule.prec.next());

        switch (typ) {
            .Plus => try parser.writer.emitOp(.Add),
            .Minus => try parser.writer.emitOp(.Subtract),
            .Star => try parser.writer.emitOp(.Multiply),
            .Slash => try parser.writer.emitOp(.Divide),
            .BangEqual => try parser.writer.emitOps(.Equal, .Not),
            .EqualEqual => try parser.writer.emitOp(.Equal),
            .Greater => try parser.writer.emitOp(.Greater),
            .GreaterEqual => try parser.writer.emitOps(.Less, .Not),
            .Less => try parser.writer.emitOp(.Less),
            .LessEqual => try parser.writer.emitOps(.Greater, .Not),
            else => {},
        }
    }

    fn parsePrecedence(parser: *Parser, prec: Precedence) !void {
        const canAssign = prec.lowerOrEqualTo(.Assignment);
        parser.advance();
        const rule = getRule(parser.previous.type);
        if (rule.prefix) |prefix| {
            try prefix(parser, canAssign);

            // TODO: The book suggests we can error here, but I get this:
            //   [line 1] Error at '=': Expect ';' after value.

            // if (canAssign and parser.match(.Equal)) {
            //     parser.errorAt(parser.previous, "Invalid assignment target.");
            // }
        } else {
            parser.errorAt(parser.previous, "Expect expression.");
            return;
        }

        while (@intFromEnum(prec) <= @intFromEnum(getRule(parser.current.type).prec)) {
            parser.advance();
            if (getRule(parser.previous.type).infix) |infix| try infix(parser, canAssign);
        }
    }

    fn identifierConstant(parser: *Parser, token: Token) !u8 {
        return try parser.writer.makeString(token.source);
    }

    fn parseVariable(parser: *Parser, err: []const u8) !u8 {
        parser.consume(.Identifier, err);

        try parser.declareVariable();
        if (parser.compiler.scopeDepth > 0) return 0;

        return try parser.identifierConstant(parser.previous);
    }

    fn declareVariable(parser: *Parser) !void {
        if (parser.compiler.scopeDepth == 0) return;

        if (parser.compiler.definesInSubscope(parser.previous)) {
            parser.errorAt(parser.previous, "Already a variable with this name in this scope.");
        }

        try parser.compiler.addLocal(parser.previous);
    }

    fn expression(parser: *Parser) !void {
        try parsePrecedence(parser, .Assignment);
    }

    fn printStatement(parser: *Parser) !void {
        try parser.expression();
        parser.consume(.Semicolon, "Expect ';' after value.");
        try parser.writer.emitOp(.Print);
    }

    fn returnStatement(parser: *Parser) !void {
        if (parser.compiler.writer.fnType == .Script) {
            parser.errorAt(parser.previous, "Can't return from top-level code.");
        }

        if (parser.check(.Semicolon)) {
            try parser.emitReturn();
        } else {
            try parser.expression();
            parser.consume(.Semicolon, "Expect ';' after return value.");
            try parser.writer.emitOp(.Return);
        }
    }

    fn ifStatement(parser: *Parser) ParseError!void {
        parser.consume(.LeftParen, "Expect '(' after 'if'.");
        try parser.expression();
        parser.consume(.RightParen, "Expect ')' after condition.");

        // Jump to the else branch if the condition is false.
        const thenJump = try parser.writer.emitJump(.JumpIfFalse);
        try parser.writer.emitOp(.Pop);
        try parser.statement();

        const elseJump = try parser.writer.emitJump(.Jump);
        try parser.writer.emitOp(.Pop);

        try parser.writer.patchJump(thenJump);

        if (parser.match(.Else)) {
            try parser.statement();
        }
        try parser.writer.patchJump(elseJump);
    }

    fn whileStatement(parser: *Parser) ParseError!void {
        const loopStart = parser.writer.code.items.len;

        parser.consume(.LeftParen, "Expect '(' after 'while'.");
        try parser.expression();
        parser.consume(.RightParen, "Expect ')' after condition.");

        const exitJump = try parser.writer.emitJump(.JumpIfFalse);
        try parser.writer.emitOp(.Pop);
        try parser.statement();
        try parser.writer.emitLoop(loopStart);
        try parser.writer.patchJump(exitJump);
        try parser.writer.emitOp(.Pop);
    }

    fn forStatement(parser: *Parser) ParseError!void {
        parser.beginScope();
        parser.consume(.LeftParen, "Expect '(' after 'for'.");

        if (parser.match(.Semicolon)) {
            // No initializer.
        } else if (parser.match(.Var)) {
            try parser.varDeclaration();
        } else {
            try parser.expressionStatement();
        }

        var loopStart = parser.writer.code.items.len;

        var exitJump: usize = 0;
        if (!parser.match(.Semicolon)) {
            try parser.expression();
            parser.consume(.Semicolon, "Expect ';' after loop condition.");
            exitJump = try parser.writer.emitJump(.JumpIfFalse);
            try parser.writer.emitOp(.Pop);
        }

        if (!parser.match(.RightParen)) {
            const bodyJump = try parser.writer.emitJump(.Jump);
            const incrementStart = parser.writer.code.items.len;
            try parser.expression();
            try parser.writer.emitOp(.Pop);
            parser.consume(.RightParen, "Expect ')' after for clauses.");
            try parser.writer.emitLoop(loopStart);
            loopStart = incrementStart;
            try parser.writer.patchJump(bodyJump);
        }

        try parser.statement();
        try parser.writer.emitLoop(loopStart);

        if (exitJump != 0) {
            try parser.writer.patchJump(exitJump);
            try parser.writer.emitOp(.Pop);
        }

        try parser.endScope();
    }

    fn expressionStatement(parser: *Parser) !void {
        try parser.expression();
        parser.consume(.Semicolon, "Expect ';' after value.");
        try parser.writer.emitOp(.Pop);
    }

    fn block(parser: *Parser) !void {
        while (!parser.check(.RightBrace) and !parser.check(.Eof)) {
            try parser.declaration();
        }
        parser.consume(.RightBrace, "Expect '}' after block.");
    }

    fn beginScope(parser: *Parser) void {
        parser.compiler.scopeDepth += 1;
    }

    fn endScope(parser: *Parser) !void {
        var localCount = parser.compiler.pop();

        while (localCount > 0) {
            localCount -= 1;
            if (parser.compiler.locals[localCount].isCaptured) {
                try parser.writer.emitOp(.CloseUpvalue);
            } else {
                try parser.writer.emitOp(.Pop);
            }
        }
    }

    fn statement(parser: *Parser) !void {
        if (parser.match(.LeftBrace)) {
            parser.beginScope();
            try parser.block();
            try parser.endScope();
        } else if (parser.match(.Print)) {
            try parser.printStatement();
        } else if (parser.match(.If)) {
            try parser.ifStatement();
        } else if (parser.match(.For)) {
            try parser.forStatement();
        } else if (parser.match(.Return)) {
            try parser.returnStatement();
        } else if (parser.match(.While)) {
            try parser.whileStatement();
        } else {
            try parser.expressionStatement();
        }
    }

    fn funDeclaration(parser: *Parser) !void {
        const global = try parser.parseVariable("Expect function name.");
        parser.compiler.markInitialized();
        try parser.function(.Function);
        try parser.defineVariable(global);
    }

    fn classDeclaration(parser: *Parser) !void {
        parser.consume(.Identifier, "Expect class name.");

        const nameConstant = try parser.identifierConstant(parser.previous);
        try parser.declareVariable();

        try parser.compiler.writer.emitOp(.Class);
        try parser.compiler.writer.emit(nameConstant);
        try parser.defineVariable(nameConstant);

        parser.consume(.LeftBrace, "Expect '{' before class body.");
        parser.consume(.RightBrace, "Expect '}' after class body.");
    }

    fn function(parser: *Parser, funTy: FunctionType) !void {
        var compiler = Compiler.init(parser.alloc, funTy);
        compiler.enclosing = parser.compiler;
        defer compiler.deinit();

        const nameBytes = parser.previous.source;

        // Restore the previous compiler and writer after we're done, since
        // functions can be nested.
        const lastCompiler = parser.compiler;
        const lastWriter = parser.writer;
        parser.compiler = &compiler;
        parser.writer = &compiler.writer;

        parser.beginScope();

        var arity: u8 = 0;
        parser.consume(.LeftParen, "Expect '(' after function name.");
        if (!parser.check(.RightParen)) {
            while (true) {
                arity += 1;
                if (arity == MAX_LOCALS) {
                    parser.errorAt(parser.current, "Can't have more than 128 parameters.");
                }

                const id = try parser.parseVariable("Expect parameter name.");
                try parser.defineVariable(id);

                if (!parser.match(.Comma)) break;
            }
        }
        parser.consume(.RightParen, "Expect ')' after parameters.");
        parser.consume(.LeftBrace, "Expect '{' before function body.");
        try parser.block();

        try parser.emitReturn();

        var func = try parser.writer.func();
        func.arity = arity;
        errdefer func.deinit(parser.alloc);

        func.name = try String.init(parser.alloc, nameBytes);

        parser.compiler = lastCompiler;
        parser.writer = lastWriter;

        if (comptime debug.PrintBytecode) {
            try func.debugDisassemble();
        }

        const obj = try parser.alloc.create(Object);
        errdefer parser.alloc.destroy(obj);
        obj.* = Object{ .function = func };
        const index = try parser.writer.makeConstant(Value{ .obj = obj });
        try parser.writer.emitOp(.Closure);
        try parser.writer.emit(index);

        for (0..compiler.writer.upvalueCount) |i| {
            try parser.writer.emit(if (compiler.upvalues[i].isLocal) 1 else 0);
            try parser.writer.emit(compiler.upvalues[i].index);
        }
    }

    fn emitReturn(parser: *Parser) !void {
        try parser.writer.emitOp(.Nil);
        try parser.writer.emitOp(.Return);
    }

    fn varDeclaration(parser: *Parser) !void {
        const global = try parser.parseVariable("Expect variable name.");

        if (parser.match(.Equal)) {
            try parser.expression();
        } else {
            try parser.writer.emitOp(.Nil);
        }
        parser.consume(.Semicolon, "Expect ';' after variable declaration.");

        try parser.defineVariable(global);
    }

    fn defineVariable(parser: *Parser, global: u8) !void {
        if (parser.compiler.scopeDepth > 0) {
            parser.compiler.markInitialized();
            return;
        }

        try parser.writer.emitOp(.DefineGlobal);
        try parser.writer.emit(global);
    }

    fn declaration(parser: *Parser) ParseError!void {
        if (parser.match(.Var)) {
            try parser.varDeclaration();
        } else if (parser.match(.Fun)) {
            try parser.funDeclaration();
        } else if (parser.match(.Class)) {
            try parser.classDeclaration();
        } else {
            try parser.statement();
        }

        // Recover from panic.
        if (parser.panicMode) parser.synchronize();
    }

    fn synchronize(parser: *Parser) void {
        parser.panicMode = false;

        while (!parser.check(.Eof)) {
            if (parser.previous.type == .Semicolon) return;
            switch (parser.current.type) {
                .Class, .Fun, .Var, .For, .If, .While, .Print, .Return => return,
                else => {},
            }

            parser.advance();
        }
    }

    fn variable(parser: *Parser, canAssign: bool) !void {
        return parser.namedVariable(parser.previous, canAssign);
    }

    fn namedVariable(parser: *Parser, token: Token, canAssign: bool) !void {
        var get: OpCode = .GetGlobal;
        var set: OpCode = .SetGlobal;
        var index: u8 = 0;

        if (parser.compiler.resolveLocal(token) catch {
            parser.errorAt(token, "Can't read local variable in its own initializer.");
            return;
        }) |i| {
            index = i;
            get = .GetLocal;
            set = .SetLocal;
        } else if (parser.compiler.resolveUpvalue(token) catch {
            parser.errorAt(token, "Can't read upvalue in its own initializer.");
            return;
        }) |i| {
            index = i;
            get = .GetUpvalue;
            set = .SetUpvalue;
        } else {
            index = try parser.identifierConstant(token);
        }

        if (canAssign and parser.match(.Equal)) {
            try parser.expression();
            try parser.writer.emitOp(set);
        } else {
            try parser.writer.emitOp(get);
        }
        try parser.writer.emit(index);
    }

    const ParseFn = *const fn (parser: *Parser, canAssign: bool) ParseError!void;

    const Rule = struct {
        prec: Precedence = .None,
        prefix: ?ParseFn = null,
        infix: ?ParseFn = null,
    };

    fn getRule(typ: TokenType) Rule {
        return switch (typ) {
            .LeftParen => Rule{ .prec = .Call, .prefix = grouping, .infix = call },
            .Number => Rule{ .prec = .Term, .prefix = number },
            .And => Rule{ .prec = .And, .infix = land },
            .Or => Rule{ .prec = .And, .infix = lor },
            .String => Rule{ .prefix = string },
            .Identifier => Rule{ .prefix = variable },
            .Plus => Rule{ .prec = .Term, .infix = binary },
            .Minus => Rule{ .prec = .Term, .prefix = unary, .infix = binary },
            .Dot => Rule{ .prec = .Call, .infix = dot },
            .Star => Rule{ .prec = .Factor, .infix = binary },
            .Slash => Rule{ .prec = .Factor, .infix = binary },
            .False => Rule{ .prefix = literal },
            .True => Rule{ .prefix = literal },
            .Nil => Rule{ .prefix = literal },
            .Bang => Rule{ .prefix = unary },
            .Del => Rule{ .prefix = del },
            .BangEqual => Rule{ .prec = .Equality, .infix = binary },
            .EqualEqual => Rule{ .prec = .Comparison, .infix = binary },
            .Greater => Rule{ .prec = .Comparison, .infix = binary },
            .GreaterEqual => Rule{ .prec = .Comparison, .infix = binary },
            .Less => Rule{ .prec = .Comparison, .infix = binary },
            .LessEqual => Rule{ .prec = .Comparison, .infix = binary },
            else => Rule{},
        };
    }

    /// Compiles the source code into a function.
    /// The lifetime of the function is tied to the parser.
    fn compile(parser: *Parser) !Function {
        parser.advance();

        while (!parser.match(.Eof)) {
            try parser.declaration();
        }

        if (parser.hadError) {
            return error.ParseError;
        }

        try parser.emitReturn();

        return parser.writer.func();
    }
};

// GC metadata that we can access from the allocator hooks. We use this to
// track how many bytes we've allocated, and trigger GC when we hit a certain
// threshold.
const GcMeta = struct {
    vm: *VM,
    alloc: std.mem.Allocator,
    bytesAllocated: isize = 0,
    nextGc: isize = 1024 * 1024,

    fn init(vm: *VM) GcMeta {
        return GcMeta{ .vm = vm, .alloc = vm.alloc };
    }
};

const gcVtable = struct {
    const vtable = std.mem.Allocator.VTable{
        .alloc = gcAlloc,
        .resize = gcResize,
        .remap = gcRemap,
        .free = gcFree,
    };

    fn gc(self: *GcMeta) !void {
        const before = self.bytesAllocated;
        self.vm.gc() catch {};
        if (comptime debug.LogGC) {
            std.debug.print("GC: collected {d} bytes, {d} remaining\n", .{ before - self.bytesAllocated, self.bytesAllocated });
        }
    }

    fn gcAlloc(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        // std.debug.print("Allocating {d} bytes\n", .{len});
        var meta: *GcMeta = @ptrCast(@alignCast(ptr));
        const res = meta.alloc.vtable.alloc(meta.alloc.ptr, len, alignment, ret_addr);
        if (res) |_| {
            meta.bytesAllocated += @intCast(len);
            if (meta.bytesAllocated > meta.nextGc) {
                meta.vm.gc() catch {};
                meta.nextGc = meta.bytesAllocated * 2;
            }
        }
        return res;
    }

    fn gcResize(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        var meta: *GcMeta = @ptrCast(@alignCast(ptr));
        // if (new_len > memory.len) {
        //     triggerGC(ptr);
        // }
        return meta.alloc.vtable.resize(meta.alloc.ptr, memory, alignment, new_len, ret_addr);
    }

    fn gcRemap(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        var meta: *GcMeta = @ptrCast(@alignCast(ptr));
        // if (new_len > memory.len) {
        //     triggerGC(ptr);
        // }
        return meta.alloc.vtable.remap(meta.alloc.ptr, memory, alignment, new_len, ret_addr);
    }

    fn gcFree(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        var meta: *GcMeta = @ptrCast(@alignCast(ptr));
        meta.alloc.vtable.free(meta.alloc.ptr, memory, alignment, ret_addr);
    }
}.vtable;

/// A function to test the bytecode output for an input string.
fn bytecode(source: []const u8, expected_bytecode: []const u8) !void {
    const stderr = std.Io.File.stderr();
    var buf: [1024]u8 = undefined;
    var writer = stderr.writer(std.testing.io, &buf);

    const alloc = std.testing.allocator;

    var parser = try Parser.init(alloc, &writer.interface, source);
    defer parser.deinit();

    const func = try parser.compile();
    defer func.deinit(alloc);

    var output = std.Io.Writer.Allocating.init(alloc);
    defer output.deinit();
    const output_writer = &output.writer;

    try func.disassemble(output_writer);

    try std.testing.expectEqualStrings(expected_bytecode, output.written());
}

/// A function to test the parse output for an input string.
fn parse_error(source: []const u8, expected_error: []const u8) !void {
    const alloc = std.testing.allocator;
    var output = std.Io.Writer.Allocating.init(alloc);
    defer output.deinit();
    const output_writer = &output.writer;

    var parser = try Parser.init(alloc, output_writer, source);
    defer parser.deinit();

    const func = parser.compile() catch {
        return std.testing.expectEqualStrings(expected_error, output.written());
    };

    std.debug.print("No parse error!", .{});
    func.deinit(alloc);
}

fn run(source: []const u8, expected_output: []const u8) !void {
    return runs(&.{source}, expected_output);
}

/// Evaluates a number of scripts in sequence, asserting on their combined output.
fn runs(source: []const []const u8, expected_output: []const u8) !void {
    const alloc = std.testing.allocator;

    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    const output_writer = &output.writer;
    defer output.deinit();

    var stack: [128]Value = undefined;
    var vm = VM.init(alloc, output_writer, &stack);
    defer vm.deinit();

    // Hooks into the GC system so we can track allocations and trigger GC when needed.
    var gcMeta = GcMeta.init(&vm);
    vm.alloc = .{
        .ptr = &gcMeta,
        .vtable = &gcVtable,
    };

    // Hacky, but works. We can't do this in init, since the VM struct isn't
    // yet in its final memory location.
    if (comptime debug.StressGC) {

        // If we're stress testing the GC, we need to use a custom allocator
        // that calls gc() before every allocation. Otherwise, we can just use
        // the standard testing allocator.
        vm.alloc = .{
            .ptr = &vm,
            .vtable = &struct {
                const vtable = std.mem.Allocator.VTable{
                    .alloc = stressAlloc,
                    .resize = stressResize,
                    .remap = stressRemap,
                    .free = stressFree,
                };

                fn triggerGC(ptr: *anyopaque) void {
                    @as(*VM, @ptrCast(@alignCast(ptr))).gc() catch {};
                }

                // We call GC before every allocation, so we can stress test it. We
                // don't trigger GC when we free memory, since that risks putting
                // us in an infinite loop.
                fn stressAlloc(ptr: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
                    triggerGC(ptr);
                    return std.testing.allocator.vtable.alloc(std.testing.allocator.ptr, len, alignment, ret_addr);
                }

                fn stressResize(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
                    if (new_len > memory.len) {
                        triggerGC(ptr);
                    }
                    return std.testing.allocator.vtable.resize(std.testing.allocator.ptr, memory, alignment, new_len, ret_addr);
                }

                fn stressRemap(ptr: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
                    if (new_len > memory.len) {
                        triggerGC(ptr);
                    }
                    return std.testing.allocator.vtable.remap(std.testing.allocator.ptr, memory, alignment, new_len, ret_addr);
                }

                fn stressFree(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
                    std.testing.allocator.vtable.free(std.testing.allocator.ptr, memory, alignment, ret_addr);
                }
            }.vtable,
        };
    }

    for (source) |src| {
        try std.testing.expectEqual(.OK, vm.interpret(src));

        // Force a GC pass to run, so we can catch any issues with it.
        try vm.gc();
    }

    try std.testing.expectEqualStrings(expected_output, output.written());
}

test "nothing" {
    try runs(&.{
        \\
        ,
        \\
    },
        \\
    );
}

test "string" {
    try run(
        \\"hello";
    ,
        \\
    );
}

test "re-assign variable" {
    try run(
        \\var a = 1;
        \\a = 3;
        \\print a;
    ,
        \\3
        \\
    );
}

test "print variable" {
    try run(
        \\var beverage = "cafe au lait";
        \\var breakfast = "beignets with " + beverage;
        \\print breakfast;
    ,
        \\beignets with cafe au lait
        \\
    );
}

test "variable scope" {
    try run(
        \\var a = 1;
        \\{
        \\  var a = 2;
        \\  a = 3;
        \\  print a;
        \\}
        \\print a;
    ,
        \\3
        \\1
        \\
    );
}

test "if statement" {
    try run(
        \\if (true) {
        \\  print "less";
        \\} else {
        \\  print "greater";
        \\}
    ,
        \\less
        \\
    );

    try run(
        \\if (false) {
        \\  print "less";
        \\} else {
        \\  print "greater";
        \\}
    ,
        \\greater
        \\
    );
}

test "logical expressions" {
    try run(
        \\print nil or "yes";
        \\print nil and "yes";
        \\print "yes" and "no";
    ,
        \\yes
        \\nil
        \\no
        \\
    );
}

test "while loop" {
    try run(
        \\var i = 0;
        \\while (i < 3) {
        \\  print i;
        \\  i = i + 1;
        \\}
    ,
        \\0
        \\1
        \\2
        \\
    );
}

test "for loop" {
    try run(
        \\for (var i = 0; i < 3; i = i + 1) {
        \\  print i;
        \\}
    ,
        \\0
        \\1
        \\2
        \\
    );
    try run(
        \\for (var i = 0; i < 3;) {
        \\  print i;
        \\  i = i + 1;
        \\}
    ,
        \\0
        \\1
        \\2
        \\
    );
    try parse_error(
        \\for (var i = 0;
    ,
        \\Error at end: Expect expression.
        \\  1: for (var i = 0;
        \\                    ^
        \\
    );
}

test "function" {
    try run(
        \\fun noop(i, g, n, o, r, e, d) {
        \\  print "no-op";
        \\}
        \\print noop;
    ,
        \\<fn noop/7>
        \\
    );
}

test "function call" {
    try run(
        \\fun add(a, b) {
        \\  return a + b;
        \\}
        \\print add(1, 2);
    ,
        \\3
        \\
    );
}

test "interpret twice" {
    try runs(&.{
        \\var a = 1;
        ,
        \\print a;
    },
        \\1
        \\
    );
}

fn frameDepth(vm: *VM, _: []Value) NativeErr!Value {
    return Value{ .float = @floatFromInt(vm.frameCount) };
}

test "native" {
    const alloc = std.testing.allocator;

    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    const output_writer = &output.writer;
    defer output.deinit();

    var stack: [128]Value = undefined;
    var vm = VM.init(alloc, output_writer, &stack);
    defer vm.deinit();

    try vm.defineNative("frameDepth", frameDepth);

    try std.testing.expectEqual(.OK, vm.interpret(
        \\fun wrapper() {
        \\  print frameDepth();
        \\}
        \\wrapper();
    ));

    try std.testing.expectEqualStrings(
        \\2
        \\
    , output.written());
}

test "read outer scope upvalue" {
    try run(
        \\ var x = "global";
        \\ fun outer() {
        \\   var x = "outer";
        \\   fun inner() {
        \\     print x;
        \\   }
        \\   inner();
        \\ }
        \\ outer();
    ,
        \\outer
        \\
    );
}

test "assign upvalue" {
    try run(
        \\ fun outer() {
        \\   var x = "before";
        \\   fun inner() {
        \\     x = "assigned";
        \\   }
        \\   inner();
        \\   print x;
        \\ }
        \\ outer();
    ,
        \\assigned
        \\
    );
}

test "closed upvalue" {
    try run(
        \\ fun outer() {
        \\   var x = "outside";
        \\   fun inner() {
        \\     print x;
        \\   }
        \\
        \\   return inner;
        \\ }
        \\ var closure = outer();
        \\ closure();
    ,
        \\outside
        \\
    );
}

test "get/set upvalue" {
    try run(
        \\ var globalSet;
        \\ var globalGet;
        \\
        \\ fun main() {
        \\   var a = "initial";
        \\
        \\   fun set() { a = "updated"; }
        \\   fun get() { print a; }
        \\
        \\   globalSet = set;
        \\   globalGet = get;
        \\ }
        \\main();
        \\globalSet();
        \\globalGet();
    ,
        \\updated
        \\
    );
}

test "closure" {
    try run(
        \\fun makeCounter() {
        \\  var i = 0;
        \\  fun count() {
        \\    i = i + 1;
        \\    print i;
        \\  }
        \\  return count;
        \\}
        \\
        \\var counter = makeCounter();
        \\counter();
        \\counter();
    ,
        \\1
        \\2
        \\
    );
}

test "lots of as" {
    try run(
        \\var a = "a";
        \\a = "a";
        \\print a;
    ,
        \\a
        \\
    );
}

test "loop closures" {
    try run(
        \\ var globalOne;
        \\ var globalTwo;
        \\
        \\ fun main() {
        \\   for (var a = 1; a <= 2; a = a + 1) {
        \\     fun closure() {
        \\       print a;
        \\     }
        \\     if (globalOne == nil) {
        \\       globalOne = closure;
        \\     } else {
        \\       globalTwo = closure;
        \\     }
        \\   }
        \\ }
        \\
        \\ main();
        \\ globalOne();
        \\ globalTwo();
    ,
        \\3
        \\3
        \\
    );
}

test "print the same" {
    try run(
        \\var a = "a";
        \\print a;
        \\print a;
    ,
        \\a
        \\a
        \\
    );
}

test "persist functions" {
    const alloc = std.testing.allocator;

    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    const output_writer = &output.writer;
    defer output.deinit();

    var stack: [128]Value = undefined;
    var vm = VM.init(alloc, output_writer, &stack);
    defer vm.deinit();

    try std.testing.expectEqual(.OK, vm.interpret(
        \\fun spooky() {
        \\  print "boo!";
        \\}
    ));

    vm.gc() catch {};

    try std.testing.expectEqual(.OK, vm.interpret(
        \\spooky();
    ));

    try std.testing.expectEqualStrings(
        \\boo!
        \\
    , output.written());
}

test "pair" {
    try run(
        \\class Pair {}
        \\
        \\var pair = Pair();
        \\pair.first = 1;
        \\pair.second = 2;
        \\print pair.first + pair.second; // 3.
    ,
        \\3
        \\
    );
}

test "set and get property" {
    try bytecode(
        \\class A {}
        \\var a = A();
        \\a.foo = 1;
        \\a.foo;
    ,
        \\== <script> ==
        \\0000 Class          0
        \\0002 DefineGlobal   0 A
        \\0004 GetGlobal      0 A
        \\0006 Call           0
        \\0008 DefineGlobal   1 a
        \\0010 GetGlobal      1 a
        \\0012 Constant       3 1
        \\0014 SetProperty    2
        \\0016 Pop
        \\0017 GetGlobal      1 a
        \\0019 GetProperty    2
        \\0021 Pop
        \\0022 Nil
        \\0023 Return
        \\
    );
    try parse_error(
        \\a.
    ,
        \\Error at end: Expect property name after '.'.
        \\  1: a.
        \\       ^
        \\
    );
}

test "del property" {
    try bytecode(
        \\del a.b;
    ,
        \\== <script> ==
        \\0000 GetGlobal      0 a
        \\0002 DelProperty    1
        \\0004 Pop
        \\0005 Nil
        \\0006 Return
        \\
    );
    try run(
        \\class A {}
        \\var a = A();
        \\print del a.b;
    ,
        \\false
        \\
    );
    try parse_error(
        \\
        \\del a();
    ,
        \\Error at ')': Requires a variable access!
        \\  2: del a();
        \\           ^
        \\
    );
}

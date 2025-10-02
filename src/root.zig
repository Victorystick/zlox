//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

const ValueType = enum {
    nil,
    bool,
    float,
    obj,
};
const Value = union(ValueType) {
    nil,
    bool: bool,
    float: f64,
    obj: *Object,

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
                // else => false,
            },
            else => false,
        };
    }
};

const String = struct {
    chars: []const u8,
    hash: u64,
};
const StringHashing = struct {
    pub fn hash(self: @This(), s: String) u64 {
        _ = self;
        return s.hash;
    }
    pub fn eql(self: @This(), a: String, b: String) bool {
        _ = self;
        return std.mem.eql(u8, a.chars, b.chars);
    }
};

fn StringMap(comptime V: type) type {
    return std.hash_map.HashMapUnmanaged(String, V, StringHashing, 80);
}

const ObjectType = enum {
    string,
};
const Object = union(ObjectType) {
    string: String,

    pub fn format(self: Object, writer: *std.Io.Writer) !void {
        try switch (self) {
            .string => |string| writer.writeAll(string.chars),
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
    _,
};

const Chunk = struct {
    code: []const u8,
    constants: []const Value,

    pub fn disassemble(chunk: *const Chunk, name: []const u8, io: *std.Io.Writer) !void {
        try io.print("== {s} ==\n", .{name});

        var offset: usize = 0;
        while (offset < chunk.code.len) {
            offset = try disassembleInstruction(chunk, offset, io);
        }
    }
};

fn disassembleInstruction(chunk: *const Chunk, offset: usize, io: *std.Io.Writer) !usize {
    try io.print("{d:0>4} ", .{offset});

    return switch (@as(OpCode, @enumFromInt(chunk.code[offset]))) {
        .Constant => {
            const index = chunk.code[offset + 1];
            const constant = chunk.constants[index];
            try io.print("{s} {d: >3} {f}\n", .{ "Constant", index, constant });
            return offset + 2;
        },
        .Return => {
            try io.writeAll("Return\n");
            return offset + 1;
        },
        .Negate => {
            try io.writeAll("Negate\n");
            return offset + 1;
        },
        .Add => {
            try io.writeAll("Add\n");
            return offset + 1;
        },
        .Subtract => {
            try io.writeAll("Subtract\n");
            return offset + 1;
        },
        .Multiply => {
            try io.writeAll("Multiply\n");
            return offset + 1;
        },
        .Divide => {
            try io.writeAll("Divide\n");
            return offset + 1;
        },
        .False => {
            try io.writeAll("False\n");
            return offset + 1;
        },
        .True => {
            try io.writeAll("True\n");
            return offset + 1;
        },
        .Nil => {
            try io.writeAll("Nil\n");
            return offset + 1;
        },
        .Not => {
            try io.writeAll("Not\n");
            return offset + 1;
        },
        .Equal => {
            try io.writeAll("Equal\n");
            return offset + 1;
        },
        .Greater => {
            try io.writeAll("Greater\n");
            return offset + 1;
        },
        .Less => {
            try io.writeAll("Less\n");
            return offset + 1;
        },
        _ => {
            try io.print("Unknown opcode {d}\n", .{chunk.code[offset]});
            return offset + 1;
        },
    };
}

pub const ChunkWriter = struct {
    gpa: std.mem.Allocator,
    code: std.ArrayList(u8) = .empty,
    constants: std.ArrayList(Value) = .empty,
    // Objects owned by the chunk, which share its lifetime.
    objects: std.ArrayList(Object) = .empty,

    /// Deinitialize with `deinit` or use `toOwnedSlice`.
    pub fn init(gpa: std.mem.Allocator) ChunkWriter {
        return ChunkWriter{
            .gpa = gpa,
        };
    }

    /// Release all allocated memory.
    pub fn deinit(self: *ChunkWriter) void {
        self.code.deinit(self.gpa);
        self.objects.deinit(self.gpa);
        self.constants.deinit(self.gpa);
    }

    /// Valid for the lifetime of the ChunkWriter.
    pub fn chunk(self: *const ChunkWriter) Chunk {
        return Chunk{
            .code = self.code.items,
            .constants = self.constants.items,
        };
    }

    pub fn emit(self: *ChunkWriter, byte: u8) !void {
        try self.code.append(self.gpa, byte);
    }

    pub fn emitOp(self: *ChunkWriter, op: OpCode) !void {
        try self.emit(@as(u8, @intFromEnum(op)));
    }

    pub fn emitOps(self: *ChunkWriter, op1: OpCode, op2: OpCode) !void {
        try self.emitOp(op1);
        try self.emitOp(op2);
    }

    pub fn emitString(self: *ChunkWriter, str: []const u8) !void {
        const new_item_ptr = try self.objects.addOne(self.gpa);
        new_item_ptr.* = Object{ .string = String{
            .chars = str,
            .hash = std.hash_map.hashString(str),
        } };
        try self.emitConstant(Value{ .obj = new_item_ptr });
    }

    pub fn emitConstant(self: *ChunkWriter, val: Value) !void {
        // var index = std.mem.indexOfScalar(self.constants.items, val);
        const index = self.constants.items.len;
        try self.constants.append(self.gpa, val);
        try self.emitOp(.Constant);
        try self.emit(@intCast(index));
    }
};

test "disassemble" {
    var allocating_writer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer allocating_writer.deinit();

    var values = [_]Value{Value{ .float = 1 }};
    const chunk = Chunk{ .code = &.{ 0, 1, 0, 2 }, .constants = &values };
    try chunk.disassemble("test chunk", &allocating_writer.writer);

    try std.testing.expectEqualStrings(
        \\== test chunk ==
        \\0000 Return
        \\0001 Constant   0 1
        \\0003 Negate
        \\
    ,
        allocating_writer.written(),
    );
}

pub fn interpretChunk(alloc: std.mem.Allocator, chunk: *const Chunk, io: *std.Io.Writer) !VM.Result {
    var stack: [64]Value = undefined;
    var vm = VM{ .alloc = alloc, .io = io, .chunk = chunk, .ip = 0, .stack = &stack };
    defer vm.deinit();
    return vm.run();
}

const debugTraceExecution = true;

pub const VM = struct {
    alloc: std.mem.Allocator,
    io: *std.Io.Writer,
    chunk: *const Chunk,
    ip: usize,
    stack: []Value,
    stackTop: usize = 0,

    // Interned strings.
    strings: StringMap(void) = .empty,

    // A linked list of all allocated objects.
    objects: ?*ObjectNode = null,

    const ObjectNode = struct {
        next: ?*ObjectNode,
        data: Object,
    };

    const Result = enum { OK, CompileError, RuntimeError };

    fn deinit(vm: *VM) void {
        var keys = vm.strings.keyIterator();
        while (keys.next()) |key| {
            vm.alloc.free(key.chars);
        }
        vm.strings.deinit(vm.alloc);

        while (vm.objects) |n| {
            vm.objects = n.next;
            vm.alloc.destroy(n);
        }
    }

    fn obj(vm: *VM, data: Object) !*Object {
        const node = try vm.alloc.create(ObjectNode);
        node.* = ObjectNode{ .next = vm.objects, .data = data };
        vm.objects = node;
        return &node.data;
    }

    fn run(vm: *VM) !Result {
        while (true) {
            if (comptime debugTraceExecution) {
                try vm.io.writeAll("[ ");
                for (vm.stack[0..vm.stackTop]) |val| {
                    try vm.io.print("{f} ", .{val});
                }
                try vm.io.writeAll("]\n");
                _ = disassembleInstruction(vm.chunk, vm.ip, vm.io) catch {};
                try vm.io.flush();
            }

            switch (vm.op()) {
                .Constant => {
                    const constant = vm.chunk.constants[vm.byte()];

                    switch (constant) {
                        .obj => |ref| {
                            switch (ref.*) {
                                .string => |str| {
                                    // If we've already interned the string, use that instance.
                                    if (vm.strings.getKey(str)) |string| {
                                        try vm.pushObj(Object{ .string = string });
                                    } else {
                                        // Otherwise, allocate a new one.
                                        const chars = try vm.alloc.alloc(u8, str.chars.len);
                                        errdefer vm.alloc.free(chars);
                                        @memcpy(chars, str.chars);
                                        const string = String{ .chars = chars, .hash = str.hash };
                                        try vm.strings.put(vm.alloc, string, undefined);
                                        try vm.pushObj(Object{ .string = string });
                                    }
                                },
                            }
                        },
                        else => vm.push(constant),
                    }
                },
                .Nil => vm.push(.nil),
                .True => vm.push(Value{ .bool = true }),
                .False => vm.push(Value{ .bool = false }),
                .Not => {
                    const val = vm.pop();
                    vm.push(Value{ .bool = val.isFalsey() });
                },
                .Return => {
                    try vm.io.print("Return {f}\n", .{vm.pop()});
                    try vm.io.flush();
                    return .OK;
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
            }
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
        const b = vm.pop();
        const a = vm.pop();

        const first = a.obj.*.string.chars;
        const second = b.obj.*.string.chars;
        var result = try vm.alloc.alloc(u8, first.len + second.len);
        errdefer vm.alloc.free(result);

        @memcpy(result[0..first.len], first);
        @memcpy(result[first.len..], second);

        const string = String{
            .chars = result,
            .hash = std.hash_map.hashString(result),
        };

        const entry = try vm.strings.getOrPut(vm.alloc, string);
        if (entry.found_existing) {
            // Free old allocation.
            std.debug.print("Found string {s}!\n", .{entry.key_ptr.*.chars});
            vm.alloc.free(result);
        }
        try vm.pushObj(Object{ .string = entry.key_ptr.* });
    }

    /// Returns the next opcode.
    fn op(vm: *VM) OpCode {
        return @as(OpCode, @enumFromInt(vm.byte()));
    }

    fn byte(vm: *VM) u8 {
        const val = vm.chunk.code[vm.ip];
        vm.ip += 1;
        return val;
    }

    fn peek(vm: *VM, distance: usize) Value {
        return vm.stack[vm.stackTop - (distance + 1)];
    }

    fn push(vm: *VM, val: Value) void {
        vm.stack[vm.stackTop] = val;
        vm.stackTop += 1;
    }

    fn pushObj(vm: *VM, object: Object) !void {
        vm.push(Value{ .obj = try vm.obj(object) });
    }

    fn pop(vm: *VM) Value {
        vm.stackTop -= 1;
        return vm.stack[vm.stackTop];
    }
};

test "interpretChunk" {
    var values = [_]Value{Value{ .float = 1 }};
    const chunk = Chunk{ .code = &.{ 1, 0, 2, 0 }, .constants = &values };

    var buffer: [128]u8 = undefined;
    const file = std.fs.File.stderr();
    var writer = std.fs.File.writer(file, &buffer);

    try std.testing.expectEqual(.OK, interpretChunk(std.testing.allocator, &chunk, &writer.interface));
}

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

    Error,
    Eof,
};

const Token = struct {
    type: TokenType,
    source: []const u8,
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
        return Token{ .type = .Error, .source = text, .line = scanner.line };
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
            .line = scanner.line,
        };
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

const Parser = struct {
    scanner: Scanner,
    writer: *ChunkWriter,
    current: Token = Token{
        .type = .Error,
        .source = "No parse\n",
        .line = 0,
    },
    previous: Token = undefined,
    hadError: bool = false,
    panicMode: bool = false,

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
            // std.debug.print("Advance: {s}\n", .{parser.current.source});
            if (parser.current.type != .Error) break;

            parser.errorAt(parser.current, "BOOM!");
        }
    }

    fn errorAt(parser: *Parser, token: Token, text: []const u8) void {
        if (parser.panicMode) return;
        parser.panicMode = true;
        std.debug.print("[line {d}] Error", .{token.line});

        switch (token.type) {
            .Eof => std.debug.print(" at end", .{}),
            .Error => {},
            else => std.debug.print(" at '{s}'", .{token.source}),
        }

        std.debug.print(": {s}\n", .{text});
        parser.hadError = true;
    }

    fn consume(parser: *Parser, typ: TokenType, text: []const u8) void {
        if (parser.current.type == typ) {
            parser.advance();
        } else {
            parser.errorAt(parser.current, text);
        }
    }

    fn number(parser: *Parser) !void {
        const val = try std.fmt.parseFloat(f64, parser.previous.source);
        try parser.writer.emitConstant(Value{ .float = val });
    }

    fn string(parser: *Parser) !void {
        const source = parser.previous.source;
        try parser.writer.emitString(source[1 .. source.len - 1]);
    }

    fn literal(parser: *Parser) !void {
        switch (parser.previous.type) {
            .False => try parser.writer.emitOp(.False),
            .Nil => try parser.writer.emitOp(.Nil),
            .True => try parser.writer.emitOp(.True),
            else => unreachable,
        }
    }

    fn grouping(parser: *Parser) !void {
        try parser.expression();
        parser.consume(.RightParen, "Expect ')' after expression.");
    }

    fn unary(parser: *Parser) !void {
        const typ = parser.previous.type;

        try parser.parsePrecedence(.Unary);

        switch (typ) {
            .Minus => try parser.writer.emitOp(.Negate),
            .Bang => try parser.writer.emitOp(.Not),
            else => {},
        }
    }

    fn binary(parser: *Parser) !void {
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
        parser.advance();
        const rule = getRule(parser.previous.type);
        if (rule.prefix) |prefix| {
            try prefix(parser);
        } else {
            parser.errorAt(parser.previous, "Expect expression.");
            return;
        }

        while (@intFromEnum(prec) <= @intFromEnum(getRule(parser.current.type).prec)) {
            parser.advance();
            if (getRule(parser.previous.type).infix) |infix| try infix(parser);
        }
    }

    fn expression(parser: *Parser) !void {
        try parsePrecedence(parser, .Assignment);
    }

    const ParseFn = *const fn (parser: *Parser) error{ InvalidCharacter, OutOfMemory }!void;

    const Rule = struct {
        prec: Precedence = .None,
        prefix: ?ParseFn = null,
        infix: ?ParseFn = null,
    };

    fn getRule(typ: TokenType) Rule {
        return switch (typ) {
            .LeftParen => Rule{ .prefix = grouping },
            .Number => Rule{ .prec = .Term, .prefix = number },
            .String => Rule{ .prefix = string },
            .Plus => Rule{ .prec = .Term, .infix = binary },
            .Minus => Rule{ .prec = .Term, .prefix = unary, .infix = binary },
            .Star => Rule{ .prec = .Factor, .infix = binary },
            .Slash => Rule{ .prec = .Factor, .infix = binary },
            .False => Rule{ .prefix = literal },
            .True => Rule{ .prefix = literal },
            .Nil => Rule{ .prefix = literal },
            .Bang => Rule{ .prefix = unary },
            .BangEqual => Rule{ .prec = .Equality, .infix = binary },
            .EqualEqual => Rule{ .prec = .Comparison, .infix = binary },
            .Greater => Rule{ .prec = .Comparison, .infix = binary },
            .GreaterEqual => Rule{ .prec = .Comparison, .infix = binary },
            .Less => Rule{ .prec = .Comparison, .infix = binary },
            .LessEqual => Rule{ .prec = .Comparison, .infix = binary },
            else => Rule{},
        };
    }
};

pub fn compile(source: []const u8, writer: *ChunkWriter) !bool {
    var parser = Parser{ .scanner = Scanner{ .source = source }, .writer = writer };

    parser.advance();
    try parser.expression();
    parser.consume(.Eof, "Expect end of expression.");
    try writer.emitOp(.Return);

    return !parser.hadError;
}

test "compile" {
    const alloc = std.testing.allocator;

    var writer = ChunkWriter.init(alloc);
    defer writer.deinit();

    try std.testing.expect(try compile(
        \\"st" == "s" + "t"
    , &writer));

    var allocating_writer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer allocating_writer.deinit();

    var buffer: [128]u8 = undefined;
    const file = std.fs.File.stderr();
    var stderr = std.fs.File.writer(file, &buffer);

    try std.testing.expectEqual(.OK, interpretChunk(alloc, &writer.chunk(), &stderr.interface));
}

pub fn interpret(source: []const u8, writer: *std.Io.Writer) !void {
    const alloc = std.heap.page_allocator;

    var cWriter = ChunkWriter.init(alloc);
    defer cWriter.deinit();

    if (try compile(source, &cWriter)) {
        _ = try interpretChunk(alloc, &cWriter.chunk(), writer);
    }
}

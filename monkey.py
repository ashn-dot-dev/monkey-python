#!/usr/bin/env python3


class SourceLocation:
    def __init__(self, filename, line):
        self.filename = filename
        self.line = line

    def __str__(self):
        return f"{self.filename}:{self.line}"


class Token:
    # Meta
    ILLEGAL = "ILLEGAL"
    EOF = "EOF"
    # Identifiers + literals
    IDENT = "IDENT"
    INT = "INT"
    # Operators
    ASSIGN = "="
    PLUS = "+"
    MINUS = "-"
    BANG = "!"
    ASTERISK = "*"
    SLASH = "/"
    LT = "<"
    GT = ">"
    EQ = "=="
    NOT_EQ = "!="
    # Delimiters
    COMMA = ","
    SEMICOLON = ";"
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    # Keywords
    FUNCTION = "FUNCTION"
    LET = "LET"
    TRUE = "TRUE"
    FALSE = "FALSE"
    IF = "IF"
    ELSE = "ELSE"
    RETURN = "RETURN"

    @staticmethod
    def lookup_ident(ident):
        keywords = {
            "fn": Token.FUNCTION,
            "let": Token.LET,
            "true": Token.TRUE,
            "false": Token.FALSE,
            "if": Token.IF,
            "else": Token.ELSE,
            "return": Token.RETURN,
        }
        return keywords.get(ident, Token.IDENT)

    def __init__(self, kind, literal, source_location):
        self.kind = kind
        self.literal = literal
        self.source_location = source_location

    def __str__(self):
        if self.kind == Token.IDENT:
            return f"{self.kind}({self.literal})"
        if self.kind == Token.INT:
            return f"{self.kind}({self.literal})"
        return f"{self.kind}"


class Lexer:
    EOF_LITERAL = ""

    def __init__(self, source, filename="<nofile>"):
        self.source = source
        self.source_filename = filename
        self.source_line = 0
        self.position = 0
        self.read_position = 0
        self.ch = None
        self._read_char()

    def next_token(self):
        tok = Token(None, None, None)
        self._skip_whitespace()

        if self.ch == "=":
            if self._peek_char() == "=":
                self._read_char()
                tok = self._new_token(Token.EQ, "==")
            else:
                tok = self._new_token(Token.ASSIGN, self.ch)
        elif self.ch == "+":
            tok = self._new_token(Token.PLUS, self.ch)
        elif self.ch == "-":
            tok = self._new_token(Token.MINUS, self.ch)
        elif self.ch == "!":
            if self._peek_char() == "=":
                self._read_char()
                tok = self._new_token(Token.NOT_EQ, "!=")
            else:
                tok = self._new_token(Token.BANG, self.ch)
        elif self.ch == "/":
            tok = self._new_token(Token.SLASH, self.ch)
        elif self.ch == "*":
            tok = self._new_token(Token.ASTERISK, self.ch)
        elif self.ch == "<":
            tok = self._new_token(Token.LT, self.ch)
        elif self.ch == ">":
            tok = self._new_token(Token.GT, self.ch)
        elif self.ch == ";":
            tok = self._new_token(Token.SEMICOLON, self.ch)
        elif self.ch == ",":
            tok = self._new_token(Token.COMMA, self.ch)
        elif self.ch == "(":
            tok = self._new_token(Token.LPAREN, self.ch)
        elif self.ch == ")":
            tok = self._new_token(Token.RPAREN, self.ch)
        elif self.ch == "{":
            tok = self._new_token(Token.LBRACE, self.ch)
        elif self.ch == "}":
            tok = self._new_token(Token.RBRACE, self.ch)
        elif self.ch == Lexer.EOF_LITERAL:
            tok = self._new_token(Token.EOF, self.ch)
        else:
            if Lexer._is_letter(self.ch):
                literal = self._read_identifier()
                kind = Token.lookup_ident(literal)
                return self._new_token(kind, literal)
            elif self.ch.isdigit():
                kind = Token.INT
                literal = self._read_number()
                return self._new_token(kind, literal)
            else:
                tok = self._new_token(Token.ILLEGAL, self.ch)

        self._read_char()
        return tok

    @staticmethod
    def _is_letter(ch):
        return ch.isalpha() or ch == "_"

    def _new_token(self, kind, literal):
        return Token(
            kind,
            literal,
            SourceLocation(self.source_filename, self.source_line),
        )

    def _skip_whitespace(self):
        while self.ch.isspace():
            self._read_char()

    def _is_eof(self):
        return self.read_position >= len(self.source)

    def _read_char(self):
        if self._is_eof():
            self.ch = Lexer.EOF_LITERAL
        else:
            self.source_line += self.ch == "\n"
            self.ch = self.source[self.read_position]
        self.position = self.read_position
        self.read_position += 1

    def _peek_char(self):
        return (
            Lexer.EOF_LITERAL
            if self._is_eof()
            else self.source[self.read_position]
        )

    def _read_identifier(self):
        start = self.position
        while Lexer._is_letter(self.ch):
            self._read_char()
        return self.source[start : self.position]

    def _read_number(self):
        start = self.position
        while self.ch.isdigit():
            self._read_char()
        return self.source[start : self.position]


class REPL:
    def __init__(self):
        pass

    def run(self):
        while True:
            try:
                self._step()
            except (EOFError, KeyboardInterrupt):
                return

    def _step(self):
        line = input(">> ")
        l = Lexer(line, "<repl>")
        tok = l.next_token()
        while tok.kind != Token.EOF:
            print(tok)
            tok = l.next_token()


if __name__ == "__main__":
    REPL().run()

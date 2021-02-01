#!/usr/bin/env python3

import enum
from typing import Callable, Dict, List, Optional


class SourceLocation:
    def __init__(self, filename: Optional[str], line: int) -> None:
        self.filename: Optional[str] = filename
        self.line: int = line

    def __repr__(self) -> str:
        if self.filename == None:
            return f"line {self.line}"
        return f"{self.filename}, line {self.line}"


class TokenKind(enum.Enum):
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


class Token:
    def __init__(
        self, kind: TokenKind, literal: str, source_location: SourceLocation
    ) -> None:
        self.kind = kind
        self.literal = literal
        self.source_location = source_location

    def __str__(self) -> str:
        if self.kind == TokenKind.IDENT:
            return f"{self.kind}({self.literal})"
        if self.kind == TokenKind.INT:
            return f"{self.kind}({self.literal})"
        return f"{self.kind}"

    @staticmethod
    def lookup_ident(ident: str) -> TokenKind:
        keywords = {
            "fn": TokenKind.FUNCTION,
            "let": TokenKind.LET,
            "true": TokenKind.TRUE,
            "false": TokenKind.FALSE,
            "if": TokenKind.IF,
            "else": TokenKind.ELSE,
            "return": TokenKind.RETURN,
        }
        return keywords.get(ident, TokenKind.IDENT)


class Lexer:
    EOF_LITERAL = ""

    def __init__(self, source: str, filename: Optional[str] = None) -> None:
        self.source: str = source
        self.source_filename: Optional[str] = filename
        self.source_line: int = 1
        self.position: int = 0
        self.read_position: int = 0
        self.ch: Optional[str] = None
        self._read_char()

    def next_token(self) -> Token:
        self._skip_whitespace()

        if self.ch == "=":
            if self._peek_char() == "=":
                self._read_char()
                tok = self._new_token(TokenKind.EQ, "==")
            else:
                tok = self._new_token(TokenKind.ASSIGN, self.ch)
        elif self.ch == "+":
            tok = self._new_token(TokenKind.PLUS, self.ch)
        elif self.ch == "-":
            tok = self._new_token(TokenKind.MINUS, self.ch)
        elif self.ch == "!":
            if self._peek_char() == "=":
                self._read_char()
                tok = self._new_token(TokenKind.NOT_EQ, "!=")
            else:
                tok = self._new_token(TokenKind.BANG, self.ch)
        elif self.ch == "/":
            tok = self._new_token(TokenKind.SLASH, self.ch)
        elif self.ch == "*":
            tok = self._new_token(TokenKind.ASTERISK, self.ch)
        elif self.ch == "<":
            tok = self._new_token(TokenKind.LT, self.ch)
        elif self.ch == ">":
            tok = self._new_token(TokenKind.GT, self.ch)
        elif self.ch == ";":
            tok = self._new_token(TokenKind.SEMICOLON, self.ch)
        elif self.ch == ",":
            tok = self._new_token(TokenKind.COMMA, self.ch)
        elif self.ch == "(":
            tok = self._new_token(TokenKind.LPAREN, self.ch)
        elif self.ch == ")":
            tok = self._new_token(TokenKind.RPAREN, self.ch)
        elif self.ch == "{":
            tok = self._new_token(TokenKind.LBRACE, self.ch)
        elif self.ch == "}":
            tok = self._new_token(TokenKind.RBRACE, self.ch)
        elif self.ch == Lexer.EOF_LITERAL:
            tok = self._new_token(TokenKind.EOF, self.ch)
        else:
            if Lexer._is_letter(self.ch):
                literal = self._read_identifier()
                kind = Token.lookup_ident(literal)
                return self._new_token(kind, literal)
            elif self.ch.isdigit():
                kind = TokenKind.INT
                literal = self._read_number()
                return self._new_token(kind, literal)
            else:
                tok = self._new_token(TokenKind.ILLEGAL, self.ch)

        self._read_char()
        return tok

    @staticmethod
    def _is_letter(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    def _new_token(self, kind: TokenKind, literal: str) -> Token:
        return Token(
            kind,
            literal,
            SourceLocation(self.source_filename, self.source_line),
        )

    def _skip_whitespace(self) -> None:
        while self.ch.isspace():
            self._read_char()

    def _is_eof(self) -> bool:
        return self.read_position >= len(self.source)

    def _read_char(self) -> None:
        if self._is_eof():
            self.ch = Lexer.EOF_LITERAL
        else:
            self.source_line += self.ch == "\n"
            self.ch = self.source[self.read_position]
        self.position = self.read_position
        self.read_position += 1

    def _peek_char(self) -> str:
        return (
            Lexer.EOF_LITERAL
            if self._is_eof()
            else self.source[self.read_position]
        )

    def _read_identifier(self) -> str:
        start = self.position
        while Lexer._is_letter(self.ch):
            self._read_char()
        return self.source[start : self.position]

    def _read_number(self) -> str:
        start = self.position
        while self.ch.isdigit():
            self._read_char()
        return self.source[start : self.position]


class ParseError(Exception):
    def __init__(self, tok: Token, why: str) -> None:
        self.tok = tok
        self.why = why

    def __str__(self) -> str:
        return f"[{self.tok.source_location}] {self.why}"


class Node:
    def __init__(self) -> None:
        self.token: Token
        raise NotImplementedError()

    def token_literal(self) -> str:
        """
        Returns the literal value of the token this node is associated with.
        """
        return self.token.literal

    def __str__(self) -> str:
        """
        The Node.String() method from the "Writing and Interpreter in Go".
        Used to print the AST nodes for debugging (pg 64).
        """
        raise NotImplementedError()


class Statement(Node):
    def __init__(self) -> None:
        raise NotImplementedError()


class Expression(Node):
    def __init__(self) -> None:
        raise NotImplementedError()


class Program(Node):
    def __init__(self) -> None:
        self.statements: List[Statement] = list()

    def token_literal(self) -> str:
        if len(self.statements) > 0:
            return self.statements[0].token_literal()
        return ""

    def __str__(self):
        return str().join(map(str, self.statements))


class Identifier(Expression):
    def __init__(self, token: Token, value: str) -> None:
        self.token: Token = token
        self.value: str = value

    def __str__(self):
        return str(self.value)


class LetStatement(Statement):
    def __init__(
        self, token: Token, name: Identifier, value: Expression
    ) -> None:
        self.token: Token = token  # The "let" token.
        self.name: Identifier = name  # Identifier being bound to.
        self.value: Expression = value  # Expression being bound.

    def __str__(self):
        return f"{self.token_literal()} {self.name} = {self.value};"


class ReturnStatement(Statement):
    def __init__(self, token: Token, return_value: Expression) -> None:
        self.token: Token = token  # The "return" token.
        self.return_value: Expression = return_value

    def __str__(self):
        return f"{self.token_literal()} {self.return_value};"


class BlockStatement(Statement):
    def __init__(self, token: Token, statements: List[Statement]) -> None:
        self.token: Token = token  # The "{" token
        self.statements: List[Statement] = statements

    def __str__(self) -> str:
        s = "".join(map(str, self.statements))
        return f"{{ {s} }}"


class ExpressionStatement(Statement):
    def __init__(self, token: Token, expression: Expression) -> None:
        self.token: Token = token
        self.expression: Expression = expression

    def __str__(self):
        return str(self.expression)


class IntegerLiteral(Expression):
    def __init__(self, token: Token, value: int) -> None:
        self.token: Token = token
        self.value: int = value

    def __str__(self):
        return str(self.value)


class BooleanLiteral(Expression):
    def __init__(self, token: Token, value: bool) -> None:
        self.token: Token = token
        self.value: bool = value

    def __str__(self):
        return self.token.literal


class FunctionLiteral(Expression):
    def __init__(
        self, token: Token, parameters: List[Identifier], body: BlockStatement,
    ) -> None:
        self.token: Token = token  # the "fn" token
        self.parameters: List[Identifier] = parameters
        self.body: BlockStatement = body

    def __str__(self) -> str:
        params = ", ".join(map(str, self.parameters))
        return f"{self.token}({params}) {self.body} }}"


class PrefixExpression(Expression):
    def __init__(self, token: Token, operator: str, right: Expression) -> None:
        self.token: Token = token  # The prefix token, e.g. !
        self.operator: str = operator
        self.right: Expression = right

    def __str__(self):
        return f"({self.operator}{self.right})"


class InfixExpression(Expression):
    def __init__(
        self, token: Token, left: Expression, operator: str, right: Expression
    ) -> None:
        self.token: Token = token  # The infix token, e.g. +
        self.left: Expression = left
        self.operator: str = operator
        self.right: Expression = right

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"


class CallExpression(Expression):
    def __init__(
        self, token: Token, function: Expression, arguments: List[Expression],
    ) -> None:
        self.token: Token = token  # the "(" token.
        self.function: Expression = function
        self.arguments: List[Expression] = arguments

    def __str__(self) -> str:
        args = ", ".join(map(str, self.arguments))
        return f"{self.function}({args})"


class IfExpression(Expression):
    def __init__(
        self,
        token: Token,
        condition: Expression,
        consequence: BlockStatement,
        alternative: Optional[BlockStatement],
    ) -> None:
        self.token: Token = token  # The "if" token
        self.condition: Expression = condition
        self.consequence: BlockStatement = consequence
        self.alternative: BlockStatement = alternative

    def __str__(self) -> str:
        consequencestr = f"if {self.condition} {self.consequence}"
        alternativestr = f"else {self.alternative}" if self.alternative else ""
        return f"{consequencestr} {alternativestr}"


class Precedence(enum.IntEnum):
    # fmt: off
    LOWEST      = enum.auto()
    EQUALS      = enum.auto() # ==
    LESSGREATER = enum.auto() # > or <
    SUM         = enum.auto() # +
    PRODUCT     = enum.auto() # *
    PREFIX      = enum.auto() # -X or !X
    CALL        = enum.auto() # myFunction(X)
    # fmt: on


class Parser:
    # > A Pratt parser's main idea is the association of parsing functions with
    # > token types. Whenever this token type is encountered, the parsing
    # > functions are called to parse the appropriate expression and return an
    # > AST node that represents it. Each token type can have up to two parsing
    # > functions associated with it, depending on whether the token is found in
    # > a prefix or an infix position.
    # >     - Writing and Interpreter in Go, ver 1.7, pg. 67.
    #
    # PrefixParseFunction : func()           -> Expression
    # InfixParseFunction  : func(Expression) -> Expression
    #
    # Both the prefix and infix parse functions return an Expression.
    # The prefix parse function takes no argument and parses the expression on
    # the right hand side of a prefix operator being parsed. The infix parse
    # functions takes and argument representing the left hand side of and infix
    # operator being parsed.
    PrefixParseFunction = Callable[["Parser"], Expression]
    InfixParseFunction = Callable[["Parser", Expression], Expression]

    PRECEDENCES: Dict[TokenKind, Precedence] = {
        # fmt: off
        TokenKind.EQ:       Precedence.EQUALS,
        TokenKind.NOT_EQ:   Precedence.EQUALS,
        TokenKind.LT:       Precedence.LESSGREATER,
        TokenKind.GT:       Precedence.LESSGREATER,
        TokenKind.PLUS:     Precedence.SUM,
        TokenKind.MINUS:    Precedence.SUM,
        TokenKind.SLASH:    Precedence.PRODUCT,
        TokenKind.ASTERISK: Precedence.PRODUCT,
        TokenKind.LPAREN:   Precedence.CALL,
        # fmt: on
    }

    def __init__(self, lexer: Lexer) -> None:
        self.lexer: Lexer = lexer
        self.cur_token: Optional[Token] = None
        self.peek_token: Optional[Token] = None

        self.prefix_parse_fns: Dict[
            TokenKind, Parser.PrefixParseFunction
        ] = dict()
        self._register_prefix(TokenKind.IDENT, Parser.parse_identifier)
        self._register_prefix(TokenKind.INT, Parser.parse_integer_literal)
        self._register_prefix(TokenKind.TRUE, Parser.parse_boolean_literal)
        self._register_prefix(TokenKind.FALSE, Parser.parse_boolean_literal)
        self._register_prefix(TokenKind.BANG, Parser.parse_prefix_expression)
        self._register_prefix(TokenKind.MINUS, Parser.parse_prefix_expression)
        self._register_prefix(TokenKind.LPAREN, Parser.parse_grouped_expression)
        self._register_prefix(TokenKind.IF, Parser.parse_if_expression)
        self._register_prefix(TokenKind.FUNCTION, Parser.parse_function_literal)

        self.infix_parse_fns: Dict[
            TokenKind, Parser.InfixParseFunction
        ] = dict()
        self._register_infix(TokenKind.PLUS, Parser.parse_infix_expression)
        self._register_infix(TokenKind.MINUS, Parser.parse_infix_expression)
        self._register_infix(TokenKind.SLASH, Parser.parse_infix_expression)
        self._register_infix(TokenKind.ASTERISK, Parser.parse_infix_expression)
        self._register_infix(TokenKind.EQ, Parser.parse_infix_expression)
        self._register_infix(TokenKind.NOT_EQ, Parser.parse_infix_expression)
        self._register_infix(TokenKind.LT, Parser.parse_infix_expression)
        self._register_infix(TokenKind.GT, Parser.parse_infix_expression)
        self._register_infix(TokenKind.LPAREN, Parser.parse_call_expression)

        # Read two tokens, so curToken and peekToken are both set.
        self._next_token()
        self._next_token()

    def parse_program(self) -> Program:
        program = Program()
        while not self._cur_token_is(TokenKind.EOF):
            stmt = self.parse_statement()
            program.statements.append(stmt)
            self._next_token()
        return program

    def parse_statement(self) -> Statement:
        if self._cur_token_is(TokenKind.LET):
            return self.parse_let_statement()
        if self._cur_token_is(TokenKind.RETURN):
            return self.parse_return_statement()
        return self.parse_expression_statement()

    def parse_let_statement(self) -> LetStatement:
        token = self.cur_token
        self._expect_peek(TokenKind.IDENT)
        name = Identifier(self.cur_token, self.cur_token.literal)
        self._expect_peek(TokenKind.ASSIGN)
        self._next_token()
        value = self.parse_expression(Precedence.LOWEST)
        if self._peek_token_is(TokenKind.SEMICOLON):
            self._next_token()
        return LetStatement(token, name, value)

    def parse_return_statement(self) -> ReturnStatement:
        token = self.cur_token
        self._next_token()
        return_value = self.parse_expression(Precedence.LOWEST)
        if self._peek_token_is(TokenKind.SEMICOLON):
            self._next_token()
        return ReturnStatement(token, return_value)

    def parse_expression_statement(self) -> ExpressionStatement:
        token = self.cur_token
        expression = self.parse_expression(Precedence.LOWEST)
        if self._peek_token_is(TokenKind.SEMICOLON):
            # Expression statements have optional semicolons which makes it
            # easier to type something like:
            # >> 5 + 5
            # into the REPL later on.
            self._next_token()
        return ExpressionStatement(token, expression)

    def parse_expression(self, precedence: "Precedence") -> Expression:
        prefix = self.prefix_parse_fns.get(self.cur_token.kind)
        if prefix == None:
            tok = self.cur_token
            msg = f"Expected expression, found {tok}"
            raise ParseError(tok, msg)
        left_exp = prefix(self)
        while precedence < self._peek_precedence():
            infix = self.infix_parse_fns.get(self.peek_token.kind, None)
            if infix == None:
                return left_exp
            self._next_token()
            left_exp = infix(self, left_exp)
        return left_exp

    def parse_identifier(self) -> Identifier:
        assert self.cur_token.kind == TokenKind.IDENT
        return Identifier(self.cur_token, self.cur_token.literal)

    def parse_integer_literal(self) -> IntegerLiteral:
        assert self.cur_token.kind == TokenKind.INT
        return IntegerLiteral(self.cur_token, int(self.cur_token.literal))

    def parse_boolean_literal(self) -> BooleanLiteral:
        assert (
            self.cur_token.kind == TokenKind.TRUE
            or self.cur_token.kind == TokenKind.FALSE
        )
        value = True if self.cur_token.kind == TokenKind.TRUE else False
        return BooleanLiteral(self.cur_token, value)

    def parse_prefix_expression(self) -> PrefixExpression:
        token = self.cur_token
        operator = self.cur_token.literal

        # Consume the prefix operator.
        self._next_token()

        right = self.parse_expression(Precedence.PREFIX)
        return PrefixExpression(token, operator, right)

    def parse_infix_expression(self, left) -> InfixExpression:
        assert isinstance(left, Expression)
        token = self.cur_token
        operator = self.cur_token.literal

        precedence = self._cur_precedence()
        self._next_token()
        right = self.parse_expression(precedence)

        return InfixExpression(token, left, operator, right)

    def parse_grouped_expression(self) -> Expression:
        self._next_token()
        exp = self.parse_expression(Precedence.LOWEST)
        self._expect_peek(TokenKind.RPAREN)
        return exp

    def parse_if_expression(self) -> IfExpression:
        token = self.cur_token
        self._expect_peek(TokenKind.LPAREN)
        self._next_token()  # Consume (
        condition = self.parse_expression(Precedence.LOWEST)
        self._expect_peek(TokenKind.RPAREN)
        self._expect_peek(TokenKind.LBRACE)
        consequence = self.parse_block_statement()
        alternative = None
        if self._peek_token_is(TokenKind.ELSE):
            self._next_token()
            self._expect_peek(TokenKind.LBRACE)
            alternative = self.parse_block_statement()
        return IfExpression(token, condition, consequence, alternative)

    def parse_block_statement(self) -> BlockStatement:
        token = self.cur_token
        statements = list()
        self._next_token()  # Consume {
        while not self._cur_token_is(TokenKind.RBRACE):
            if self._cur_token_is(TokenKind.EOF):
                tok = self.cur_token
                raise ParseError(tok, "Unexpected {tok} in block statment")
            statements.append(self.parse_statement())
            self._next_token()
        return BlockStatement(token, statements)

    def parse_function_literal(self) -> FunctionLiteral:
        token = self.cur_token
        self._expect_peek(TokenKind.LPAREN)
        parameters = self.parse_function_parameters()
        self._expect_peek(TokenKind.LBRACE)
        body = self.parse_block_statement()
        return FunctionLiteral(token, parameters, body)

    def parse_function_parameters(self) -> List[Identifier]:
        identifiers: List[Identifier] = list()
        if self._peek_token_is(TokenKind.RPAREN):
            self._next_token()
            return identifiers

        self._next_token()
        ident = Identifier(self.cur_token, self.cur_token.literal)
        identifiers.append(ident)

        while self._peek_token_is(TokenKind.COMMA):
            self._next_token()
            self._next_token()
            ident = Identifier(self.cur_token, self.cur_token.literal)
            identifiers.append(ident)

        self._expect_peek(TokenKind.RPAREN)
        return identifiers

    def parse_call_expression(self, function: Expression) -> CallExpression:
        token = self.cur_token
        arguments = self.parse_call_arguments(function)
        return CallExpression(token, function, arguments)

    def parse_call_arguments(self, function) -> List[Expression]:
        args: List[Expression] = []
        if self._peek_token_is(TokenKind.RPAREN):
            self._next_token()
            return args

        self._next_token()
        args.append(self.parse_expression(Precedence.LOWEST))
        while self._peek_token_is(TokenKind.COMMA):
            self._next_token()
            self._next_token()
            args.append(self.parse_expression(Precedence.LOWEST))

        self._expect_peek(TokenKind.RPAREN)
        return args

    def _register_prefix(
        self,
        token_kind: TokenKind,
        prefix_parse_fn: "Parser.PrefixParseFunction",
    ) -> None:
        self.prefix_parse_fns[token_kind] = prefix_parse_fn

    def _register_infix(
        self, token_kind: TokenKind, infix_parse_fn: "Parser.InfixParseFunction"
    ) -> None:
        self.infix_parse_fns[token_kind] = infix_parse_fn

    def _next_token(self) -> None:
        self.cur_token = self.peek_token
        self.peek_token = self.lexer.next_token()

    def _cur_token_is(self, kind) -> bool:
        return self.cur_token.kind == kind

    def _peek_token_is(self, kind) -> bool:
        return self.peek_token.kind == kind

    def _expect_peek(self, kind) -> None:
        if not self._peek_token_is(kind):
            tok = self.peek_token
            msg = f"Expected token {kind}, found {tok}"
            raise ParseError(tok, msg)
        self._next_token()

    def _cur_precedence(self) -> Precedence:
        return Parser.PRECEDENCES.get(self.cur_token.kind, Precedence.LOWEST)

    def _peek_precedence(self) -> Precedence:
        return Parser.PRECEDENCES.get(self.peek_token.kind, Precedence.LOWEST)


class REPL:
    def __init__(self):
        pass

    def run(self) -> None:
        while True:
            try:
                self._step()
            except ParseError as e:
                print(e)
            except (EOFError, KeyboardInterrupt):
                print("", end="\n")
                return

    def _step(self) -> None:
        line = input(">> ")
        l = Lexer(line)
        p = Parser(l)
        try:
            program = p.parse_program()
        except ParseError as e:
            print(e)
            return
        print(program)


if __name__ == "__main__":
    REPL().run()

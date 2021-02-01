#!/usr/bin/env python3

import enum
from typing import Callable, Dict, List, Union


class SourceLocation:
    def __init__(self, filename: str, line: int) -> None:
        self.filename = filename
        self.line = line

    def __repr__(self) -> str:
        return f"{self.filename}, line {self.line}"


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
    def lookup_ident(ident: str) -> str:
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

    def __init__(
        self, kind: str, literal: str, source_location: SourceLocation
    ) -> None:
        self.kind = kind
        self.literal = literal
        self.source_location = source_location

    def __str__(self) -> str:
        if self.kind == Token.IDENT:
            return f"{self.kind}({self.literal})"
        if self.kind == Token.INT:
            return f"{self.kind}({self.literal})"
        return f"{self.kind}"


class Lexer:
    EOF_LITERAL = ""

    def __init__(self, source: str, filename: str = "<nofile>") -> None:
        self.source: str = source
        self.source_filename: str = filename
        self.source_line: int = 0
        self.position: int = 0
        self.read_position: int = 0
        self.ch: Union[str, None] = None
        self._read_char()

    def next_token(self) -> Token:
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
    def _is_letter(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    def _new_token(self, kind: str, literal: str) -> Token:
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
        raise NotImplementedError()

    def token_literal(self) -> str:
        """
        Returns the literal value of the token this node is associated with.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        The Node.String() method from the book.
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

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return str(self.value)


class LetStatement(Statement):
    def __init__(
        self, token: Token, name: Identifier, value: Expression
    ) -> None:
        self.token: Token = token  # The "let" token.
        self.name: Identifier = name  # Identifier being bound to.
        self.value: Expression = value  # Expression being bound.

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return f"{self.token_literal()} {self.name} = {self.value};"


class ReturnStatement(Statement):
    def __init__(self, token: Token, return_value: Expression) -> None:
        self.token: Token = token  # The "return" token.
        self.return_value: Expression = return_value

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return f"{self.token_literal()} {self.return_value};"


class ExpressionStatement(Statement):
    def __init__(self, token: Token, expression: Expression) -> None:
        self.token: Token = token
        self.expression: Expression = expression

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return str(self.expression)


class IntegerLiteral(Expression):
    def __init__(self, token: Token, value: int) -> None:
        self.token: Token = token
        self.value: int = value

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return str(self.value)


class BooleanLiteral(Expression):
    def __init__(self, token: Token, value: bool) -> None:
        self.token: Token = token
        self.value: bool = value

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return self.token.literal


class PrefixExpression(Expression):
    def __init__(self, token: Token, operator: str, right: Expression) -> None:
        self.token: Token = token  # The prefix token, e.g. !
        self.operator: str = operator
        self.right: Expression = right

    def token_literal(self) -> str:
        return self.token.literal

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

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"


class BlockStatement(Statement):
    def __init__(self, token: Token, statements: List[Statement]) -> None:
        self.token: Token = token  # The "{" token
        self.statements: List[Statement] = statements

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        s = "".join(map(str, self.statements))
        return f"{{ {s} }}"


class IfExpression(Expression):
    def __init__(
        self,
        token: Token,
        condition: Expression,
        consequence: BlockStatement,
        alternative: Union[BlockStatement, None],
    ) -> None:
        self.token: Token = token  # The "if" token
        self.condition: Expression = condition
        self.consequence: BlockStatement = consequence
        self.alternative: BlockStatement = alternative

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        consequencestr = f"if {self.condition} {self.consequence}"
        alternativestr = f"else {self.alternative}" if self.alternative else ""
        return f"{consequencestr} {alternativestr}"


class FunctionLiteral(Expression):
    def __init__(
        self, token: Token, parameters: List[Identifier], body: BlockStatement,
    ) -> None:
        self.token: Token = token  # the "fn" token
        self.parameters: List[Identifier] = parameters
        self.body: BlockStatement = body

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        params = ", ".join(map(str, self.parameters))
        return f"{self.token}({params}) {self.body} }}"


class CallExpression(Expression):
    def __init__(
        self, token: Token, function: Expression, arguments: List[Expression],
    ) -> None:
        self.token: Token = token  # the "(" token.
        self.function: Expression = function
        self.arguments: List[Expression] = arguments

    def token_literal(self) -> str:
        return self.token.literal

    def __str__(self) -> str:
        args = ", ".join(map(str, self.arguments))
        return f"{self.function}({args})"


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

    PRECEDENCES: Dict[str, "Parser.Precedence"] = {
        # fmt: off
        Token.EQ:       Precedence.EQUALS,
        Token.NOT_EQ:   Precedence.EQUALS,
        Token.LT:       Precedence.LESSGREATER,
        Token.GT:       Precedence.LESSGREATER,
        Token.PLUS:     Precedence.SUM,
        Token.MINUS:    Precedence.SUM,
        Token.SLASH:    Precedence.PRODUCT,
        Token.ASTERISK: Precedence.PRODUCT,
        Token.LPAREN:   Precedence.CALL,
        # fmt: on
    }

    def __init__(self, lexer: Lexer) -> None:
        self.lexer: Lexer = lexer
        self.cur_token: Union[Token, None] = None
        self.peek_token: Union[Token, None] = None

        # Token Kind -> Prefix Parse Function
        self.prefix_parse_fns: Dict[str, Parser.PrefixParseFunction] = dict()
        self._register_prefix(Token.IDENT, Parser.parse_identifier)
        self._register_prefix(Token.INT, Parser.parse_integer_literal)
        self._register_prefix(Token.TRUE, Parser.parse_boolean_literal)
        self._register_prefix(Token.FALSE, Parser.parse_boolean_literal)
        self._register_prefix(Token.BANG, Parser.parse_prefix_expression)
        self._register_prefix(Token.MINUS, Parser.parse_prefix_expression)
        self._register_prefix(Token.LPAREN, Parser.parse_grouped_expression)
        self._register_prefix(Token.IF, Parser.parse_if_expression)
        self._register_prefix(Token.FUNCTION, Parser.parse_function_literal)
        # Token Kind -> Infix Parse Function
        self.infix_parse_fns: Dict[str, Parser.InfixParseFunction] = dict()
        self._register_infix(Token.PLUS, Parser.parse_infix_expression)
        self._register_infix(Token.MINUS, Parser.parse_infix_expression)
        self._register_infix(Token.SLASH, Parser.parse_infix_expression)
        self._register_infix(Token.ASTERISK, Parser.parse_infix_expression)
        self._register_infix(Token.EQ, Parser.parse_infix_expression)
        self._register_infix(Token.NOT_EQ, Parser.parse_infix_expression)
        self._register_infix(Token.LT, Parser.parse_infix_expression)
        self._register_infix(Token.GT, Parser.parse_infix_expression)
        self._register_infix(Token.LPAREN, Parser.parse_call_expression)

        # Read two tokens, so curToken and peekToken are both set.
        self._next_token()
        self._next_token()

    def parse_program(self) -> Program:
        program = Program()
        while not self._cur_token_is(Token.EOF):
            stmt = self.parse_statement()
            program.statements.append(stmt)
            self._next_token()
        return program

    def parse_statement(self) -> Statement:
        if self._cur_token_is(Token.LET):
            return self.parse_let_statement()
        if self._cur_token_is(Token.RETURN):
            return self.parse_return_statement()
        return self.parse_expression_statement()

    def parse_let_statement(self) -> LetStatement:
        token = self.cur_token
        self._expect_peek(Token.IDENT)
        name = Identifier(self.cur_token, self.cur_token.literal)
        self._expect_peek(Token.ASSIGN)
        self._next_token()
        value = self.parse_expression(Parser.Precedence.LOWEST)
        if self._peek_token_is(Token.SEMICOLON):
            self._next_token()
        return LetStatement(token, name, value)

    def parse_return_statement(self) -> ReturnStatement:
        token = self.cur_token
        self._next_token()
        return_value = self.parse_expression(Parser.Precedence.LOWEST)
        if self._peek_token_is(Token.SEMICOLON):
            self._next_token()
        return ReturnStatement(token, return_value)

    def parse_expression_statement(self) -> ExpressionStatement:
        token = self.cur_token
        expression = self.parse_expression(Parser.Precedence.LOWEST)
        if self._peek_token_is(Token.SEMICOLON):
            # Expression statements have optional semicolons which makes it
            # easier to type something like:
            # >> 5 + 5
            # into the REPL later on.
            self._next_token()
        return ExpressionStatement(token, expression)

    def parse_expression(self, precedence: "Parser.Precedence") -> Expression:
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
        assert self.cur_token.kind == Token.IDENT
        return Identifier(self.cur_token, self.cur_token.literal)

    def parse_integer_literal(self) -> IntegerLiteral:
        assert self.cur_token.kind == Token.INT
        return IntegerLiteral(self.cur_token, int(self.cur_token.literal))

    def parse_boolean_literal(self) -> BooleanLiteral:
        assert (
            self.cur_token.kind == Token.TRUE
            or self.cur_token.kind == Token.FALSE
        )
        value = True if self.cur_token.kind == Token.TRUE else False
        return BooleanLiteral(self.cur_token, value)

    def parse_prefix_expression(self) -> PrefixExpression:
        token = self.cur_token
        operator = self.cur_token.literal

        # Consume the prefix operator.
        self._next_token()

        right = self.parse_expression(Parser.Precedence.PREFIX)
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
        exp = self.parse_expression(Parser.Precedence.LOWEST)
        self._expect_peek(Token.RPAREN)
        return exp

    def parse_if_expression(self) -> IfExpression:
        token = self.cur_token
        self._expect_peek(Token.LPAREN)
        self._next_token()  # Consume (
        condition = self.parse_expression(Parser.Precedence.LOWEST)
        self._expect_peek(Token.RPAREN)
        self._expect_peek(Token.LBRACE)
        consequence = self.parse_block_statement()
        alternative = None
        if self._peek_token_is(Token.ELSE):
            self._next_token()
            self._expect_peek(Token.LBRACE)
            alternative = self.parse_block_statement()
        return IfExpression(token, condition, consequence, alternative)

    def parse_block_statement(self) -> BlockStatement:
        token = self.cur_token
        statements = list()
        self._next_token()  # Consume {
        while not self._cur_token_is(Token.RBRACE):
            if self._cur_token_is(Token.EOF):
                tok = self.cur_token
                raise ParseError(tok, "Unexpected {tok} in block statment")
            statements.append(self.parse_statement())
            self._next_token()
        return BlockStatement(token, statements)

    def parse_function_literal(self) -> FunctionLiteral:
        token = self.cur_token
        self._expect_peek(Token.LPAREN)
        parameters = self.parse_function_parameters()
        self._expect_peek(Token.LBRACE)
        body = self.parse_block_statement()
        return FunctionLiteral(token, parameters, body)

    def parse_function_parameters(self) -> List[Identifier]:
        identifiers: List[Identifier] = list()
        if self._peek_token_is(Token.RPAREN):
            self._next_token()
            return identifiers

        self._next_token()
        ident = Identifier(self.cur_token, self.cur_token.literal)
        identifiers.append(ident)

        while self._peek_token_is(Token.COMMA):
            self._next_token()
            self._next_token()
            ident = Identifier(self.cur_token, self.cur_token.literal)
            identifiers.append(ident)

        self._expect_peek(Token.RPAREN)
        return identifiers

    def parse_call_expression(self, function: Expression) -> CallExpression:
        token = self.cur_token
        arguments = self.parse_call_arguments(function)
        return CallExpression(token, function, arguments)

    def parse_call_arguments(self, function) -> List[Expression]:
        args: List[Expression] = []
        if self._peek_token_is(Token.RPAREN):
            self._next_token()
            return args

        self._next_token()
        args.append(self.parse_expression(Parser.Precedence.LOWEST))
        while self._peek_token_is(Token.COMMA):
            self._next_token()
            self._next_token()
            args.append(self.parse_expression(Parser.Precedence.LOWEST))

        self._expect_peek(Token.RPAREN)
        return args

    def _register_prefix(
        self, token_kind: str, prefix_parse_fn: "Parser.PrefixParseFunction"
    ) -> None:
        self.prefix_parse_fns[token_kind] = prefix_parse_fn

    def _register_infix(
        self, token_kind: str, infix_parse_fn: "Parser.InfixParseFunction"
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

    def _cur_precedence(self) -> "Parser.Precedence":
        return Parser.PRECEDENCES.get(
            self.cur_token.kind, Parser.Precedence.LOWEST
        )

    def _peek_precedence(self) -> "Parser.Precedence":
        return Parser.PRECEDENCES.get(
            self.peek_token.kind, Parser.Precedence.LOWEST
        )


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
        l = Lexer(line, "<repl>")
        p = Parser(l)
        try:
            program = p.parse_program()
        except ParseError as e:
            print(e)
            return
        print(program)


if __name__ == "__main__":
    REPL().run()

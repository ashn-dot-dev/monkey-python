#!/usr/bin/env python3

import enum
from typing import Callable, Dict, List, Optional, Union, cast


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


class AstNode:
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
        The Node.String() method from "Writing and Interpreter in Go".
        Used to print the AST nodes for debugging (pg 64).
        """
        raise NotImplementedError()


class AstStatement(AstNode):
    def __init__(self) -> None:
        raise NotImplementedError()


class AstExpression(AstNode):
    def __init__(self) -> None:
        raise NotImplementedError()


class AstProgram(AstNode):
    def __init__(self) -> None:
        self.statements: List[AstStatement] = list()

    def token_literal(self) -> str:
        if len(self.statements) > 0:
            return self.statements[0].token_literal()
        return ""

    def __str__(self):
        return str().join(map(str, self.statements))


class AstIdentifier(AstExpression):
    def __init__(self, token: Token, value: str) -> None:
        self.token: Token = token
        self.value: str = value

    def __str__(self):
        return str(self.value)


class AstLetStatement(AstStatement):
    def __init__(
        self, token: Token, name: AstIdentifier, value: AstExpression
    ) -> None:
        self.token: Token = token  # The "let" token.
        self.name: AstIdentifier = name  # Identifier being bound to.
        self.value: AstExpression = value  # Expression being bound.

    def __str__(self):
        return f"{self.token_literal()} {self.name} = {self.value};"


class AstReturnStatement(AstStatement):
    def __init__(self, token: Token, return_value: AstExpression) -> None:
        self.token: Token = token  # The "return" token.
        self.return_value: AstExpression = return_value

    def __str__(self):
        return f"{self.token_literal()} {self.return_value};"


class AstBlockStatement(AstStatement):
    def __init__(self, token: Token, statements: List[AstStatement]) -> None:
        self.token: Token = token  # The "{" token
        self.statements: List[AstStatement] = statements

    def __str__(self) -> str:
        s = "; ".join(map(str, self.statements))
        return f"{{ {s} }}"


class AstExpressionStatement(AstStatement):
    def __init__(self, token: Token, expression: AstExpression) -> None:
        self.token: Token = token
        self.expression: AstExpression = expression

    def __str__(self):
        return str(self.expression)


class AstIntegerLiteral(AstExpression):
    def __init__(self, token: Token, value: int) -> None:
        self.token: Token = token
        self.value: int = value

    def __str__(self):
        return str(self.value)


class AstBooleanLiteral(AstExpression):
    def __init__(self, token: Token, value: bool) -> None:
        self.token: Token = token
        self.value: bool = value

    def __str__(self):
        return self.token.literal


class AstFunctionLiteral(AstExpression):
    def __init__(
        self,
        token: Token,
        parameters: List[AstIdentifier],
        body: AstBlockStatement,
    ) -> None:
        self.token: Token = token  # the "fn" token
        self.parameters: List[AstIdentifier] = parameters
        self.body: AstBlockStatement = body

    def __str__(self) -> str:
        params = ", ".join(map(str, self.parameters))
        return f"{self.token}({params}) {self.body} }}"


class AstPrefixExpression(AstExpression):
    def __init__(
        self, token: Token, operator: str, right: AstExpression
    ) -> None:
        self.token: Token = token  # The prefix token, e.g. !
        self.operator: str = operator
        self.right: AstExpression = right

    def __str__(self):
        return f"({self.operator}{self.right})"


class AstInfixExpression(AstExpression):
    def __init__(
        self,
        token: Token,
        left: AstExpression,
        operator: str,
        right: AstExpression,
    ) -> None:
        self.token: Token = token  # The infix token, e.g. +
        self.left: AstExpression = left
        self.operator: str = operator
        self.right: AstExpression = right

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"


class AstCallExpression(AstExpression):
    def __init__(
        self,
        token: Token,
        function: AstExpression,
        arguments: List[AstExpression],
    ) -> None:
        self.token: Token = token  # the "(" token.
        self.function: AstExpression = function
        self.arguments: List[AstExpression] = arguments

    def __str__(self) -> str:
        args = ", ".join(map(str, self.arguments))
        return f"{self.function}({args})"


class AstIfExpression(AstExpression):
    def __init__(
        self,
        token: Token,
        condition: AstExpression,
        consequence: AstBlockStatement,
        alternative: Optional[AstBlockStatement],
    ) -> None:
        self.token: Token = token  # The "if" token
        self.condition: AstExpression = condition
        self.consequence: AstBlockStatement = consequence
        self.alternative: AstBlockStatement = alternative

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
    # PrefixParseFunction : func()              -> AstExpression
    # InfixParseFunction  : func(AstExpression) -> AstExpression
    #
    # Both the prefix and infix parse functions return an AstExpression.
    # The prefix parse function takes no argument and parses the expression on
    # the right hand side of a prefix operator being parsed. The infix parse
    # functions takes and argument representing the left hand side of and infix
    # operator being parsed.
    PrefixParseFunction = Callable[["Parser"], AstExpression]
    InfixParseFunction = Callable[["Parser", AstExpression], AstExpression]

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

    def parse_program(self) -> AstProgram:
        program = AstProgram()
        while not self._cur_token_is(TokenKind.EOF):
            stmt = self.parse_statement()
            program.statements.append(stmt)
            self._next_token()
        return program

    def parse_statement(self) -> AstStatement:
        if self._cur_token_is(TokenKind.LET):
            return self.parse_let_statement()
        if self._cur_token_is(TokenKind.RETURN):
            return self.parse_return_statement()
        return self.parse_expression_statement()

    def parse_let_statement(self) -> AstLetStatement:
        token = self.cur_token
        self._expect_peek(TokenKind.IDENT)
        name = AstIdentifier(self.cur_token, self.cur_token.literal)
        self._expect_peek(TokenKind.ASSIGN)
        self._next_token()
        value = self.parse_expression(Precedence.LOWEST)
        if self._peek_token_is(TokenKind.SEMICOLON):
            self._next_token()
        return AstLetStatement(token, name, value)

    def parse_return_statement(self) -> AstReturnStatement:
        token = self.cur_token
        self._next_token()
        return_value = self.parse_expression(Precedence.LOWEST)
        if self._peek_token_is(TokenKind.SEMICOLON):
            self._next_token()
        return AstReturnStatement(token, return_value)

    def parse_block_statement(self) -> AstBlockStatement:
        token = self.cur_token
        statements = list()
        self._next_token()  # Consume {
        while not self._cur_token_is(TokenKind.RBRACE):
            if self._cur_token_is(TokenKind.EOF):
                tok = self.cur_token
                raise ParseError(tok, "Unexpected {tok} in block statment")
            statements.append(self.parse_statement())
            self._next_token()
        return AstBlockStatement(token, statements)

    def parse_expression_statement(self) -> AstExpressionStatement:
        token = self.cur_token
        expression = self.parse_expression(Precedence.LOWEST)
        if self._peek_token_is(TokenKind.SEMICOLON):
            # Expression statements have optional semicolons which makes it
            # easier to type something like:
            # >> 5 + 5
            # into the REPL later on.
            self._next_token()
        return AstExpressionStatement(token, expression)

    def parse_expression(self, precedence: "Precedence") -> AstExpression:
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

    def parse_identifier(self) -> AstIdentifier:
        assert self.cur_token.kind == TokenKind.IDENT
        return AstIdentifier(self.cur_token, self.cur_token.literal)

    def parse_integer_literal(self) -> AstIntegerLiteral:
        assert self.cur_token.kind == TokenKind.INT
        return AstIntegerLiteral(self.cur_token, int(self.cur_token.literal))

    def parse_boolean_literal(self) -> AstBooleanLiteral:
        assert (
            self.cur_token.kind == TokenKind.TRUE
            or self.cur_token.kind == TokenKind.FALSE
        )
        value = True if self.cur_token.kind == TokenKind.TRUE else False
        return AstBooleanLiteral(self.cur_token, value)

    def parse_function_literal(self) -> AstFunctionLiteral:
        token = self.cur_token
        self._expect_peek(TokenKind.LPAREN)
        parameters = self.parse_function_parameters()
        self._expect_peek(TokenKind.LBRACE)
        body = self.parse_block_statement()
        return AstFunctionLiteral(token, parameters, body)

    def parse_function_parameters(self) -> List[AstIdentifier]:
        identifiers: List[AstIdentifier] = list()
        if self._peek_token_is(TokenKind.RPAREN):
            self._next_token()
            return identifiers

        self._next_token()
        ident = AstIdentifier(self.cur_token, self.cur_token.literal)
        identifiers.append(ident)

        while self._peek_token_is(TokenKind.COMMA):
            self._next_token()
            self._next_token()
            ident = AstIdentifier(self.cur_token, self.cur_token.literal)
            identifiers.append(ident)

        self._expect_peek(TokenKind.RPAREN)
        return identifiers

    def parse_prefix_expression(self) -> AstPrefixExpression:
        token = self.cur_token
        operator = self.cur_token.literal

        # Consume the prefix operator.
        self._next_token()

        right = self.parse_expression(Precedence.PREFIX)
        return AstPrefixExpression(token, operator, right)

    def parse_infix_expression(self, left: AstExpression) -> AstInfixExpression:
        token = self.cur_token
        operator = self.cur_token.literal

        precedence = self._cur_precedence()
        self._next_token()
        right = self.parse_expression(precedence)

        return AstInfixExpression(token, left, operator, right)

    def parse_grouped_expression(self) -> AstExpression:
        self._next_token()
        exp = self.parse_expression(Precedence.LOWEST)
        self._expect_peek(TokenKind.RPAREN)
        return exp

    def parse_call_expression(
        self, function: AstExpression
    ) -> AstCallExpression:
        token = self.cur_token
        arguments = self.parse_call_arguments(function)
        return AstCallExpression(token, function, arguments)

    def parse_call_arguments(self, function) -> List[AstExpression]:
        args: List[AstExpression] = []
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

    def parse_if_expression(self) -> AstIfExpression:
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
        return AstIfExpression(token, condition, consequence, alternative)

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


class Object:
    def __init__(self) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        The Object.Inspect() method from "Writing and Interpreter in Go".
        """
        raise NotImplementedError()

    @property
    def type(self) -> str:
        """
        The Object.Type() method from the "Writing and Interpreter in Go".
        """
        raise NotImplementedError()


class Environment:
    def __init__(self, outer: Optional["Environment"] = None) -> None:
        self.outer: Optional["Environment"] = outer
        self.store: Dict[str, Object] = dict()

    def get(self, name: str) -> Optional[Object]:
        obj = self.store.get(name, None)
        if obj == None and self.outer != None:
            return self.outer.get(name)
        return obj

    def set(self, name: str, val: Object) -> None:
        self.store[name] = val


class ObjectNull(Object):
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "null"

    @property
    def type(self) -> str:
        return "NULL"


class ObjectInteger(Object):
    def __init__(self, value: int) -> None:
        self.value: int = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def type(self) -> str:
        return "INTEGER"


class ObjectBoolean(Object):
    def __init__(self, value: bool) -> None:
        self.value: bool = value

    def __str__(self) -> str:
        return str(self.value).lower()

    @property
    def type(self) -> str:
        return "BOOLEAN"


class ObjectFunction(Object):
    def __init__(
        self,
        parameters: List[AstIdentifier],
        body: AstBlockStatement,
        env: Environment,
    ) -> None:
        self.parameters: List[AstIdentifier] = parameters
        self.body: AstBlockStatement = body
        self.env: Environment = env

    def __str__(self) -> str:
        params = ", ".join(map(str, self.parameters))
        return f"fn({params}) {self.body} }}"

    @property
    def type(self) -> str:
        return "FUNCTION"


class ObjectReturnValue(Object):
    def __init__(self, value: Object) -> None:
        self.value: Object = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def type(self) -> str:
        return "RETURN VALUE"


class ObjectError(Object):
    def __init__(self, what: str) -> None:
        self.what = what

    def __str__(self) -> str:
        return str(self.what)

    @property
    def type(self) -> str:
        return "ERROR"


def eval_ast(node: AstNode, env: Environment) -> Object:
    if isinstance(node, AstProgram):
        return eval_ast_program(node, env)
    if isinstance(node, AstBlockStatement):
        return eval_ast_block_statement(node, env)
    if isinstance(node, AstLetStatement):
        val = eval_ast(node.value, env)
        if isinstance(val, ObjectError):
            return val
        env.set(node.name.value, val)
        # Note: The book's implementation of Eval does not return a value here
        # and instead returns nil at the end of the evaluator.Eval function.
        # This interpreter chooses to make the result of a let statement a null
        # object.
        return ObjectNull()
    if isinstance(node, AstReturnStatement):
        ret = eval_ast(node.return_value, env)
        if isinstance(ret, ObjectError):
            return ret
        return ObjectReturnValue(ret)
    if isinstance(node, AstIdentifier):
        return eval_ast_identifier(node, env)
    if isinstance(node, AstExpressionStatement):
        return eval_ast(node.expression, env)
    if isinstance(node, AstIntegerLiteral):
        return ObjectInteger(node.value)
    if isinstance(node, AstBooleanLiteral):
        return ObjectBoolean(node.value)
    if isinstance(node, AstFunctionLiteral):
        params = node.parameters
        body = node.body
        return ObjectFunction(params, body, env)
    if isinstance(node, AstPrefixExpression):
        rhs = eval_ast(node.right, env)
        if isinstance(rhs, ObjectError):
            return rhs
        return eval_ast_prefix_expression(node.operator, rhs)
    if isinstance(node, AstInfixExpression):
        lhs = eval_ast(node.left, env)
        if isinstance(lhs, ObjectError):
            return lhs
        rhs = eval_ast(node.right, env)
        if isinstance(rhs, ObjectError):
            return rhs
        return eval_ast_infix_expression(lhs, node.operator, rhs)
    if isinstance(node, AstCallExpression):
        function = eval_ast(node.function, env)
        if isinstance(function, ObjectError):
            return function
        args = eval_ast_expressions(node.arguments, env)
        if isinstance(args, ObjectError):
            return args
        return apply_function(function, args)
    if isinstance(node, AstIfExpression):
        return eval_ast_if_expression(node, env)
    raise RuntimeError(f"Unhandled AST node {type(node)}")


def eval_ast_identifier(node: AstIdentifier, env: Environment) -> Object:
    val: Optional[Object] = env.get(node.value)
    return val or ObjectError(f"identifier not found: {node.value}")


def eval_ast_program(node: AstProgram, env: Environment) -> Object:
    result: Object = ObjectNull()
    for statement in node.statements:
        result = eval_ast(statement, env)
        if isinstance(result, ObjectReturnValue):
            return result.value
        if isinstance(result, ObjectError):
            return result
    return result


def eval_ast_block_statement(
    node: AstBlockStatement, env: Environment
) -> Object:
    result: Object = ObjectNull()
    for statement in node.statements:
        result = eval_ast(statement, env)
        if isinstance(result, ObjectReturnValue):
            return result
        if isinstance(result, ObjectError):
            return result
    return result


def eval_ast_prefix_expression(operator: str, rhs: Object) -> Object:
    def eval_ast_prefix_bang(rhs: Object) -> Object:
        if isinstance(rhs, ObjectNull):
            return ObjectBoolean(True)
        if isinstance(rhs, ObjectBoolean):
            return ObjectBoolean(not rhs.value)
        return ObjectBoolean(False)

    def eval_ast_prefix_minus(rhs: Object) -> Object:
        if not isinstance(rhs, ObjectInteger):
            return ObjectError(f"unknown operator: -{rhs.type}")
        return ObjectInteger(-rhs.value)

    if operator == "!":
        return eval_ast_prefix_bang(rhs)
    if operator == "-":
        return eval_ast_prefix_minus(rhs)
    return ObjectError(f"unknown operator: {operator}{rhs.type}")


def eval_ast_infix_expression(
    lhs: Object, operator: str, rhs: Object
) -> Object:
    def eval_ast_integer_infix_expression(
        operator: str, lhs: ObjectInteger, rhs: ObjectInteger
    ) -> Object:
        if operator == "+":
            return ObjectInteger(lhs.value + rhs.value)
        if operator == "-":
            return ObjectInteger(lhs.value - rhs.value)
        if operator == "*":
            return ObjectInteger(lhs.value * rhs.value)
        if operator == "/":
            return ObjectInteger(lhs.value // rhs.value)
        if operator == "<":
            return ObjectBoolean(lhs.value < rhs.value)
        if operator == ">":
            return ObjectBoolean(lhs.value > rhs.value)
        if operator == "==":
            return ObjectBoolean(lhs.value == rhs.value)
        if operator == "!=":
            return ObjectBoolean(lhs.value != rhs.value)
        return ObjectError(
            f"unknown operator: {lhs.type} {operator} {rhs.type}"
        )

    def eval_ast_boolean_infix_expression(
        operator: str, lhs: ObjectBoolean, rhs: ObjectBoolean
    ) -> Object:
        if operator == "==":
            return ObjectBoolean(lhs.value == rhs.value)
        if operator == "!=":
            return ObjectBoolean(lhs.value != rhs.value)
        return ObjectError(
            f"unknown operator: {lhs.type} {operator} {rhs.type}"
        )

    if isinstance(lhs, ObjectInteger) and isinstance(rhs, ObjectInteger):
        return eval_ast_integer_infix_expression(operator, lhs, rhs)
    if isinstance(lhs, ObjectBoolean) and isinstance(rhs, ObjectBoolean):
        return eval_ast_boolean_infix_expression(operator, lhs, rhs)
    return ObjectError(f"type mismatch: {lhs.type} {operator} {rhs.type}")


def eval_ast_if_expression(node: AstIfExpression, env: Environment) -> Object:
    def is_truthy(obj: Object) -> bool:
        if isinstance(obj, ObjectNull):
            return False
        if isinstance(obj, ObjectBoolean) and not obj.value:
            return False
        return True

    condition = eval_ast(node.condition, env)
    if isinstance(condition, ObjectError):
        return condition
    if is_truthy(condition):
        return eval_ast(node.consequence, env)
    elif node.alternative != None:
        return eval_ast(node.alternative, env)
    return ObjectNull()


def eval_ast_expressions(
    expressions: List[AstExpression], env: Environment
) -> Union[List[Object], ObjectError]:
    result: List[Object] = list()
    for expr in expressions:
        evaluated = eval_ast(expr, env)
        if isinstance(evaluated, ObjectError):
            return evaluated
        result.append(evaluated)
    return result


def apply_function(fn: Object, args: List[Object]) -> Object:
    if not isinstance(fn, Object):
        return ObjectError(f"not a function: {fn.type}")
    fn = cast(ObjectFunction, fn)

    def extend_env(fn: ObjectFunction, args: List[Object]) -> Environment:
        env = Environment(fn.env)
        for i in range(len(fn.parameters)):
            env.set(fn.parameters[i].value, args[i])
        return env

    env = extend_env(fn, args)
    evaluated = eval_ast(fn.body, env)
    if isinstance(evaluated, ObjectReturnValue):
        return evaluated.value
    return evaluated


def eval_source(source: str, env: Optional[Environment] = None) -> Object:
    lexer: Lexer = Lexer(source)
    parser: Parser = Parser(lexer)
    ast: AstNode = parser.parse_program()
    return eval_ast(ast, env or Environment())


class REPL:
    def __init__(self):
        pass

    def run(self) -> None:
        env: Environment = Environment()
        while True:
            try:
                self._step(env)
            except (EOFError, KeyboardInterrupt):
                print("", end="\n")
                return

    def _step(self, env: Environment) -> None:
        line = input(">> ")
        try:
            print(eval_source(line, env))
        except ParseError as e:
            print(e)


if __name__ == "__main__":
    REPL().run()

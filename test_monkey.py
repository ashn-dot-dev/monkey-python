#!/usr/bin/env python3

import collections
import typing
import unittest

import monkey


class LexerTest(unittest.TestCase):
    def test_next_token(self):
        source = "\n".join(
            [
                "let five = 5;",
                "let ten = 10;",
                "let add = fn(x, y) {",
                "   x + y;",
                "};",
                "let result = add(five, ten);",
                "!-/*5",
                "5 < 10 > 5",
                "if (5 < 10) { return true; } else { return false; }",
                "10 == 10",
                "10 != 9",
            ]
        )
        TestData = collections.namedtuple("TestData", ["kind", "literal"])
        expected = [
            TestData(monkey.TokenKind.LET, "let"),
            TestData(monkey.TokenKind.IDENT, "five"),
            TestData(monkey.TokenKind.ASSIGN, "="),
            TestData(monkey.TokenKind.INT, "5"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            #
            TestData(monkey.TokenKind.LET, "let"),
            TestData(monkey.TokenKind.IDENT, "ten"),
            TestData(monkey.TokenKind.ASSIGN, "="),
            TestData(monkey.TokenKind.INT, "10"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            #
            TestData(monkey.TokenKind.LET, "let"),
            TestData(monkey.TokenKind.IDENT, "add"),
            TestData(monkey.TokenKind.ASSIGN, "="),
            TestData(monkey.TokenKind.FUNCTION, "fn"),
            TestData(monkey.TokenKind.LPAREN, "("),
            TestData(monkey.TokenKind.IDENT, "x"),
            TestData(monkey.TokenKind.COMMA, ","),
            TestData(monkey.TokenKind.IDENT, "y"),
            TestData(monkey.TokenKind.RPAREN, ")"),
            TestData(monkey.TokenKind.LBRACE, "{"),
            TestData(monkey.TokenKind.IDENT, "x"),
            TestData(monkey.TokenKind.PLUS, "+"),
            TestData(monkey.TokenKind.IDENT, "y"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            TestData(monkey.TokenKind.RBRACE, "}"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            #
            TestData(monkey.TokenKind.LET, "let"),
            TestData(monkey.TokenKind.IDENT, "result"),
            TestData(monkey.TokenKind.ASSIGN, "="),
            TestData(monkey.TokenKind.IDENT, "add"),
            TestData(monkey.TokenKind.LPAREN, "("),
            TestData(monkey.TokenKind.IDENT, "five"),
            TestData(monkey.TokenKind.COMMA, ","),
            TestData(monkey.TokenKind.IDENT, "ten"),
            TestData(monkey.TokenKind.RPAREN, ")"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            #
            TestData(monkey.TokenKind.BANG, "!"),
            TestData(monkey.TokenKind.MINUS, "-"),
            TestData(monkey.TokenKind.SLASH, "/"),
            TestData(monkey.TokenKind.ASTERISK, "*"),
            TestData(monkey.TokenKind.INT, "5"),
            #
            TestData(monkey.TokenKind.INT, "5"),
            TestData(monkey.TokenKind.LT, "<"),
            TestData(monkey.TokenKind.INT, "10"),
            TestData(monkey.TokenKind.GT, ">"),
            TestData(monkey.TokenKind.INT, "5"),
            #
            TestData(monkey.TokenKind.IF, "if"),
            TestData(monkey.TokenKind.LPAREN, "("),
            TestData(monkey.TokenKind.INT, "5"),
            TestData(monkey.TokenKind.LT, "<"),
            TestData(monkey.TokenKind.INT, "10"),
            TestData(monkey.TokenKind.RPAREN, ")"),
            TestData(monkey.TokenKind.LBRACE, "{"),
            TestData(monkey.TokenKind.RETURN, "return"),
            TestData(monkey.TokenKind.TRUE, "true"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            TestData(monkey.TokenKind.RBRACE, "}"),
            TestData(monkey.TokenKind.ELSE, "else"),
            TestData(monkey.TokenKind.LBRACE, "{"),
            TestData(monkey.TokenKind.RETURN, "return"),
            TestData(monkey.TokenKind.FALSE, "false"),
            TestData(monkey.TokenKind.SEMICOLON, ";"),
            TestData(monkey.TokenKind.RBRACE, "}"),
            #
            TestData(monkey.TokenKind.INT, "10"),
            TestData(monkey.TokenKind.EQ, "=="),
            TestData(monkey.TokenKind.INT, "10"),
            #
            TestData(monkey.TokenKind.INT, "10"),
            TestData(monkey.TokenKind.NOT_EQ, "!="),
            TestData(monkey.TokenKind.INT, "9"),
            #
            TestData(monkey.TokenKind.EOF, ""),
        ]

        l = monkey.Lexer(source)
        for exp in expected:
            tok = l.next_token()
            self.assertEqual(tok.kind, exp.kind, tok.source_location)
            self.assertEqual(tok.literal, exp.literal, tok.source_location)


class AstTest(unittest.TestCase):
    def test_str(self):
        source = "let myvar = anothervar;"

        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(str(program), source)


class ParserTest(unittest.TestCase):
    def check_integer_literal(self, expr, value):
        self.assertIsInstance(expr, monkey.IntegerLiteral)
        self.assertEqual(expr.value, value)
        self.assertEqual(expr.token_literal(), str(value))

    def check_boolean_literal(self, expr, value):
        self.assertIsInstance(expr, monkey.BooleanLiteral)
        self.assertEqual(expr.value, value)
        self.assertEqual(expr.token_literal(), str(value).lower())

    def check_identifier(self, expr, value):
        self.assertIsInstance(expr, monkey.Identifier)
        self.assertEqual(expr.value, value)
        self.assertEqual(expr.token_literal(), value)

    def check_simple_prefix_expression(self, expr, operator, right):
        """
        Check a prefix expression in the form <OP><LITERAL>
        """
        self.assertIsInstance(expr, monkey.PrefixExpression)
        self.assertEqual(expr.operator, operator)
        self.check_literal_expression(expr.right, right)

    def check_simple_infix_expression(self, expr, left, operator, right):
        """
        Check an infix expression in the form <LITERAL><OP><LITERAL>
        """
        self.assertIsInstance(expr, monkey.InfixExpression)
        self.assertEqual(expr.operator, operator)
        self.check_literal_expression(expr.left, left)
        self.check_literal_expression(expr.right, right)

    def check_literal_expression(self, expr, value):
        t = type(value)
        if t == int:
            return self.check_integer_literal(expr, value)
        elif t == bool:
            return self.check_boolean_literal(expr, value)
        elif t == str:
            return self.check_identifier(expr, value)
        raise Exception(f"Unexpected type {t}")

    def test_identifier(self):
        statements = ["foobar;"]

        source = "\n".join(statements)
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(len(program.statements), len(statements))

        stmt = program.statements[0]
        self.assertIsInstance(stmt, monkey.ExpressionStatement)
        self.check_identifier(stmt.expression, "foobar")

    def test_let_statements(self):
        statements = ["let x = 5;", "let y = true;", "let foobar = y;"]
        TestData = collections.namedtuple("TestData", ["ident", "value"])
        expected = [
            TestData("x", 5),
            TestData("y", True),
            TestData("foobar", "y"),
        ]

        source = "\n".join(statements)
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(
            len(program.statements), len(statements), len(expected)
        )

        for i in range(len(program.statements)):
            stmt = program.statements[i]
            self.assertIsInstance(stmt, monkey.LetStatement)
            self.assertEqual(stmt.token_literal(), "let")
            self.assertEqual(stmt.name.value, expected[i].ident)
            self.assertEqual(stmt.name.token_literal(), expected[i].ident)
            self.check_literal_expression(stmt.value, expected[i].value)

    def test_return_statements(self):
        statements = ["return 5;", "return true;", "return foobar;"]
        TestData = collections.namedtuple("TestData", ["ident", "value"])
        expected = [
            TestData("x", 5),
            TestData("y", True),
            TestData("foobar", "foobar"),
        ]

        source = "\n".join(statements)
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(len(program.statements), len(statements))

        for i in range(len(program.statements)):
            stmt = program.statements[i]
            self.assertIsInstance(stmt, monkey.ReturnStatement)
            self.assertEqual(stmt.token_literal(), "return")
            self.check_literal_expression(stmt.return_value, expected[i].value)

    def test_integer_literal(self):
        statements = ["5;"]

        source = "\n".join(statements)
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(len(program.statements), len(statements))

        stmt = program.statements[0]
        self.assertIsInstance(stmt, monkey.ExpressionStatement)
        self.check_integer_literal(stmt.expression, 5)

    def test_function_literal(self):
        source = "fn(x, y) { x + y; }"
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(len(program.statements), 1)

        stmt = program.statements[0]
        self.assertIsInstance(stmt, monkey.ExpressionStatement)

        expr = stmt.expression
        self.assertIsInstance(expr, monkey.FunctionLiteral)
        self.assertEqual(len(expr.parameters), 2)
        self.check_literal_expression(expr.parameters[0], "x")
        self.check_literal_expression(expr.parameters[1], "y")

        body = stmt.expression.body
        self.assertIsInstance(body, monkey.BlockStatement)
        self.assertEqual(len(body.statements), 1)
        self.check_simple_infix_expression(
            body.statements[0].expression, "x", "+", "y"
        )

    def test_function_parameters(self):
        TestData = collections.namedtuple("TestData", ["source", "params"])
        tests = [
            TestData("fn() {}", []),
            TestData("fn(x) {}", ["x"]),
            TestData("fn(x, y, z) {}", ["x", "y", "z"]),
        ]

        for test in tests:
            l = monkey.Lexer(test.source)
            p = monkey.Parser(l)
            program = p.parse_program()
            self.assertEqual(len(program.statements), 1)

            stmt = program.statements[0]
            self.assertIsInstance(stmt, monkey.ExpressionStatement)
            expr = stmt.expression
            self.assertIsInstance(expr, monkey.FunctionLiteral)

            for i in range(len(expr.parameters)):
                self.check_literal_expression(
                    expr.parameters[i], test.params[i]
                )

    def test_parsing_prefix_expressions(self):
        TestData = collections.namedtuple(
            "TestData", ["source", "operator", "right_value"]
        )
        tests = [
            TestData("!5", "!", 5),
            TestData("-15", "-", 15),
            TestData("!true;", "!", True),
            TestData("!false;", "!", False),
        ]

        for test in tests:
            l = monkey.Lexer(test.source)
            p = monkey.Parser(l)
            program = p.parse_program()
            self.assertEqual(len(program.statements), 1)

            stmt = program.statements[0]
            self.assertIsInstance(stmt, monkey.ExpressionStatement)
            self.check_simple_prefix_expression(
                stmt.expression, test.operator, test.right_value,
            )

    def test_parsing_infix_expressions(self):
        TestData = collections.namedtuple(
            "TestData", ["source", "left_value", "operator", "right_value"]
        )
        tests = [
            TestData("5 + 5;", 5, "+", 5),
            TestData("5 - 5;", 5, "-", 5),
            TestData("5 * 5;", 5, "*", 5),
            TestData("5 / 5;", 5, "/", 5),
            TestData("5 > 5;", 5, ">", 5),
            TestData("5 < 5;", 5, "<", 5),
            TestData("5 == 5;", 5, "==", 5),
            TestData("5 != 5;", 5, "!=", 5),
            TestData("true == true", True, "==", True),
            TestData("true != false", True, "!=", False),
            TestData("false == false", False, "==", False),
        ]

        for test in tests:
            l = monkey.Lexer(test.source)
            p = monkey.Parser(l)
            program = p.parse_program()
            self.assertEqual(len(program.statements), 1)

            stmt = program.statements[0]
            self.assertIsInstance(stmt, monkey.ExpressionStatement)
            self.check_simple_infix_expression(
                stmt.expression,
                test.left_value,
                test.operator,
                test.right_value,
            )

    def test_operator_precedence_parsing(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("-a * b", "((-a) * b)"),
            TestData("!-a", "(!(-a))"),
            TestData("a + b + c", "((a + b) + c)"),
            TestData("a + b - c", "((a + b) - c)"),
            TestData("a * b * c", "((a * b) * c)"),
            TestData("a * b / c", "((a * b) / c)"),
            TestData("a + b / c", "(a + (b / c))"),
            TestData(
                "a + b * c + d / e - f", "(((a + (b * c)) + (d / e)) - f)"
            ),
            TestData("3 + 4; -5 * 5", "(3 + 4)((-5) * 5)"),
            TestData("5 > 4 == 3 < 4", "((5 > 4) == (3 < 4))"),
            TestData("5 < 4 != 3 > 4", "((5 < 4) != (3 > 4))"),
            TestData(
                "3 + 4 * 5 == 3 * 1 + 4 * 5",
                "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
            ),
            TestData("true", "true"),
            TestData("false", "false"),
            TestData("3 > 5 == false", "((3 > 5) == false)"),
            TestData("3 < 5 == true", "((3 < 5) == true)"),
            TestData("1 + (2 + 3) + 4", "((1 + (2 + 3)) + 4)"),
            TestData("(5 + 5) * 2", "((5 + 5) * 2)"),
            TestData("2 / (5 + 5)", "(2 / (5 + 5))"),
            TestData("-(5 + 5)", "(-(5 + 5))"),
            TestData("!(true == true)", "(!(true == true))"),
            TestData("a + add(b * c) + d", "((a + add((b * c))) + d)"),
            TestData(
                "add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8))",
                "add(a, b, 1, (2 * 3), (4 + 5), add(6, (7 * 8)))",
            ),
            TestData(
                "add(a + b + c * d / f + g)",
                "add((((a + b) + ((c * d) / f)) + g))",
            ),
        ]

        for test in tests:
            l = monkey.Lexer(test.source)
            p = monkey.Parser(l)
            program = p.parse_program()
            self.assertEqual(str(program), test.expected)

    def test_call_expression(self):
        source = "add(1, 2 * 3, 4 + 5)"
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(len(program.statements), 1)

        stmt = program.statements[0]
        self.assertIsInstance(stmt, monkey.ExpressionStatement)

        expr = stmt.expression
        self.assertIsInstance(expr, monkey.CallExpression)
        self.check_identifier(expr.function, "add")
        self.assertEqual(len(expr.arguments), 3)
        self.check_literal_expression(expr.arguments[0], 1)
        self.check_simple_infix_expression(expr.arguments[1], 2, "*", 3)
        self.check_simple_infix_expression(expr.arguments[2], 4, "+", 5)

    def test_if_expression(self):
        statements = ["if (x < y) { x }", "if (x < y) { x } else { y }"]
        source = "\n".join(statements)
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        self.assertEqual(len(program.statements), len(statements))

        for stmt in program.statements:
            self.assertIsInstance(stmt, monkey.ExpressionStatement)
            expr = stmt.expression
            self.assertIsInstance(expr, monkey.IfExpression)
            self.check_simple_infix_expression(expr.condition, "x", "<", "y")

            self.assertIsInstance(expr.consequence, monkey.BlockStatement)
            self.assertEqual(len(expr.consequence.statements), 1)
            self.check_identifier(
                expr.consequence.statements[0].expression, "x"
            )

            if expr.alternative != None:
                self.assertEqual(len(expr.alternative.statements), 1)
                self.check_identifier(
                    expr.alternative.statements[0].expression, "y"
                )


class EvalTest(unittest.TestCase):
    @staticmethod
    def eval(source: str) -> monkey.Object:
        l = monkey.Lexer(source)
        p = monkey.Parser(l)
        program = p.parse_program()
        return monkey.eval_ast(program, monkey.Environment())

    def check_null(self, obj: monkey.Object) -> None:
        self.assertIsInstance(obj, monkey.ObjectNull)

    def check_integer(self, obj: monkey.Object, expected: int) -> None:
        self.assertIsInstance(obj, monkey.ObjectInteger)
        self.assertEqual(typing.cast(monkey.ObjectInteger, obj).value, expected)

    def check_boolean(self, obj: monkey.Object, expected: bool) -> None:
        self.assertIsInstance(obj, monkey.ObjectBoolean)
        self.assertEqual(typing.cast(monkey.ObjectBoolean, obj).value, expected)

    def check_result(
        self, obj: monkey.Object, expected: typing.Union[None, int, bool]
    ) -> None:
        if isinstance(expected, type(None)):
            return self.check_null(obj)
        if isinstance(expected, int):
            return self.check_integer(obj, expected)
        if isinstance(expected, bool):
            return self.check_boolean(obj, expected)
        raise Exception(f"Unkown type {type(expected)}")

    def test_eval_integer(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("5", 5),
            TestData("10", 10),
            TestData("-5", -5),
            TestData("-10", -10),
            TestData("5 + 5 + 5 + 5 - 10", 10),
            TestData("2 * 2 * 2 * 2 * 2", 32),
            TestData("-50 + 100 - 50", 0),
            TestData("5 * 2 + 10", 20),
            TestData("5 + 2 * 10", 25),
            TestData("20 + 2 * -10", 0),
            TestData("50 / 2 * 2 + 10", 60),
            TestData("2 * (5 + 10)", 30),
            TestData("3 * 3 * 3 + 10", 37),
            TestData("3 * (3 * 3) + 10", 37),
            TestData("(5 + 10 * 2 + 15 / 3) * 2 + -10", 50),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_integer(evaluated, test.expected)

    def test_eval_boolean(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("true", True),
            TestData("false", False),
            TestData("1 < 2", True),
            TestData("1 > 2", False),
            TestData("1 < 1", False),
            TestData("1 > 1", False),
            TestData("1 == 1", True),
            TestData("1 != 1", False),
            TestData("1 == 2", False),
            TestData("1 != 2", True),
            TestData("true == true", True),
            TestData("false == false", True),
            TestData("true == false", False),
            TestData("false == true", False),
            TestData("true != true", False),
            TestData("false != false", False),
            TestData("true != false", True),
            TestData("false != true", True),
            TestData("(1 < 2) == true", True),
            TestData("(1 < 2) == false", False),
            TestData("(1 > 2) == true", False),
            TestData("(1 > 2) == false", True),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_boolean(evaluated, test.expected)

    def test_eval_function(self):
        source = "fn(x, y) { x + y + 2; };"
        evaluated = EvalTest.eval(source)
        self.assertIsInstance(evaluated, monkey.ObjectFunction)
        self.assertEqual(len(evaluated.parameters), 2)
        self.assertEqual(str(evaluated.parameters[0]), "x")
        self.assertEqual(str(evaluated.parameters[1]), "y")
        self.assertEqual(str(evaluated.body), "{ ((x + y) + 2) }")

    def test_prefix_bang(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("!true", False),
            TestData("!false", True),
            TestData("!5", False),
            TestData("!!true", True),
            TestData("!!false", False),
            TestData("!!5", True),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_boolean(evaluated, test.expected)

    def test_if_expression(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("if (true) { 10 }", 10),
            TestData("if (false) { 10 }", None),
            TestData("if (1) { 10 }", 10),
            TestData("if (1 < 2) { 10 }", 10),
            TestData("if (1 > 2) { 10 }", None),
            TestData("if (1 < 2) { 10 } else { 20 }", 10),
            TestData("if (1 > 2) { 10 } else { 20 }", 20),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_result(evaluated, test.expected)

    def test_return_statement(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("return 10;", 10),
            TestData("return 10; 9", 10),
            TestData("return 2 * 5; 9", 10),
            TestData("9; return 2 * 5; 9", 10),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_integer(evaluated, test.expected)

    def test_function_application(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("let identity = fn(x) { x;        }; identity(5);", 5),
            TestData("let identity = fn(x) { return x; }; identity(5);", 5),
            TestData("let double = fn(x) { x * 2; }; double(5);", 10),
            TestData("let add = fn(x, y) { x + y; }; add(3, 5);", 8),
            TestData(
                "let add = fn(x, y) { x + y; }; add(1 + 3, add(5, 7));", 16
            ),
            TestData("fn(x) { x; }(5);", 5),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_integer(evaluated, test.expected)

    def test_closures(self):
        source = "\n".join(
            [
                "let adder = fn(x) {",
                "  fn(y) { x + y }",
                "}",
                "let addtwo = adder(2)",
                "addtwo(3)",
            ]
        )
        evaluated = EvalTest.eval(source)
        self.check_integer(evaluated, 5)

    def test_let_statement(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("let a = 5; a;", 5),
            TestData("let a = 5 * 5; a;", 25),
            TestData("let a = 5; let b = a; b;", 5),
            TestData("let a = 5; let b = a; let c = a + b + 5; c;", 15),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.check_integer(evaluated, test.expected)

    def test_error_handling(self):
        TestData = collections.namedtuple("TestData", ["source", "expected"])
        tests = [
            TestData("5 + true;", "type mismatch: INTEGER + BOOLEAN"),
            TestData("5 + true; 5;", "type mismatch: INTEGER + BOOLEAN"),
            TestData("-true", "unknown operator: -BOOLEAN"),
            TestData("true + false", "unknown operator: BOOLEAN + BOOLEAN"),
            TestData(
                "5; true + false; 5;", "unknown operator: BOOLEAN + BOOLEAN"
            ),
            TestData(
                "if (10 > 1) { true + false }",
                "unknown operator: BOOLEAN + BOOLEAN",
            ),
            TestData(
                "if (true) { if (true) { true + false } }",
                "unknown operator: BOOLEAN + BOOLEAN",
            ),
            TestData("foobar", "identifier not found: foobar"),
        ]
        for test in tests:
            evaluated = EvalTest.eval(test.source)
            self.assertIsInstance(evaluated, monkey.ObjectError)
            self.assertEqual(evaluated.what, test.expected)


if __name__ == "__main__":
    unittest.main()

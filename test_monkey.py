#!/usr/bin/env python3

import collections
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
            TestData(monkey.Token.LET, "let"),
            TestData(monkey.Token.IDENT, "five"),
            TestData(monkey.Token.ASSIGN, "="),
            TestData(monkey.Token.INT, "5"),
            TestData(monkey.Token.SEMICOLON, ";"),
            #
            TestData(monkey.Token.LET, "let"),
            TestData(monkey.Token.IDENT, "ten"),
            TestData(monkey.Token.ASSIGN, "="),
            TestData(monkey.Token.INT, "10"),
            TestData(monkey.Token.SEMICOLON, ";"),
            #
            TestData(monkey.Token.LET, "let"),
            TestData(monkey.Token.IDENT, "add"),
            TestData(monkey.Token.ASSIGN, "="),
            TestData(monkey.Token.FUNCTION, "fn"),
            TestData(monkey.Token.LPAREN, "("),
            TestData(monkey.Token.IDENT, "x"),
            TestData(monkey.Token.COMMA, ","),
            TestData(monkey.Token.IDENT, "y"),
            TestData(monkey.Token.RPAREN, ")"),
            TestData(monkey.Token.LBRACE, "{"),
            TestData(monkey.Token.IDENT, "x"),
            TestData(monkey.Token.PLUS, "+"),
            TestData(monkey.Token.IDENT, "y"),
            TestData(monkey.Token.SEMICOLON, ";"),
            TestData(monkey.Token.RBRACE, "}"),
            TestData(monkey.Token.SEMICOLON, ";"),
            #
            TestData(monkey.Token.LET, "let"),
            TestData(monkey.Token.IDENT, "result"),
            TestData(monkey.Token.ASSIGN, "="),
            TestData(monkey.Token.IDENT, "add"),
            TestData(monkey.Token.LPAREN, "("),
            TestData(monkey.Token.IDENT, "five"),
            TestData(monkey.Token.COMMA, ","),
            TestData(monkey.Token.IDENT, "ten"),
            TestData(monkey.Token.RPAREN, ")"),
            TestData(monkey.Token.SEMICOLON, ";"),
            #
            TestData(monkey.Token.BANG, "!"),
            TestData(monkey.Token.MINUS, "-"),
            TestData(monkey.Token.SLASH, "/"),
            TestData(monkey.Token.ASTERISK, "*"),
            TestData(monkey.Token.INT, "5"),
            #
            TestData(monkey.Token.INT, "5"),
            TestData(monkey.Token.LT, "<"),
            TestData(monkey.Token.INT, "10"),
            TestData(monkey.Token.GT, ">"),
            TestData(monkey.Token.INT, "5"),
            #
            TestData(monkey.Token.IF, "if"),
            TestData(monkey.Token.LPAREN, "("),
            TestData(monkey.Token.INT, "5"),
            TestData(monkey.Token.LT, "<"),
            TestData(monkey.Token.INT, "10"),
            TestData(monkey.Token.RPAREN, ")"),
            TestData(monkey.Token.LBRACE, "{"),
            TestData(monkey.Token.RETURN, "return"),
            TestData(monkey.Token.TRUE, "true"),
            TestData(monkey.Token.SEMICOLON, ";"),
            TestData(monkey.Token.RBRACE, "}"),
            TestData(monkey.Token.ELSE, "else"),
            TestData(monkey.Token.LBRACE, "{"),
            TestData(monkey.Token.RETURN, "return"),
            TestData(monkey.Token.FALSE, "false"),
            TestData(monkey.Token.SEMICOLON, ";"),
            TestData(monkey.Token.RBRACE, "}"),
            #
            TestData(monkey.Token.INT, "10"),
            TestData(monkey.Token.EQ, "=="),
            TestData(monkey.Token.INT, "10"),
            #
            TestData(monkey.Token.INT, "10"),
            TestData(monkey.Token.NOT_EQ, "!="),
            TestData(monkey.Token.INT, "9"),
            #
            TestData(monkey.Token.EOF, ""),
        ]

        l = monkey.Lexer(source)
        for exp in expected:
            tok = l.next_token()
            self.assertEqual(tok.kind, exp.kind, tok.source_location)
            self.assertEqual(tok.literal, exp.literal, tok.source_location)


if __name__ == "__main__":
    unittest.main()

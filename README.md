# The Monkey Programming Language
Python 3 implementation of the interpreter for the Monkey language from Thorsten
Ball's [Writing an Interpreter in Go](https://interpreterbook.com/).

## Notable Implementation Deviations from the Book
+ The interpreter in the book uses the type `int64` for the `Value` field of the
`IntegerLiteral` struct. This interpreter uses Python's builtin arbitrary
precision `int` type to represent the value of integers.
+ The interpreter in the book propagates `nil` up the stack upon parse failure.
Rather than return the equivalent `None` this interpreter chooses to instead
raise a `ParseError` exception. This simplifies much of the parsing code in
exchange for stopping the parsing phase at the first parse failure.

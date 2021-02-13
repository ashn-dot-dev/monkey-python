# The Monkey Programming Language
Python 3 implementation of the interpreter for the Monkey language from Thorsten
Ball's [Writing an Interpreter in Go](https://interpreterbook.com/).

## Notable Implementation Deviations from the Book
+ The interpreter in the book uses the type `int64` for the `Value` field of the
`IntegerLiteral` struct. This interpreter uses Python's builtin arbitrary
precision `int` type to represent the value of integers.
+ The interpreter in the book propagates `nil` up the stack upon parse failure.
Rather than return the equivalent `None` this interpreter chooses to instead
raise a `ParseError` exception.
+ The interpreter in the book creates and reuses a single instance of the
boolean-true, boolean-false, and null objects. This interpreter creates new
instances of these object any time such values are produced during evaluation.
The book correctly notes that using a single instance of these objects is
(likely) faster and will save on resources, but in my opinion it creates much
uglier code, and performance is not a particularly important issue for a toy
interpreter.
+ The interpreter in the book returns `nil` as the result of a call to
`evaluator.Eval` when passed an `ast.LetStatement`. Returning `nil` from a
function that should always return an `object.Object` violates type safety, so
this interpreter chooses to instead return a null object.
+ The interpreter in the book creates a struct, `HashKey`, for holding the keys of
a hash data type. Rather than define a separate type, this interpreter uses
Python's builtin `__hash__` and `__eq__` methods for hash key comparison.

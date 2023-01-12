# AfterMath

AfterMath is a math parsing and evaluating library that can natively handle complex numbers.
This library rose from a need to handle arbirary user defined math expression. 
It is based on an AST that is evaluated (you can compile an AST into RPN too)


# How it works

At first the input is transformed into a token stream, then it applies some transformation to that stream to handle multi-argument functions, unary operator and implicit multiplication

This is all done with an iterator aproch, so no unecessary allocation are made

Then it consumes the modified token stream and uses the Shunting yard algoritm to produce an AST

# Future work

- Refine the library in general
- Document more things (with examples)
- Make a way to differentiate an AST
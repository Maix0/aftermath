# AfterMath

AfterMath is a math parsing and evaluating library that can natively handle complex numbers.
This library rose from a need to handle arbirary user defined math expression. 
It is based on an AST that is evaluated (you can compile an AST into RPN too)

This library also provides a way to differentiate an parsed expression with respect to a given variable (called binding)

# How it works

## Parsing 

At first the input is transformed into a token stream, then it applies some transformation to that stream to handle multi-argument functions, unary operator and implicit multiplication

This is all done with an iterator approach, so no unnecessary allocation are made

Then it consumes the modified token stream and uses the Shunting yard algorithm to produce an AST

## Differentiating

It applies the simples rules blindly. For example `d/dx 2x` is differentiated into `(d/dx 2) * x + (d/dx x) * 2` 

It is recommended to reduce the differentiated expression before evaluating it (see next part)

## Reducing

Here the term reducing is used to mean reducing the number of nodes.
This library provides a way to evaluates constant terms for the basic operation (+, -, *, /, %, ^)
This helps remove useless nodes (if multiplied by 0 for example) and allows the user to display a cleaner representation of the AST
The functions just do an DFS on the AST and collapses every node that result in an known value.


# Future work

- Make more examples
- Document more stuff

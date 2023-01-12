use crate::token_stream::{self, Token};

use bumpalo::Bump;

/// A token Tree representing a whole Expression
/// It lives inside an [Arena](bumpalo::Bump)
#[derive(Debug, PartialEq)]
pub enum Expr<'arena> {
    /// A real number
    RealNumber {
        /// The value of the real number
        val: f64,
    },
    /// An imaginary number
    ImaginaryNumber {
        /// The value of the imaginary number, without the `i` unit
        val: f64,
    },
    /// Complex number
    ComplexNumber {
        /// The value of the complex number's node
        val: num_complex::Complex64,
    },
    /// A variable
    Binding {
        /// The name of the variable
        name: &'arena mut str,
    },
    /// A function call, with an variable amount of arguments
    FunctionCall {
        /// Name of the function
        ident: &'arena mut str,
        /// List of argument in order they appeard
        args: bumpalo::collections::Vec<'arena, &'arena mut Expr<'arena>>,
    },
    /// An operation
    Operator {
        /// The operator
        op: Operator,
        /// Left side of the operation
        rhs: &'arena mut Expr<'arena>,
        /// Right side of the operation
        lhs: &'arena mut Expr<'arena>,
    },
}

impl<'arena> Expr<'arena> {
    /// Clone an AST with another backing [arena](bumpalo::Bump)
    #[allow(clippy::mut_from_ref)]
    pub fn clone_in(&self, arena: &'arena Bump) -> &'arena mut Self {
        use Expr::{Binding, ComplexNumber, FunctionCall, ImaginaryNumber, Operator, RealNumber};
        arena.alloc(match self {
            RealNumber { val } => RealNumber { val: *val },
            ImaginaryNumber { val } => ImaginaryNumber { val: *val },
            ComplexNumber { val } => ComplexNumber { val: *val },
            Binding { name } => Binding {
                name: arena.alloc_str(name),
            },
            FunctionCall { ident, args } => FunctionCall {
                ident: arena.alloc_str(ident),
                args: bumpalo::collections::FromIteratorIn::from_iter_in(
                    args.iter().map(|c| c.clone_in(arena)),
                    arena,
                ),
            },
            Operator { op, rhs, lhs } => Operator {
                op: *op,
                rhs: rhs.clone_in(arena),
                lhs: lhs.clone_in(arena),
            },
        })
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u16)]
#[allow(missing_docs)]
pub enum Operator {
    Plus = 1,
    Minus = 2,

    Multiply = 11,
    Divide = 12,
    Modulo = 13,

    Pow = 21,

    UnaryMinus = 31,
    UnaryPlus = 32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum Associativity {
    Right,
    Left,
}

impl Operator {
    #[must_use]
    /// Get a static str representation of the
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pow => "^",
            Self::Plus | Self::UnaryPlus => "+",
            Self::Minus | Self::UnaryMinus => "-",
            Self::Divide => "/",
            Self::Multiply => "*",
            Self::Modulo => "%",
        }
    }

    pub(crate) fn from_str(input: &str) -> Option<Self> {
        match input {
            "^" => Some(Self::Pow),
            "+" => Some(Self::Plus),
            "-" => Some(Self::Minus),
            "/" => Some(Self::Divide),
            "*" => Some(Self::Multiply),
            "%" => Some(Self::Modulo),
            _ => None,
        }
    }

    fn associativity(self) -> Associativity {
        match self {
            Self::Pow => Associativity::Left,
            _ => Associativity::Right,
        }
    }

    fn class(self) -> u8 {
        self as u8 / 10
    }
}

fn function_pass<'input>(
    mut iter: std::iter::Peekable<
        impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>> + 'input,
    >,
) -> impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>> + 'input {
    let mut need_sep = None;
    std::iter::from_fn(move || {
        if let Some(n) = need_sep.as_mut() {
            *n -= 1;
            if *n == 0u8 {
                need_sep = None;
                Some(Ok(token_stream::Token::Whitespace))
            } else {
                iter.next()
            }
        } else {
            let next = iter.next();
            match &next {
                Some(Ok(token_stream::Token::Ident(word))) if word.len() > 1 => {
                    if let Some(Ok(token_stream::Token::LeftParenthesis)) = iter.peek() {
                        need_sep = Some(2);
                    }
                }
                Some(Ok(token_stream::Token::Comma)) => {
                    need_sep = Some(1);
                }
                _ => {}
            };
            next
        }
    })
}

fn implicit_multiple_pass<'input>(
    mut iter: std::iter::Peekable<
        impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>> + 'input,
    >,
) -> impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>> + 'input {
    let mut need_sep = None;
    std::iter::from_fn(move || {
        if let Some(n) = need_sep.as_mut() {
            *n -= 1;
            if *n == 0u8 {
                need_sep = None;
                Some(Ok(token_stream::Token::Operator(Operator::Multiply)))
            } else {
                iter.next()
            }
        } else {
            let next = iter.next();
            if matches!(&next, Some(Ok(token_stream::Token::Ident(w))) if w.len() == 1)
                || matches!(
                    &next,
                    Some(Ok(
                        token_stream::Token::Literal(_) | token_stream::Token::RightParenthesis
                    ))
                )
            {
                if let Some(Ok(
                    token_stream::Token::LeftParenthesis
                    | token_stream::Token::Ident(_)
                    | token_stream::Token::Literal(_),
                )) = iter.peek()
                {
                    need_sep = Some(1);
                }
            }
            next
        }
    })
}

fn unary_pass<'input>(
    mut iter: std::iter::Peekable<
        impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>> + 'input,
    >,
) -> impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>> + 'input {
    let _next = iter.peek_mut().map(|next| match next {
        Ok(token_stream::Token::Operator(op @ Operator::Minus)) => {
            *op = Operator::UnaryMinus;
        }
        Ok(token_stream::Token::Operator(op @ Operator::Plus)) => {
            *op = Operator::UnaryPlus;
        }
        _ => (),
    });
    std::iter::from_fn(move || {
        let next = iter.next();
        if let Some(Ok(
            token_stream::Token::Operator(_)
            | token_stream::Token::Comma
            | token_stream::Token::Whitespace
            | token_stream::Token::LeftParenthesis,
        )) = next
        {
            match iter.peek_mut() {
                Some(Ok(token_stream::Token::Operator(op @ Operator::Minus))) => {
                    *op = Operator::UnaryMinus;
                }
                Some(Ok(token_stream::Token::Operator(op @ Operator::Plus))) => {
                    *op = Operator::UnaryPlus;
                }
                _ => (),
            }
        }
        next
    })
}

pub use token_stream::InvalidToken;

/// Error returned by [Expr::parse](Expr::parse)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError<'input> {
    /// An underlying token in the input is wrong
    InvalidToken(InvalidToken<'input>),
    /// At least one parenthesis is missing
    MissingParenthesis,
    /// At least one operator is missing
    MissingOperator,
    /// At least one operand is missing
    MissingOperand,
    /// An unknown error occured
    UnkownError,
}

impl<'input> From<InvalidToken<'input>> for BuildError<'input> {
    fn from(value: InvalidToken<'input>) -> Self {
        Self::InvalidToken(value)
    }
}

impl<'arena> std::fmt::Display for Expr<'arena> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            self.to_string_inner(f)
        } else {
            self.to_string_inner_min_parens(f, None)
        }
    }
}

// fn print<T: std::fmt::Debug + ?Sized>(t: &T, level: u16) {
//     println!("{:width$}[{level}]{t:?}", "", width = (level * 4) as usize);
// }

// thread_local! {static CURRENT_LEVEL: std::cell::Cell<u16> = 0.into();}

impl<'arena> Expr<'arena> {
    /// Create an AST from an input str
    ///
    /// # Errors
    /// This will error on any wrong input
    pub fn parse<'input, 'words: 'input + 'word, 'word: 'input>(
        arena: &'arena Bump,
        input: &'input str,
        reserved_words: &'words [&'word str],
    ) -> Result<&'arena mut Expr<'arena>, BuildError<'input>> {
        let iter = token_stream::parse_tokens(input, reserved_words);
        let iter = function_pass(iter.peekable());
        let iter = implicit_multiple_pass(iter.peekable());
        let iter = unary_pass(iter.peekable());
        let iter = iter.fuse();
        // let iter = iter.inspect(|t| print(&t, CURRENT_LEVEL.with(std::cell::Cell::get)));

        Self::parse_iter(arena, iter, &(true.into()))
    }

    fn parse_iter<'input, 'words: 'input + 'word, 'word: 'input>(
        arena: &'arena Bump,
        mut iter: impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>>,
        check_func_sep: &std::cell::Cell<bool>,
        // level: u16,
    ) -> Result<&'arena mut Expr<'arena>, BuildError<'input>> {
        let mut output = Vec::<&mut Self>::new();
        let mut operator = Vec::<Token<'input>>::new();
        let mut was_function_call = false;
        loop {
            if let Some(token) = iter.next() {
                //print(&format_args!("Output Buffer: {output:?}"), level);
                match token? {
                    Token::Whitespace => {
                        Self::handle_whitespace(arena, &mut iter, check_func_sep, &mut output)?;
                    }
                    Token::Literal(v) => output.push(arena.alloc(Expr::RealNumber { val: v })),
                    Token::Ident(name) if name.len() == 1 => {
                        output.push(arena.alloc(Expr::Binding {
                            name: arena.alloc_str(name),
                        }));
                    }
                    Token::Ident(name) => {
                        was_function_call = true;
                        // print("FUNCTION CALL", level);
                        output.push(arena.alloc(Expr::FunctionCall {
                            ident: arena.alloc_str(name),
                            args: bumpalo::collections::Vec::with_capacity_in(2, arena),
                        }));
                    }

                    Token::Comma => {
                        Self::handle_comma(arena, &mut operator, &mut output)?;
                    }
                    t @ Token::LeftParenthesis if !was_function_call => operator.push(t),
                    Token::LeftParenthesis => was_function_call = false,
                    Token::Operator(op) => {
                        Self::handle_operator(arena, op, &mut operator, &mut output)?;
                    }
                    Token::RightParenthesis => loop {
                        let Some(op) = operator.pop() else {
                            // print("Missing Parenthesis Error", level);
                            return Err(dbg!(BuildError::MissingParenthesis));
                        };
                        match op {
                            Token::LeftParenthesis => break,
                            Token::Operator(o @ (Operator::UnaryMinus | Operator::UnaryPlus)) => {
                                let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                                output.push(arena.alloc(Expr::Operator {
                                    op: o,
                                    lhs: arena.alloc(Expr::RealNumber { val: 0.0 }),
                                    rhs,
                                }));
                            }
                            Token::Operator(o) => {
                                let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                                let lhs = output.pop().ok_or(BuildError::MissingOperand)?;

                                output.push(arena.alloc(Expr::Operator { op: o, rhs, lhs }));
                            }
                            _ => (),
                        }
                    },
                }
            } else {
                for op in operator.into_iter().rev() {
                    match op {
                        Token::LeftParenthesis => return Err(BuildError::MissingParenthesis),
                        Token::Operator(o @ (Operator::UnaryMinus | Operator::UnaryPlus)) => {
                            let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                            output.push(arena.alloc(Expr::Operator {
                                op: o,
                                lhs: arena.alloc(Expr::RealNumber { val: 0.0 }),
                                rhs,
                            }));
                        }
                        Token::Operator(o) => {
                            let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                            let lhs = output.pop().ok_or(BuildError::MissingOperand)?;

                            output.push(arena.alloc(Expr::Operator { op: o, rhs, lhs }));
                        }
                        Token::Comma | Token::Whitespace => { /* No-op but still an operator */ }
                        _ => (),
                    }
                }
                break;
            }
        }
        //print(&format_args!("End: {}", output.len()), level);
        output.pop().ok_or(match output.len() {
            0 => BuildError::UnkownError,
            _ => BuildError::MissingOperator,
        })
    }

    fn handle_comma<'input>(
        arena: &'arena Bump,
        operator: &mut Vec<Token>,
        output: &mut Vec<&'arena mut Self>,
    ) -> Result<&'arena mut Self, BuildError<'input>> {
        loop {
            let Some(op) = operator.pop() else {
                // print("Missing Parenthesis Error", level);
                break;
            };
            match op {
                Token::LeftParenthesis => break,
                Token::Operator(o @ (Operator::UnaryMinus | Operator::UnaryPlus)) => {
                    let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                    output.push(arena.alloc(Expr::Operator {
                        op: o,
                        lhs: arena.alloc(Expr::RealNumber { val: 0.0 }),
                        rhs,
                    }));
                }
                Token::Operator(o) => {
                    let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                    let lhs = output.pop().ok_or(BuildError::MissingOperand)?;

                    output.push(arena.alloc(Expr::Operator { op: o, rhs, lhs }));
                }
                _ => (),
            }
        }
        // print(&format_args!("Comma: {}", output.len()), level);
        output.pop().ok_or(match output.len() {
            0 => BuildError::UnkownError,
            _ => BuildError::MissingOperator,
        })
    }

    fn handle_operator<'input>(
        arena: &'arena Bump,
        op1: Operator,
        operator: &mut Vec<Token>,
        output: &mut Vec<&'arena mut Self>,
    ) -> Result<(), BuildError<'input>> {
        loop {
            let Some(peek) = operator.last() else {break;};
            match peek {
                Token::Operator(op2)
                    if op2.class() > op1.class()
                        || (op1.class() == op2.class()
                            && op1.associativity() == Associativity::Left) =>
                {
                    let op = operator.pop().unwrap();
                    match op {
                        Token::Operator(o @ (Operator::UnaryMinus | Operator::UnaryPlus)) => {
                            let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                            output.push(arena.alloc(Expr::Operator {
                                op: o,
                                lhs: arena.alloc(Expr::RealNumber { val: 0.0 }),
                                rhs,
                            }));
                        }
                        Token::Operator(o) => {
                            let rhs = output.pop().ok_or(BuildError::MissingOperand)?;
                            let lhs = output.pop().ok_or(BuildError::MissingOperand)?;

                            output.push(arena.alloc(Expr::Operator { op: o, rhs, lhs }));
                        }
                        _ => (),
                    }
                }
                _ => break,
            }
        }
        operator.push(Token::Operator(op1));
        Ok(())
    }

    #[allow(trivial_casts)]
    fn handle_whitespace<'input>(
        arena: &'arena Bump,
        iter: &mut impl Iterator<Item = Result<Token<'input>, InvalidToken<'input>>>,
        check_func_sep: &std::cell::Cell<bool>,
        output: &mut [&'arena mut Self],
    ) -> Result<(), BuildError<'input>> {
        check_func_sep.set(false);
        let parens_count = std::cell::Cell::new(1u16);
        let child_check_func_sep = std::cell::Cell::new(true);
        let error = std::cell::Cell::new(false);
        let mut sub_iter = iter
            .by_ref()
            .inspect(|t| {
                // print(
                //     &((child_whitespace.get() && !matches!(t, &Ok(Token::Comma)))
                //         || parens_count.get() != 0),
                // );
                match t {
                    Ok(Token::LeftParenthesis) => {
                        parens_count.set(if let Some(n) = parens_count.get().checked_add(1) {
                            n
                        } else {
                            error.set(true);
                            255
                        });
                    }
                    Ok(Token::RightParenthesis) => {
                        parens_count.set(if let Some(n) = parens_count.get().checked_sub(1) {
                            n
                        } else {
                            error.set(true);
                            255
                        });
                    }
                    _ => (),
                }
            })
            .take_while(|_| parens_count.get() != 0 || !child_check_func_sep.get());
        // print("LEVEL START", level);
        let ast = Self::parse_iter(
            arena,
            &mut sub_iter as &mut dyn Iterator<Item = Result<Token<'_>, InvalidToken<'_>>>,
            &child_check_func_sep,
            // level + 1,
        );
        // print(
        //     &format_args!(
        //         "LEVEL END: {}",
        //         if error.get() { "Error" } else { "No Error" }
        //     ),
        //     level + 1,
        // );
        if error.get() {
            return Err(dbg!(BuildError::UnkownError));
        }
        check_func_sep.set(true);
        match output.last_mut() {
            Some(Expr::FunctionCall { args, .. }) => {
                args.push(ast?);
            }
            _ => {
                // print(&output, level);
                return Err(BuildError::MissingOperator);
            }
        }
        Ok(())
    }
}

/// The real implementation of display
impl<'arena> Expr<'arena> {
    fn to_string_inner_min_parens(
        &self,
        buf: &mut impl std::fmt::Write,
        parent_precedence: Option<u8>,
    ) -> std::fmt::Result {
        match self {
            Expr::FunctionCall { ident, args } => {
                write!(buf, "{ident}(")?;
                for arg in args.iter().take(args.len() - 1) {
                    arg.to_string_inner_min_parens(buf, None)?;
                    write!(buf, ", ")?;
                }
                if let Some(arg) = args.last() {
                    arg.to_string_inner_min_parens(buf, None)?;
                }
                write!(buf, ")")?;
            }
            Expr::RealNumber { val } if val.is_sign_negative() => write!(buf, "({val})")?,
            Expr::RealNumber { val } => write!(buf, "{val}")?,
            Expr::ImaginaryNumber { val } if val.is_sign_negative() => write!(buf, "({val}i)")?,
            Expr::ImaginaryNumber { val } => write!(buf, "{val}i")?,
            Expr::ComplexNumber { val }
                if val.re.is_sign_negative() || val.im.is_sign_negative() =>
            {
                write!(buf, "({val})")?;
            }
            Expr::ComplexNumber { val } => write!(buf, "{val}")?,
            Expr::Binding { name } => write!(buf, "{name}")?,
            Expr::Operator {
                op: op @ (Operator::UnaryMinus | Operator::UnaryPlus),
                rhs,
                ..
            } => {
                if parent_precedence.map_or(false, |p| op.class() < p) {
                    write!(buf, "(")?;
                    write!(buf, "{}", op.as_str())?;
                    rhs.to_string_inner_min_parens(buf, Some(op.class()))?;
                    write!(buf, ")")?;
                } else {
                    write!(buf, "{}", op.as_str())?;
                    rhs.to_string_inner_min_parens(buf, Some(op.class()))?;
                }
            }
            Expr::Operator { op, rhs, lhs } => {
                if parent_precedence.map_or(false, |p| op.class() < p) {
                    write!(buf, "(")?;
                    lhs.to_string_inner_min_parens(buf, Some(op.class()))?;
                    write!(buf, " {} ", op.as_str())?;
                    rhs.to_string_inner_min_parens(buf, Some(op.class()))?;
                    write!(buf, ")")?;
                } else {
                    lhs.to_string_inner_min_parens(buf, Some(op.class()))?;
                    write!(buf, " {} ", op.as_str())?;
                    rhs.to_string_inner_min_parens(buf, Some(op.class()))?;
                }
            }
        }
        Ok(())
    }

    fn to_string_inner(&self, buf: &mut impl std::fmt::Write) -> std::fmt::Result {
        match self {
            Expr::FunctionCall { ident, args } => {
                write!(buf, "{ident}(")?;
                for arg in args.iter().take(args.len() - 1) {
                    arg.to_string_inner(buf)?;
                    write!(buf, ", ")?;
                }
                if let Some(arg) = args.last() {
                    arg.to_string_inner(buf)?;
                }
                write!(buf, ")")?;
            }
            Expr::RealNumber { val } => write!(buf, "({val})")?,
            Expr::ImaginaryNumber { val } => write!(buf, "({val}i)")?,
            Expr::ComplexNumber { val } => write!(buf, "({val})")?,
            Expr::Binding { name } => write!(buf, "{name}")?,
            Expr::Operator {
                op: op @ (Operator::UnaryMinus | Operator::UnaryPlus),
                rhs,
                ..
            } => {
                write!(buf, "({}", op.as_str())?;
                rhs.to_string_inner(buf)?;
                write!(buf, ")")?;
            }
            Expr::Operator { op, rhs, lhs } => {
                write!(buf, "(")?;
                lhs.to_string_inner(buf)?;
                write!(buf, " {} ", op.as_str())?;
                rhs.to_string_inner(buf)?;
                write!(buf, ")")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::token_stream::{
        stream_to_string,
        Token::{Comma, Ident, LeftParenthesis, Literal, RightParenthesis, Whitespace},
    };
    use super::*;

    #[test]
    fn function_sep() {
        let input = "max(1, 5)";
        let stream = token_stream::parse_tokens(input, token_stream::RESTRICTED_WORD);
        let first_pass = function_pass(stream.peekable());

        let res: Result<Vec<_>, _> = first_pass.collect();

        assert_eq!(
            res.unwrap(),
            vec![
                Ident("max"),
                LeftParenthesis,
                Whitespace,
                Literal(1.0),
                Comma,
                Whitespace,
                Literal(5.0),
                RightParenthesis
            ]
        );
    }

    #[test]
    fn implicit_multiple() {
        let input = "a(1) + 1(1) + 1a + aa + (1)(1)1";
        let stream = token_stream::parse_tokens(input, token_stream::RESTRICTED_WORD);
        let first_pass = implicit_multiple_pass(stream.peekable());

        let iter = first_pass
            .flat_map(|token| [Ok(Whitespace), token].into_iter())
            .skip(1);
        let res = stream_to_string(iter);

        assert_eq!(
            res.unwrap(),
            "a * ( 1 ) + 1 * ( 1 ) + 1 * a + a * a + ( 1 ) * ( 1 ) * 1"
        );
    }

    #[test]
    fn unary() {
        let input = "-(-1) + -(+a)";

        let stream = token_stream::parse_tokens(input, token_stream::RESTRICTED_WORD);
        let iter = stream
            .flat_map(|token| [Ok(Whitespace), token].into_iter())
            .skip(1);
        let res = stream_to_string(iter);

        assert_eq!(res.unwrap(), "- ( - 1 ) + - ( + a )");
    }
    #[cfg(test)]
    mod ast {
        use super::Expr;

        macro_rules! ast_test {
            ($name:ident: $input:literal $(=)?) => {
                #[test]
                fn $name() {
                    let arena = bumpalo::Bump::with_capacity(1024);
                    let expr = Expr::parse(&arena, $input, super::token_stream::RESTRICTED_WORD);

                    let expr = expr.unwrap();

                    dbg!(expr.to_string());
                    panic!();
                }
            };

            ($name:ident: $input:literal = $output:literal) => {
                #[test]
                fn $name() {
                    println!("{}", $input);
                    let arena = bumpalo::Bump::with_capacity(1024);
                    let expr = Expr::parse(&arena, $input, super::token_stream::RESTRICTED_WORD);

                    let expr = expr.unwrap();
                    println!("==================================================");

                    let same_expr =
                        Expr::parse(&arena, $output, super::token_stream::RESTRICTED_WORD);

                    let same_expr = same_expr.unwrap();

                    assert_eq!(expr.to_string(), $output);

                    assert_eq!(same_expr.to_string(), $output);
                }
            };
        }

        ast_test! {simple_addition: "1+1" = "1 + 1"}
        ast_test! {simple_substraction: "1-1" = "1 - 1"}
        ast_test! {simple_multiplication: "1*1" = "1 * 1"}
        ast_test! {simple_division: "1/1" = "1 / 1"}
        ast_test! {simple_modulo: "1%1" = "1 % 1"}
        ast_test! {simple_unary_minus: "--1" = "--1"}
        ast_test! {simple_unary_plus: "++1" = "++1"}

        ast_test! {mult1: "4 + 2 * 3" = "4 + 2 * 3"}
        ast_test! {implicit_multi1: "2a2" = "2 * a * 2"}

        ast_test! {complex1: "3 + 4 * 2 / (1 - 5) ^ 2 ^ 3" = "3 + 4 * 2 / (1 - 5) ^ 2 ^ 3"}

        ast_test! {function: "max(exp(7, 10), 3)" = "max(exp(7, 10), 3)"}
        ast_test! {function2: "max(2exp(7, 10), 3)" = "max(2 * exp(7, 10), 3)"}
        ast_test! {function3:
        "exp(exp(exp(exp(exp(exp(1), exp(1))) + 56, 2exp(exp(exp(exp(exp(1), exp(1))), exp(exp(exp(1), exp(exp(exp(1), exp(1))))))))), exp(exp(exp(exp(exp(exp(exp(5 + 7 + 54), exp(5 + 7 + 54))), exp(5 + 7 + 54))), exp(5 + 7 + 54))))" =
        "exp(exp(exp(exp(exp(exp(1), exp(1))) + 56, 2 * exp(exp(exp(exp(exp(1), exp(1))), exp(exp(exp(1), exp(exp(exp(1), exp(1))))))))), exp(exp(exp(exp(exp(exp(exp(5 + 7 + 54), exp(5 + 7 + 54))), exp(5 + 7 + 54))), exp(5 + 7 + 54))))"}
        ast_test! {function4: "max(1, 2, 4, 4, 5, 7, 30)" = "max(1, 2, 4, 4, 5, 7, 30)"}
    }
}

use std::borrow::{Borrow, Cow};

use std::collections::HashMap;
pub struct Context {
    variables: HashMap<Cow<'static, str>, num_complex::Complex64>,
    funcs: HashMap<Cow<'static, str>, std::sync::Arc<dyn Func + Send + Sync>>,
    func_names: std::collections::BTreeSet<(usize, Cow<'static, str>)>,
}

impl std::default::Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Creates a new empty context
    #[must_use]
    pub fn new() -> Self {
        Context {
            variables: HashMap::default(),
            funcs: HashMap::default(),
            func_names: std::collections::BTreeSet::default(),
        }
    }

    #[must_use]
    /// Get an iterator over the reserved names for this context
    /// You should only call this function once and cache its result
    /// But you *can* call it multiples times
    pub fn get_reserved_names(&self) -> Vec<&str> {
        self.func_names
            .iter()
            .map(|(_, s)| s.borrow())
            .rev()
            .collect()
    }

    /// # Errors
    ///
    /// Return an error if the requested binding isn't found in this context
    pub fn get_binding(&self, name: &str) -> Result<&num_complex::Complex64, CalcError> {
        self.variables.get(name).ok_or(CalcError::MissingBindings)
    }

    /// # Errors
    ///
    /// Return an error if the requested function isn't found in this context
    pub fn get_func(
        &self,
        name: &str,
    ) -> Result<&std::sync::Arc<dyn Func + Send + Sync>, CalcError> {
        self.funcs.get(name).ok_or(CalcError::MissingBindings)
    }
    /// Insert the given function into the context, overwriting if the function already existed
    pub fn insert_func(
        &mut self,
        name: Cow<'static, str>,
        func: std::sync::Arc<dyn Func + Send + Sync>,
    ) {
        self.funcs.insert(name.clone(), func);
        self.func_names.insert((name.len(), name));
    }
    /// Insert the given binding into the context, overwriting if the binding already existed
    pub fn insert_binding(&mut self, name: Cow<'static, str>, binding: num_complex::Complex64) {
        self.variables.insert(name, binding);
    }

    /// Evaluate an AST in the current Context
    ///
    /// # Errors
    ///
    /// This will return an error if any of the operation return an error
    pub fn eval<'expr: 'arena, 'arena>(
        &self,
        expr: &'expr crate::Expr<'arena>,
    ) -> Result<num_complex::Complex64, CalcError> {
        #![allow(clippy::enum_glob_use)]
        use crate::expr::Expr::*;
        use crate::expr::Operator::*;
        Ok(match expr {
            RealNumber { val } => num_complex::Complex { re: *val, im: 0.0 },
            ImaginaryNumber { val } => num_complex::Complex { re: 0.0, im: *val },
            ComplexNumber { val } => *val,
            Binding { name } => *self.get_binding(name)?,

            Operator { op, rhs, lhs } => match op {
                Plus => self.eval(lhs)? + self.eval(rhs)?,
                Minus => self.eval(lhs)? - self.eval(rhs)?,
                Multiply => self.eval(lhs)? * self.eval(rhs)?,
                Divide =>
                /* TODO: evaluate if there is a need to check for NaNs */
                {
                    self.eval(lhs)? / self.eval(rhs)?
                }
                Modulo =>
                /* TODO: evaluate if there is a need to check for NaNs */
                {
                    self.eval(lhs)? % self.eval(rhs)?
                }
                Pow => self.eval(lhs)?.powc(self.eval(rhs)?),
                UnaryMinus => -self.eval(lhs)?,
                UnaryPlus => self.eval(lhs)?,
            },
            FunctionCall { ident, args } => {
                let func = self.get_func(ident)?;
                func.call(self, Arguments::from_slice(args, self))?
            }
        })
    }

    #[allow(clippy::missing_panics_doc)]
    /// Evaluate an RPN sequence in the current context
    ///
    /// # Errors
    ///
    /// This will return an error on three separate instances:
    ///     - A User-Function has returned an error
    ///     - A bindings is missing in the current context
    ///     - A User-Function is missing in the current context
    pub fn eval_rpn<'expr: 'arena, 'arena>(
        &self,
        rpn: &'expr rpn::RpnExpr<'arena>,
    ) -> Result<num_complex::Complex64, CalcError> {
        let mut val_stack = Vec::with_capacity(rpn.seq.len() / 2);
        for token in &rpn.seq {
            match token {
                rpn::RpnToken::Literal(l) => val_stack.push(*l),
                rpn::RpnToken::Binding(name) => val_stack.push(*self.get_binding(name)?),
                rpn::RpnToken::Function(name, len) => {
                    let val = self.get_func(name)?.call(
                        self,
                        Arguments {
                            iter: ArgumentIterImpl::RPNIter({
                                let start = val_stack.len() - *len as usize;
                                val_stack.drain(start..)
                            }),
                            len: *len as usize,
                        },
                    )?;
                    val_stack.push(val);
                }
                rpn::RpnToken::Op(op) => {
                    let lhs = val_stack.pop().unwrap();
                    let rhs = val_stack.pop().unwrap();
                    val_stack.push(match op {
                        rpn::Operator::Plus => lhs + rhs,
                        rpn::Operator::Minus => lhs - rhs,
                        rpn::Operator::Mul => lhs * rhs,
                        rpn::Operator::Div => lhs / rhs,
                        rpn::Operator::Mod => lhs % rhs,
                        rpn::Operator::Pow => lhs.powc(rhs),
                    });
                }
            }
        }
        Ok(val_stack.pop().unwrap())
    }
}

#[derive(Debug)]
pub enum CalcError {
    Boxed(Box<dyn std::error::Error + Send>),
    InvalidArgumentCount,
    InvalidInput,
    DivisionByZero,
    MissingFunction,
    MissingBindings,
}

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalcError::Boxed(b) => b.fmt(f),
            CalcError::DivisionByZero => write!(f, " A division by zero occured"),
            CalcError::InvalidArgumentCount => {
                write!(f, "A function recieved an illegal number of arguments")
            }
            CalcError::InvalidInput => write!(f, "A function recieved an illegal input"),
            CalcError::MissingBindings => {
                write!(f, "A expression tried to use an binding that isn't defined")
            }
            CalcError::MissingFunction => write!(
                f,
                "A expression tried to use an function that isn't defined"
            ),
        }
    }
}

impl std::error::Error for CalcError {}

type RefRefMutToRef<T> = for<'inner, 'outer> fn(&'outer &'inner mut T) -> &'outer T;
type ExprToComplexResult<'arena, 'context, 'expr> = fn(
    (&'expr crate::Expr<'arena>, &'context Context),
)
    -> Result<num_complex::Complex64, CalcError>;

pub struct Arguments<'context, 'arena, 'expr: 'arena, 'f> {
    iter: ArgumentIterImpl<'context, 'arena, 'expr, 'f>,
    len: usize,
}

#[allow(clippy::type_complexity)]
enum ArgumentIterImpl<'context, 'arena, 'expr: 'arena, 'v> {
    ASTIter(
        std::iter::Map<
            std::iter::Zip<
                std::iter::Map<
                    std::slice::Iter<'arena, &'arena mut crate::Expr<'arena>>,
                    RefRefMutToRef<crate::Expr<'arena>>,
                >,
                std::iter::Repeat<&'context Context>,
            >,
            ExprToComplexResult<'arena, 'context, 'expr>,
        >,
    ),
    RPNIter(std::vec::Drain<'v, num_complex::Complex64>),
}

impl<'context, 'arena, 'expr: 'arena, 'v> Iterator
    for ArgumentIterImpl<'context, 'arena, 'expr, 'v>
{
    type Item = Result<num_complex::Complex64, CalcError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::ASTIter(i) => i.next(),
            Self::RPNIter(i) => i.next().map(Ok),
        }
    }
}

impl<'context, 'arena, 'expr: 'arena, 'v> Iterator for Arguments<'context, 'arena, 'expr, 'v> {
    type Item = Result<num_complex::Complex64, CalcError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'context, 'arena, 'expr: 'arena, 'v> Arguments<'context, 'arena, 'expr, 'v> {
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    fn from_slice(
        slice: &'arena [&'arena mut crate::Expr<'arena>],
        context: &'context Context,
    ) -> Self {
        let first_closure: RefRefMutToRef<crate::Expr<'arena>> = |e| &**e;
        let second_closure: ExprToComplexResult<'arena, 'context, 'expr> = |(e, c)| c.eval(e);
        Self {
            iter: ArgumentIterImpl::ASTIter(
                slice
                    .iter()
                    .map(first_closure)
                    .zip(std::iter::repeat(context))
                    .map(second_closure),
            ),
            len: slice.len(),
        }
    }

    /// Consume the inner iterator and returns it as a Vec
    ///
    /// # Errors
    ///
    /// return an error on the first error that it encounters
    pub fn into_vec(self) -> Result<Vec<num_complex::Complex64>, CalcError> {
        self.collect()
    }
}

/// Describe an mathematical function that can be used in the expressions evaluated
pub trait Func {
    /// The entry point of a user-defined function
    /// it will be called when you it is encountered
    ///
    /// The arguments are lazily calculated (but can be calculated at once and returned as a vec using [`into_vec`](Arguments::into_vec))
    ///
    /// # Errors
    ///
    /// You can return errors (and should if needed)
    /// The [`CalcError`](CalcError) enum provides an boxed error variant to return custom error
    fn call(
        &self,
        context: &Context,
        args: Arguments<'_, '_, '_, '_>,
    ) -> Result<num_complex::Complex64, CalcError>;
}

pub mod funcs {
    #![allow(clippy::wildcard_imports)]
    use super::*;
    macro_rules! define_func {
        ($($sname:ident($fname:literal): [$($args_name:ident),*] => $code:block);*$(;)?) => {
            $(pub struct $sname;
                impl $sname {
                    pub const NAME: &str = $fname;

                    pub fn add_to_context(ctx: &mut Context) {
                        ctx.insert_func(
                            std::borrow::Cow::Borrowed(Self::NAME),
                            std::sync::Arc::new(Self) as std::sync::Arc<dyn Func + Send + Sync>,
                        )
                }
            }

            impl Func for $sname {
                fn call(
                    &self,
                    _: &Context,
                    mut args: Arguments<'_, '_, '_, '_>,
                ) -> Result<num_complex::Complex64, CalcError> {
                    if [$(|$args_name: ()| $args_name),*].len() == args.len() {
                        $(let $args_name = args.next().ok_or(CalcError::InvalidArgumentCount)??;)*
                        Ok($code)
                    } else {
                        Err(CalcError::InvalidArgumentCount)
                    }
                }
            })*
        };
    }
    pub fn add_trigonometry(ctx: &mut Context) {
        Sin::add_to_context(ctx);
        Asin::add_to_context(ctx);
        Cos::add_to_context(ctx);
        Acos::add_to_context(ctx);
        Tan::add_to_context(ctx);
        Atan::add_to_context(ctx);
    }
    pub fn add_hyperbolic_trigonometry(ctx: &mut Context) {
        Sinh::add_to_context(ctx);
        Asinh::add_to_context(ctx);
        Cosh::add_to_context(ctx);
        Acosh::add_to_context(ctx);
        Tanh::add_to_context(ctx);
        Atanh::add_to_context(ctx);
    }
    pub fn add_complex(ctx: &mut Context) {
        Arg::add_to_context(ctx);
        Norm::add_to_context(ctx);
        Conj::add_to_context(ctx);
    }

    pub fn add_real_functions(ctx: &mut Context) {
        Exp::add_to_context(ctx);
        Ln::add_to_context(ctx);
        Sqrt::add_to_context(ctx);
        Cbrt::add_to_context(ctx);
        Log::add_to_context(ctx);
    }

    pub fn add_all_to_context(ctx: &mut Context) {
        add_trigonometry(ctx);
        add_hyperbolic_trigonometry(ctx);
        add_complex(ctx);
        add_real_functions(ctx);
    }

    define_func! {
        // Trigonometry function
        Sin("sin"):     [arg]       => {arg.sin()};
        Asin("asin"):   [arg]       => {arg.asin()};
        Cos("cos"):     [arg]       => {arg.cos()};
        Acos("acos"):   [arg]       => {arg.acos()};
        Tan("tan"):     [arg]       => {arg.tan()};
        Atan("atan"):   [arg]       => {arg.atan()};

        // Trigonometry function (hyperbolic)
        Sinh("sinh"):   [arg]       => {arg.sinh()};
        Asinh("asinh"): [arg]       => {arg.asinh()};
        Cosh("cosh"):   [arg]       => {arg.cosh()};
        Acosh("acosh"): [arg]       => {arg.acosh()};
        Tanh("tanh"):   [arg]       => {arg.tanh()};
        Atanh("atanh"): [arg]       => {arg.atanh()};

        // Complex Specific functions
        Arg("arg"):     [arg]       => {arg.arg().into()};
        Norm("norm"):   [arg]       => {arg.norm().into()};
        Conj("conj"):   [arg]       => {arg.conj()};
        // Normal functions
        Exp("exp"):     [arg]       => {arg.exp()};
        Ln("ln"):       [arg]       => {arg.ln()};
        Sqrt("sqrt"):   [arg]       => {arg.sqrt()};
        Cbrt("cbrt"):   [arg]       => {arg.cbrt()};
        Log("log"):     [arg, base] => {arg.log(base.re)}


    }
}

pub mod rpn {
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum Operator {
        Plus,
        Minus,
        Mul,
        Div,
        Mod,
        Pow,
    }

    #[derive(Clone, Debug)]
    pub(crate) enum RpnToken<'arena> {
        Literal(num_complex::Complex64),
        Binding(&'arena str),
        Function(&'arena str, u16),
        Op(Operator),
    }

    #[allow(clippy::module_name_repetitions)]
    /// A complied AST into a linear stream of operation
    /// Evaluating this stream of token is faster than evaluating an AST since the memory is linear
    ///
    /// It still retains the flexiblity and needs of the AST by keeping the bindings and functions as identifier
    pub struct RpnExpr<'arena> {
        pub(crate) seq: Vec<RpnToken<'arena>>,
    }

    impl<'arena> RpnExpr<'arena> {
        /// Create an RPN token from an AST
        pub fn from_ast(arena: &'arena bumpalo::Bump, ast: &crate::Expr<'_>) -> Self {
            let mut rpn = Self {
                seq: Vec::with_capacity(32),
            };
            Self::from_ast_inner(arena, ast, &mut rpn);
            rpn
        }

        #[allow(clippy::enum_glob_use)]
        fn from_ast_inner(
            arena: &'arena bumpalo::Bump,
            ast: &crate::Expr<'_>,
            rpn: &mut RpnExpr<'arena>,
        ) {
            use crate::Expr::*;
            match ast {
                RealNumber { val } => rpn.seq.push(RpnToken::Literal(num_complex::Complex64 {
                    re: *val,
                    im: 0.0,
                })),
                ImaginaryNumber { val } => {
                    rpn.seq.push(RpnToken::Literal(num_complex::Complex64 {
                        re: 0.0,
                        im: *val,
                    }));
                }
                ComplexNumber { val } => rpn.seq.push(RpnToken::Literal(*val)),
                Binding { name } => rpn.seq.push(RpnToken::Binding(arena.alloc_str(name))),
                FunctionCall { ident, args } => {
                    for expr in args {
                        Self::from_ast_inner(arena, expr, rpn);
                    }
                    rpn.seq.push(RpnToken::Function(
                        arena.alloc_str(ident),
                        args.len()
                            .try_into()
                            .expect("Number of argument overflowed an u16"),
                    ));
                }
                Operator { op, rhs, lhs } => {
                    Self::from_ast_inner(arena, lhs, rpn);
                    Self::from_ast_inner(arena, rhs, rpn);
                    rpn.seq.push(RpnToken::Op(match op {
                        crate::Operator::Minus | crate::Operator::UnaryMinus => {
                            self::Operator::Minus
                        }
                        crate::Operator::Plus | crate::Operator::UnaryPlus => self::Operator::Plus,
                        crate::Operator::Multiply => self::Operator::Mul,
                        crate::Operator::Divide => self::Operator::Div,
                        crate::Operator::Modulo => self::Operator::Mod,
                        crate::Operator::Pow => self::Operator::Pow,
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    macro_rules! check_complex {
        ($lhs:ident, $rhs:ident) => {
            ($lhs.re - $rhs.re).abs() < f64::EPSILON && ($lhs.im - $rhs.im).abs() < f64::EPSILON
        };
    }

    macro_rules! make_test {
        ($name:ident: $input:literal => $res:block) => {
            #[test]
            fn $name() {
                let input: &'static str = $input;
                let bump = bumpalo::Bump::with_capacity(512);
                let res: num_complex::Complex64 = ($res).into();

                let mut ctx = super::Context::new();
                super::funcs::add_all_to_context(&mut ctx);
                let ast = crate::Expr::parse(&bump, input, &ctx.get_reserved_names()).unwrap();
                let rpn = super::rpn::RpnExpr::from_ast(&bump, ast);

                let res_ast = ctx.eval(&ast).unwrap();
                let res_rpn = ctx.eval_rpn(&rpn).unwrap();

                assert!(check_complex!(res_ast, res_rpn));
                assert!(check_complex!(res_ast, res));
            }
        };
    }

    make_test! {simple_addition: "1 + 1" => {1.0 + 1.0}}
    make_test! {function_call: "sin(1)" => {1f64.sin()}}
}

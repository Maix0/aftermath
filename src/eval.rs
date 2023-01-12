use std::borrow::Cow;

use std::collections::HashMap;
pub struct Context {
    variables: HashMap<Cow<'static, str>, num_complex::Complex64>,
    funcs: HashMap<Cow<'static, str>, std::sync::Arc<dyn Func + Send + Sync>>,
}

impl Context {
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
        self.funcs.insert(name, func);
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

#[allow(clippy::type_complexity)]
pub struct Arguments<'context, 'arena, 'expr: 'arena> {
    iter: std::iter::Map<
        std::iter::Zip<
            std::iter::Map<
                std::slice::Iter<'arena, &'arena mut crate::Expr<'arena>>,
                RefRefMutToRef<crate::Expr<'arena>>,
            >,
            std::iter::Repeat<&'context Context>,
        >,
        ExprToComplexResult<'arena, 'context, 'expr>,
    >,
    len: usize,
}

impl<'context, 'arena, 'expr: 'arena> Iterator for Arguments<'context, 'arena, 'expr> {
    type Item = Result<num_complex::Complex64, CalcError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'context, 'arena, 'expr: 'arena> Arguments<'context, 'arena, 'expr> {
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
            iter: slice
                .iter()
                .map(first_closure)
                .zip(std::iter::repeat(context))
                .map(second_closure),
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

/// Describe an
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
        args: Arguments<'_, '_, '_>,
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
                        ctx.funcs.insert(
                            std::borrow::Cow::Borrowed(Self::NAME),
                            std::sync::Arc::new(Self) as std::sync::Arc<dyn Func + Send + Sync>,
                    );
                }
            }

            impl Func for $sname {
                fn call(
                    &self,
                    _: &Context,
                    mut args: Arguments<'_, '_, '_>,
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

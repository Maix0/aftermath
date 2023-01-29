#[derive(Clone)]
pub struct DiffContext {
    ctx: crate::Context,
    diff_funcs: std::collections::HashMap<
        std::borrow::Cow<'static, str>,
        std::sync::Arc<dyn DifferentiableFunc + Send + Sync>,
    >,
}

impl std::ops::Deref for DiffContext {
    type Target = crate::Context;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl std::ops::DerefMut for DiffContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ctx
    }
}

impl DiffContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ctx: crate::Context::new(),
            diff_funcs: std::collections::HashMap::new(),
        }
    }

    /// Insert a differentiable function into the Context
    ///
    /// This will also insert it as an regular function
    pub fn insert_diff_func(
        &mut self,
        name: std::borrow::Cow<'static, str>,
        func: std::sync::Arc<impl DifferentiableFunc + Send + Sync + 'static>,
    ) {
        self.ctx.insert_func(name.clone(), func.clone());
        self.diff_funcs.insert(name, func);
    }

    /// Differentiate an AST
    ///
    /// This will clone the AST and do the modification there.
    ///
    /// # Errors
    ///
    /// This will return an error in multiples occasion:
    ///
    /// - A function couldn't be differentiated
    /// - A function isn't in the context
    ///
    pub fn differentiate<'arena>(
        &self,
        arena: &'arena bumpalo::Bump,
        expr: &crate::Expr<'_>,
        respect_to: &str,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
        self.differentiate_inner(arena, expr.clone_in(arena), respect_to)
    }

    #[allow(clippy::too_many_lines)]
    fn differentiate_inner<'arena>(
        &self,
        arena: &'arena bumpalo::Bump,
        expr: &'arena mut crate::Expr<'arena>,
        respect_to: &str,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
        use crate::Expr::{
            Binding, ComplexNumber, FunctionCall, ImaginaryNumber, Operator, RealNumber,
        };
        use crate::Operator as Op;
        let expr_owned = std::mem::replace(expr, RealNumber { val: 0.0 });
        let res: crate::Expr = match expr_owned {
            Operator {
                op: op @ (Op::Plus | Op::Minus),
                rhs,
                lhs,
            } => Operator {
                op,
                lhs: self.differentiate_inner(arena, lhs, respect_to)?,
                rhs: self.differentiate_inner(arena, rhs, respect_to)?,
            },
            Operator {
                op: Op::Multiply,
                rhs,
                lhs,
            } => {
                // f(x) = u*v
                // f'(x) = u*v' + v * u'
                let u = lhs;
                let v = rhs;

                let u_prime = self.differentiate_inner(arena, u.clone_in(arena), respect_to)?;
                let v_prime = self.differentiate_inner(arena, v.clone_in(arena), respect_to)?;
                let left_multiplication = arena.alloc(Operator {
                    op: Op::Multiply,
                    rhs: u,
                    lhs: v_prime,
                });
                let right_multiplication = arena.alloc(Operator {
                    op: Op::Multiply,
                    rhs: v,
                    lhs: u_prime,
                });

                Operator {
                    op: Op::Plus,
                    rhs: left_multiplication,
                    lhs: right_multiplication,
                }
            }
            Operator {
                op: op @ (Op::UnaryMinus | Op::UnaryPlus),
                rhs,
                lhs,
            } => Operator {
                op,
                lhs,
                rhs: self.differentiate_inner(arena, rhs, respect_to)?,
            },
            Operator {
                op: Op::Modulo,
                lhs: input,
                rhs: mod_,
            } => {
                let lhs_diff = self.differentiate_inner(arena, input, respect_to)?;

                Operator {
                    op: Op::Modulo,
                    rhs: mod_,
                    lhs: lhs_diff,
                }
            }
            Operator {
                op: Op::Pow,
                lhs: base,
                rhs: power,
            } => {
                let alt_rep = arena.alloc(FunctionCall {
                    ident: arena.alloc_str("exp"),
                    args: bumpalo::vec![in arena; arena.alloc(Operator { op: Op::Multiply, rhs: power, lhs: arena.alloc(FunctionCall { ident: arena.alloc_str("ln"), args: bumpalo::vec![in arena; base] }) })],
                });

                let res = self.differentiate_inner(arena, alt_rep, respect_to)?;

                std::mem::replace(res, RealNumber { val: 0.0 })
            }

            Operator {
                op: Op::Divide,
                rhs,
                lhs,
            } => {
                // f(x) = u / v
                // f'(x) = (u*v' - v*u') / v²
                let u = lhs;
                let v = rhs;
                let v_clone = v.clone_in(arena);
                let u_prime = self.differentiate_inner(arena, u.clone_in(arena), respect_to)?;
                let v_prime = self.differentiate_inner(arena, v.clone_in(arena), respect_to)?;
                let left_multiplication = arena.alloc(Operator {
                    op: Op::Multiply,
                    rhs: u,
                    lhs: v_prime,
                });
                let right_multiplication = arena.alloc(Operator {
                    op: Op::Multiply,
                    rhs: v,
                    lhs: u_prime,
                });
                let top = arena.alloc(Operator {
                    op: Op::Minus,
                    rhs: left_multiplication,
                    lhs: right_multiplication,
                });

                let bottom = arena.alloc(Operator {
                    op: Op::Pow,
                    rhs: arena.alloc(RealNumber { val: 2.0 }),
                    lhs: v_clone,
                });

                Operator {
                    op: Op::Divide,
                    lhs: top,
                    rhs: bottom,
                }
            }
            Binding { name } if name == respect_to => RealNumber { val: 1.0 },
            Binding { .. } | RealNumber { .. } | ImaginaryNumber { .. } | ComplexNumber { .. } => {
                RealNumber { val: 0.0 }
            }
            FunctionCall { ident, args } => std::mem::replace(
                self.diff_funcs
                    .get(ident)
                    .ok_or(DiffError::CalcError(crate::CalcError::MissingFunction))?
                    .get_diffed_func(
                        self,
                        arena,
                        ident,
                        args,
                        respect_to,
                        DiffContext::differentiate_inner,
                    )?,
                RealNumber { val: 0.0 },
            ),
        };
        *expr = res;
        Ok(expr)
    }
}

impl Default for DiffContext {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug)]
pub enum DiffError {
    /// A function has had an error covered by [`CalcError`](crate::CalcError)
    ///
    /// Mostly used for invalid arguments count
    CalcError(crate::CalcError),
    /// A user-defined error
    ///
    /// Is used when implementing custom function with differentiation logic
    Boxed(Box<dyn std::error::Error + Send + Sync>),
    /// A function was encountered and it wasn't registered as differentiable
    DerivativeNotFound,
    /// Unknown error encountered
    UnknownError,
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CalcError(c) => c.fmt(f),
            Self::Boxed(b) => b.fmt(f),
            Self::UnknownError => {
                write!(f, "An unknown error has occured")
            }
            Self::DerivativeNotFound => {
                write!(f, "No derivative was found")
            }
        }
    }
}

impl std::error::Error for DiffError {}

pub type Differentiate<'arena> = fn(
    context: &DiffContext,
    arena: &'arena bumpalo::Bump,
    expr: &'arena mut crate::Expr<'arena>,
    respect_to: &str,
) -> Result<&'arena mut crate::Expr<'arena>, DiffError>;

#[allow(clippy::mut_from_ref, clippy::missing_errors_doc)]
/// A trait that mark a function as differentiable

pub trait DifferentiableFunc: crate::Func + Send + Sync {
    /// Called when asked to differentiate an function
    ///
    /// The implementor needs to correctly handle the arguments and differentiate them if necessary
    /// For function you will probably return something like this:
    /// ```text
    /// f(x) = your function
    /// h(x) is the current derivated function
    /// h'(f(g(x))) = f'(g(x)) + g'(x)
    /// ```
    ///
    /// the input to your function will be `f(g(x))` and your ouput will be `f'(g(x)) + g'(x)` for the above example
    ///
    /// To differentiate the arguments, use the [`Differentiate`](Differentiate) function passed to
    /// you
    fn get_diffed_func<'arena>(
        &self,
        ctx: &DiffContext,
        arena: &'arena bumpalo::Bump,
        func_name: &'arena str,
        args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
        respect_to: &str,
        diff_args: Differentiate<'arena>,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError>;
}

mod funcs {
    use super::DiffError;
    use super::DifferentiableFunc as DiffFunc;
    use crate::funcs::{
        Acos, Acosh, Asin, Asinh, Atan, Atanh, Cbrt, Cos, Cosh, Exp, Ln, Log, Norm, Sin, Sinh,
        Sqrt, Tan, Tanh,
    };
    use crate::Expr::{FunctionCall, Operator, RealNumber};
    use crate::Operator as Op;

    impl DiffFunc for Sin {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, super::DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;
            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str("cos"),
                args,
            });
            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: g_prime,
                lhs: func,
            });

            Ok(mult)
        }
    }

    impl DiffFunc for Cos {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, super::DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;
            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sin"),
                args,
            });
            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: g_prime,
                lhs: func,
            });

            let neg = arena.alloc(Operator {
                op: Op::UnaryMinus,
                lhs: arena.alloc(RealNumber { val: 0.0 }),
                rhs: mult,
            });

            Ok(neg)
        }
    }

    impl DiffFunc for Acos {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, super::DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;
            let _1_minus_g = arena.alloc(Operator {
                op: Op::Minus,
                rhs: arena.alloc(Operator {
                    op: Op::Pow,
                    rhs: arena.alloc(RealNumber { val: 2.0 }),
                    lhs: (!args.is_empty())
                        .then(|| args.swap_remove(0))
                        .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?,
                }),
                lhs: arena.alloc(RealNumber { val: 1.0 }),
            });

            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sqrt"),
                args: bumpalo::vec![in arena;

                ],
            });
            let div = arena.alloc(Operator {
                op: Op::Divide,
                lhs: g_prime,
                rhs: func,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Acosh {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.swap_remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_clone = g.clone_in(arena);
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let g_plus_1 = arena.alloc(Operator {
                op: Op::Plus,
                rhs: arena.alloc(RealNumber { val: 1.0 }),
                lhs: g,
            });
            let g_minus_1 = arena.alloc(Operator {
                op: Op::Minus,
                rhs: arena.alloc(RealNumber { val: 1.0 }),
                lhs: g_clone,
            });

            let sqrt_g_plus_1 = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sqrt"),
                args: bumpalo::vec![in arena; g_plus_1],
            });
            let sqrt_g_minus_1 = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sqrt"),
                args: bumpalo::vec![in arena; g_minus_1],
            });

            let mult_sqrts = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: sqrt_g_minus_1,
                lhs: sqrt_g_plus_1,
            });
            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: mult_sqrts,
                lhs: g_prime,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Asin {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let g_squared = arena.alloc(Operator {
                op: Op::Pow,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
                lhs: g,
            });
            let one_minus_g_squared = arena.alloc(Operator {
                op: Op::Minus,
                rhs: g_squared,
                lhs: arena.alloc(RealNumber { val: 1.0 }),
            });
            let sqrt = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sqrt"),
                args: bumpalo::vec![in arena; one_minus_g_squared],
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: sqrt,
                lhs: g_prime,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Asinh {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let g_squared = arena.alloc(Operator {
                op: Op::Pow,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
                lhs: g,
            });
            let one_plus_g_squared = arena.alloc(Operator {
                op: Op::Plus,
                rhs: g_squared,
                lhs: arena.alloc(RealNumber { val: 1.0 }),
            });
            let sqrt = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sqrt"),
                args: bumpalo::vec![in arena; one_plus_g_squared],
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: sqrt,
                lhs: g_prime,
            });

            Ok(div)
        }
    }
    impl DiffFunc for Atan {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let g_squared = arena.alloc(Operator {
                op: Op::Pow,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
                lhs: g,
            });
            let one_plus_g_squared = arena.alloc(Operator {
                op: Op::Plus,
                rhs: g_squared,
                lhs: arena.alloc(RealNumber { val: 1.0 }),
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: one_plus_g_squared,
                lhs: g_prime,
            });

            Ok(div)
        }
    }
    impl DiffFunc for Atanh {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let g_squared = arena.alloc(Operator {
                op: Op::Pow,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
                lhs: g,
            });
            let g_squared_minus_1 = arena.alloc(Operator {
                op: Op::Plus,
                lhs: g_squared,
                rhs: arena.alloc(RealNumber { val: 1.0 }),
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: g_squared_minus_1,
                lhs: g_prime,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Cbrt {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let g_squared = arena.alloc(Operator {
                op: Op::Pow,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
                lhs: g,
            });
            let cbrt_g_squared = arena.alloc(FunctionCall {
                ident: arena.alloc_str("cbrt"),
                args: bumpalo::vec![in arena; g_squared],
            });

            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: arena.alloc(RealNumber { val: 3.0 }),
                lhs: cbrt_g_squared,
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: mult,
                lhs: g_prime,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Sinh {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, super::DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;
            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str("cosh"),
                args,
            });
            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: g_prime,
                lhs: func,
            });

            Ok(mult)
        }
    }

    impl DiffFunc for Cosh {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, super::DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;
            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sinh"),
                args,
            });
            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: g_prime,
                lhs: func,
            });

            let neg = arena.alloc(Operator {
                op: Op::UnaryMinus,
                lhs: arena.alloc(RealNumber { val: 0.0 }),
                rhs: mult,
            });

            Ok(neg)
        }
    }

    impl DiffFunc for Exp {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, super::DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;
            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str(func_name),
                args,
            });
            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: g_prime,
                lhs: func,
            });

            Ok(mult)
        }
    }

    impl DiffFunc for Ln {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let div = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: g_prime,
                lhs: g,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Log {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let arg = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;

            let base = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let log_g = arena.alloc(FunctionCall {
                ident: arena.alloc_str("ln"),
                args: bumpalo::vec![in arena; arg],
            });
            let log_base = arena.alloc(FunctionCall {
                ident: arena.alloc_str("ln"),
                args: bumpalo::vec![in arena; base],
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: log_base,
                lhs: log_g,
            });

            diff_args(ctx, arena, div, respect_to)
        }
    }
    impl DiffFunc for Sqrt {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g = (!args.is_empty())
                .then(|| args.remove(0))
                .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?;
            let g_prime = diff_args(ctx, arena, g.clone_in(arena), respect_to)?;

            let sqrt_g = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sqrt"),
                args: bumpalo::vec![in arena; g],
            });

            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
                lhs: sqrt_g,
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                rhs: mult,
                lhs: g_prime,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Tan {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;

            let cos_g = arena.alloc(FunctionCall {
                ident: arena.alloc_str("cos"),
                args,
            });

            let cos_g_squared = arena.alloc(Operator {
                op: Op::Pow,
                lhs: cos_g,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                lhs: g_prime,
                rhs: cos_g_squared,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Tanh {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;

            let cosh_g = arena.alloc(FunctionCall {
                ident: arena.alloc_str("cosh"),
                args,
            });

            let cosh_g_squared = arena.alloc(Operator {
                op: Op::Pow,
                lhs: cosh_g,
                rhs: arena.alloc(RealNumber { val: 2.0 }),
            });

            let div = arena.alloc(Operator {
                op: Op::Divide,
                lhs: g_prime,
                rhs: cosh_g_squared,
            });

            Ok(div)
        }
    }

    impl DiffFunc for Norm {
        fn get_diffed_func<'arena>(
            &self,
            ctx: &super::DiffContext,
            arena: &'arena bumpalo::Bump,
            _func_name: &'arena str,
            mut args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
            respect_to: &str,
            diff_args: super::Differentiate<'arena>,
        ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
            let g_prime = diff_args(
                ctx,
                arena,
                args.get_mut(0)
                    .ok_or(DiffError::CalcError(crate::CalcError::InvalidArgumentCount))?
                    .clone_in(arena),
                respect_to,
            )?;

            let func = arena.alloc(FunctionCall {
                ident: arena.alloc_str("sign"),
                args,
            });

            let mult = arena.alloc(Operator {
                op: Op::Multiply,
                rhs: func,
                lhs: g_prime,
            });

            Ok(mult)
        }
    }
}

#[allow(clippy::wildcard_imports)]
pub fn add_all_diff_functions(ctx: &mut DiffContext) {
    use crate::funcs::*;

    macro_rules! add {
        ($name:ident) => {
            ctx.insert_diff_func($name::NAME.into(), $name.into());
        };

        ($($name:ident),+ $(,)?) => {
            $(add!($name));+
        };
    }

    add! {Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Exp, Ln, Log, Sqrt, Cbrt, Norm,};
    Sign::add_to_context(&mut ctx.ctx);
}

#[cfg(test)]
mod tests {
    macro_rules! test_diff {
        ($name:ident: $input:literal $(=)?) => {
            #[test]
            fn $name() {
                let mut ctx = super::DiffContext::new();
                super::add_all_diff_functions(&mut ctx);

                let arena = bumpalo::Bump::with_capacity(1024);
                let expr = crate::Expr::parse(&arena, $input, &ctx.get_reserved_names()).unwrap();
                let diff = ctx.differentiate(&arena, expr, "x").unwrap();

                dbg!(diff.to_string());
                panic!();
            }
        };

        ($name:ident: $input:literal = $output:literal) => {
            #[test]
            #[allow(unused_mut)]
            fn $name() {
                let mut ctx = super::DiffContext::new();
                super::add_all_diff_functions(&mut ctx);

                let arena = bumpalo::Bump::with_capacity(1024);
                let expr = crate::Expr::parse(&arena, $input, &ctx.get_reserved_names()).unwrap();
                let diff = ctx.differentiate(&arena, expr, "x").unwrap();

                assert_eq!(diff.to_string(), $output);
            }
        };
    }

    test_diff! {number: "1 + 1" = "0 + 0"}

    test_diff! {simple_sin: "sin(5x)" = "cos(5 * x) * (0 * x + 1 * 5)"}

    test_diff! {complex_no_pow: "((2x + 5i) * x) / (7x - 1)" =
    "(((0 * x + 1 * 2 + 0 * i + 0 * 5) * x + 1 * (2 * x + 5 * i)) * (7 * x - 1) - (0 * x + 1 * 7 - 0) * (2 * x + 5 * i) * x) / (7 * x - 1) ^ 2"}
}

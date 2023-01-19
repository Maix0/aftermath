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

impl DiffContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ctx: crate::Context::new(),
            diff_funcs: std::collections::HashMap::new(),
        }
    }

    pub fn insert_func(
        &mut self,
        name: std::borrow::Cow<'static, str>,
        func: std::sync::Arc<impl DifferentiableFunc + Send + Sync + 'static>,
    ) {
        self.ctx.insert_func(name.clone(), func.clone());
        self.diff_funcs.insert(name, func);
    }

    pub fn differentiate<'arena>(
        &self,
        arena: &'arena bumpalo::Bump,
        expr: &crate::Expr<'_>,
        respect_to: &str,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
        Ok(self.differentiate_inner(arena, expr.clone_in(arena), respect_to)?)
    }
    fn differentiate_inner<'arena>(
        &self,
        arena: &'arena bumpalo::Bump,
        expr: &'arena mut crate::Expr<'arena>,
        respect_to: &str,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
        use crate::Expr::*;
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
                let addition = Operator {
                    op: Op::Plus,
                    rhs: left_multiplication,
                    lhs: right_multiplication,
                };
                addition
            }
            Operator {
                op: op @ (Op::UnaryMinus | Op::UnaryPlus),
                rhs,
                ..
            } => Operator {
                op,
                lhs: arena.alloc(RealNumber { val: 0.0 }),
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
                // (u*v' - v*u') / v²
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

                let res = Operator {
                    op: Op::Divide,
                    lhs: top,
                    rhs: bottom,
                };

                res
            }
            RealNumber { .. } => RealNumber { val: 0.0 },
            ImaginaryNumber { .. } => RealNumber { val: 0.0 },
            ComplexNumber { .. } => RealNumber { val: 0.0 },
            Binding { name } if name == respect_to => RealNumber { val: 1.0 },
            Binding { .. } => RealNumber { val: 0.0 },
            FunctionCall { ident, args } => std::mem::replace(
                self.diff_funcs
                    .get(ident)
                    .ok_or(DiffError::CalcError(crate::CalcError::MissingFunction))?
                    .get_diffed_func(
                        &self,
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
    CalcError(crate::CalcError),
    Boxed(Box<dyn std::error::Error + Send + Sync>),
    UnableToDifferentiate,
    DerivativeNotFound,
    Panicked,
    UnknownError,
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CalcError(c) => c.fmt(f),
            Self::Boxed(b) => b.fmt(f),
            Self::UnableToDifferentiate => {
                write!(f, "Unable to differentiate the given expression")
            }
            Self::Panicked => {
                write!(
                    f,
                    "An function has panicked while differentiating a function"
                )
            }
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

#[allow(clippy::mut_from_ref)]
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
    /// To differentiate the arguments, use the [`Differentiate`](Differentiate)
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
    use crate::funcs::*;
    use crate::Expr::*;
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
                    .ok_or(DiffError::UnableToDifferentiate)?
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
}

pub fn add_all_diff(ctx: &mut DiffContext) {
    ctx.insert_func("cos".into(), crate::funcs::Cos.into());
    ctx.insert_func("sin".into(), crate::funcs::Sin.into());
}

#[cfg(test)]
mod tests {
    macro_rules! test_diff {
        ($name:ident: $input:literal $(=)?) => {
            #[test]
            fn $name() {
                let mut ctx = super::DiffContext::new();
                super::add_all_diff(&mut ctx);

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
                super::add_all_diff(&mut ctx);

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
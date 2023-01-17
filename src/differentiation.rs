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

    pub fn differentiate_inner<'arena>(
        &self,
        arena: &'arena bumpalo::Bump,
        expr: &'arena mut crate::Expr<'arena>,
        respect_to: &str,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError> {
        use crate::Expr::*;
        use crate::Operator as Op;
        let res: Option<&'arena mut crate::Expr> = match expr {
            Operator {
                op: Op::Plus | Op::Minus,
                rhs,
                lhs,
            } => {
                self.differentiate_inner(arena, lhs, respect_to)?;
                self.differentiate_inner(arena, rhs, respect_to)?;
                None
            }
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
                Some(arena.alloc(addition))
            }
            Operator {
                op: Op::UnaryMinus | Op::UnaryPlus,
                rhs,
                ..
            } => {
                *rhs = self.differentiate_inner(arena, rhs, respect_to)?;
                None
            }
            Operator {
                op: Op::Divide,
                rhs,
                lhs,
            } => {
                // u*v' - v*u'
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

                Some(arena.alloc(res))
            }
            RealNumber { val } => todo!(),
            ImaginaryNumber { val } => todo!(),
            ComplexNumber { val } => todo!(),
            Binding { name } if *name == respect_to => {}
            Binding { name } => {}
            FunctionCall { ident, args } => {
                let FunctionCall { ident, args } = std::mem::replace(expr, RealNumber { val: 0.0 }) else {return Err(DiffError::UnknownError);};
                Some(
                    self.diff_funcs
                        .get(ident)
                        .ok_or(DiffError::CalcError(crate::CalcError::MissingFunction))?
                        .get_diffed_func(
                            arena,
                            ident,
                            args,
                            respect_to,
                            DiffContext::differentiate_inner,
                        )?,
                )
            }
        };
        if let Some(r) = res {
            std::mem::swap(expr, r);
        }
        Ok(expr)
    }
}

impl Default for DiffContext {
    fn default() -> Self {
        Self::new()
    }
}
#[derive()]
pub enum DiffError {
    CalcError(crate::CalcError),
    Boxed(Box<dyn std::error::Error + Send + Sync>),
    UnableToDifferentiate,
    DerivativeNotFound,
    Panicked,
    UnknownError,
}

type Differentiate<'arena> = fn(
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
        arena: &'arena bumpalo::Bump,
        func_name: &'arena str,
        args: bumpalo::collections::Vec<'arena, &'arena mut crate::Expr<'arena>>,
        respect_to: &str,
        diff_args: Differentiate<'arena>,
    ) -> Result<&'arena mut crate::Expr<'arena>, DiffError>;
}

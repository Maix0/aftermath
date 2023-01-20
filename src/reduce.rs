//! Everything related to reducing an expression
//!
//! This is used after differentiating an expression since it produces lots of garbage that
//! *does not change the result* but may change the cost of evaluating the expression
//!
//! In the context of this crate, reducing an expression is pre-computing every constant operation
//! that can be. This will remove stuff like identity operation (i.e. multiply by 1 or adding 0)
//! It won't compute functions and use bindings, but will remove them if not used in the end

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::Context {}
}

fn expr_to_complex(e: &crate::Expr<'_>) -> Option<num_complex::Complex64> {
    match e {
        crate::Expr::RealNumber { val } => Some(num_complex::Complex { re: *val, im: 0.0 }),
        crate::Expr::ImaginaryNumber { val } => Some(num_complex::Complex { re: 0.0, im: *val }),
        crate::Expr::ComplexNumber { val } => Some(*val),
        _ => None,
    }
}

#[allow(clippy::missing_errors_doc)]
pub trait Reduce: sealed::Sealed {
    fn reduce(&self, expr: &'_ mut crate::Expr<'_>) -> Result<(), crate::CalcError>;
}

impl Reduce for crate::Context {
    #[allow(clippy::too_many_lines)]
    fn reduce(&self, expr: &'_ mut crate::Expr<'_>) -> Result<(), crate::CalcError> {
        use crate::Expr::{ComplexNumber, FunctionCall, ImaginaryNumber, Operator, RealNumber};
        use crate::Operator as Op;
        match expr {
            Operator { lhs, rhs, .. } => {
                self.reduce(lhs)?;
                self.reduce(rhs)?;
            }
            FunctionCall { args, .. } => {
                for e in args.iter_mut() {
                    self.reduce(e)?;
                }
            }
            _ => {}
        };

        let to_swap = match expr {
            RealNumber { val } => Some(ComplexNumber {
                val: num_complex::Complex { re: *val, im: 0.0 },
            }),
            ImaginaryNumber { val } => Some(ComplexNumber {
                val: num_complex::Complex { im: *val, re: 0.0 },
            }),

            Operator {
                op: Op::Multiply,
                lhs: zero,
                ..
            }
            | Operator {
                op: Op::Multiply,
                rhs: zero,
                ..
            } if self.is_complex_zero(expr_to_complex(zero).unwrap_or(num_complex::Complex {
                re: f64::MAX,
                im: f64::MAX,
            })) =>
            {
                Some(ComplexNumber {
                    val: num_complex::Complex { re: 0.0, im: 0.0 },
                })
            }
            Operator {
                op: Op::Multiply,
                lhs: val,
                rhs: one,
            }
            | Operator {
                op: Op::Multiply,
                lhs: one,
                rhs: val,
            } if self.is_complex_near(
                expr_to_complex(one).unwrap_or(num_complex::Complex {
                    re: f64::MAX,
                    im: f64::MAX,
                }),
                num_complex::Complex { re: 1.0, im: 0.0 },
            ) =>
            {
                Some(std::mem::replace(*val, RealNumber { val: 0.0 }))
            }
            Operator {
                op: Op::Plus,
                lhs: val,
                rhs: zero,
            }
            | Operator {
                op: Op::Plus,
                lhs: zero,
                rhs: val,
            } if self.is_complex_zero(expr_to_complex(zero).unwrap_or(num_complex::Complex {
                re: f64::MAX,
                im: f64::MAX,
            })) =>
            {
                Some(std::mem::replace(*val, RealNumber { val: 0.0 }))
            }
            Operator {
                op: Op::Minus,
                lhs: val,
                rhs: zero,
            } if self.is_complex_zero(expr_to_complex(zero).unwrap_or(num_complex::Complex {
                re: f64::MAX,
                im: f64::MAX,
            })) =>
            {
                Some(std::mem::replace(*val, RealNumber { val: 0.0 }))
            }

            Operator {
                op: Op::Divide,
                rhs: one,
                lhs,
            } if self.is_complex_near(
                expr_to_complex(one).unwrap_or(num_complex::Complex {
                    re: f64::MAX,
                    im: f64::MAX,
                }),
                num_complex::Complex { re: 1.0, im: 0.0 },
            ) =>
            {
                Some(std::mem::replace(*lhs, RealNumber { val: 0.0 }))
            }

            Operator {
                op: Op::Divide,
                lhs: zero,
                ..
            } if self.is_complex_zero(expr_to_complex(zero).unwrap_or(num_complex::Complex {
                re: f64::MAX,
                im: f64::MAX,
            })) =>
            {
                Some(ComplexNumber {
                    val: num_complex::Complex { re: 0.0, im: 0.0 },
                })
            }

            Operator {
                op,
                lhs: ComplexNumber { val: lhs },
                rhs: ComplexNumber { val: rhs },
            } => Some(ComplexNumber {
                val: match op {
                    Op::Plus => *lhs + *rhs,
                    Op::Minus => *lhs - *rhs,
                    Op::Multiply => *lhs * *rhs,
                    Op::Divide => *lhs / *rhs,
                    Op::Modulo => *lhs % *rhs,
                    Op::Pow => lhs.powc(*rhs),
                    Op::UnaryMinus => -*rhs,
                    Op::UnaryPlus => *rhs,
                },
            }),
            _ => None,
        };
        *expr = match to_swap {
            Some(e) => e,
            None => return Ok(()),
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "differentiation")]
    mod test_with_diff {
        use super::super::Reduce;
        use crate::differentiation::DiffContext;
        use crate::Expr;

        macro_rules! make_test_diffed {
            ($name:ident: $input:literal = $output:literal) => {
                #[test]
                fn $name() {
                    let input: &str = $input;
                    let output: &str = $output;

                    let arena = bumpalo::Bump::with_capacity(1024);

                    let mut ctx = DiffContext::new();
                    crate::differentiation::add_all_diff_functions(&mut ctx);

                    let reserved_words = ctx.get_reserved_names();
                    let parsed = Expr::parse(&arena, input, &reserved_words).unwrap();
                    let diffed = ctx.differentiate(&arena, parsed, "x").unwrap();
                    ctx.reduce(diffed).unwrap();

                    assert_eq!(diffed.to_string(), output);
                }
            };
        }

        make_test_diffed! {simple: "x + 0*x" = "1+0i"}
        make_test_diffed! {sin: "sin(x)" = "cos(x)"}
        make_test_diffed! {cos2x: "cos(2x)" = "-(sin(2+0i * x) * 2+0i)"}
        make_test_diffed! {div: "sin(x)/1" = "cos(x)"}
    }
    #[cfg(not(feature = "differentiation"))]
    mod test_with_diff {
        macro_rules! compile_warning {
            ($message:literal) => {
                #[warn(dead_code)]
                const MESSAGE: &str = $message;
            };
        }

        compile_warning!("Some tests are disabled without the `differentiation` feature");
    }
    mod test_without_diff {
        use super::super::Reduce;
        use crate::differentiation::DiffContext;
        use crate::Expr;

        macro_rules! make_test {
            ($name:ident: $input:literal = $output:literal) => {
                #[test]
                fn $name() {
                    let input: &str = $input;
                    let output: &str = $output;

                    let arena = bumpalo::Bump::with_capacity(1024);

                    let mut ctx = DiffContext::new();
                    crate::differentiation::add_all_diff_functions(&mut ctx);

                    let reserved_words = ctx.get_reserved_names();
                    let parsed = Expr::parse(&arena, input, &reserved_words).unwrap();
                    ctx.reduce(parsed).unwrap();

                    assert_eq!(parsed.to_string(), output);
                }
            };
        }

        make_test! {multipy_by_zero: "a * (0 + 0)" = "0+0i"}
        make_test! {multipy_by_one:  "a * (0.5 + 0.5)" = "a"}
        make_test! {divide_by_one:   "a / (1/3 + 1/3 + 1/3)" = "a"}
        make_test! {add_by_zero:   "(a + b) + (1 - 1)" = "a + b"}
        make_test! {add_by_zero2:   "(1-1) + (a + b)" = "a + b"}
        make_test! {sub_by_zero:   "(a + b) - (0 / 1000)" = "a + b"}
        make_test! {zero_divided_by_x:   "0 / x" = "0+0i"}
    }
}

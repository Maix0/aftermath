#![warn(clippy::pedantic)]
#[allow(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    // unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]
mod expr;
mod token_stream;

pub use expr::AstBuildError;
pub use expr::Expr;
pub use expr::InvalidToken;
pub use expr::Operator;

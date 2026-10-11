//! Recompute MPFR reference bounds for the checked-in regression corpus.
//! Inputs, case order, and operation groups are preserved. Add regression
//! inputs to the corpus explicitly; random sampling belongs to the live tests.
//!
//! Run with `cargo run -p mwise-tools --features oracle --bin update-fixtures`.
//! An optional output path writes a copy instead of updating the corpus.

use mwise_tools::{oracle, serialization};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(serialization::FIXTURE_PATH));
    let mut groups = serialization::load(serialization::FIXTURE_PATH)?;
    for group in &mut groups {
        for case in &mut group.cases {
            *case = oracle::case(group.width, &group.op, case.input);
        }
    }
    serialization::save(&path, &groups)?;
    let count: usize = groups.iter().map(|group| group.cases.len()).sum();
    println!(
        "Wrote {count} cases with {}-bit MPFR references to {}",
        oracle::PRECISION,
        path.display()
    );
    Ok(())
}

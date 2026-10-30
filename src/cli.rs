use clap::{Args, CommandFactory};
use std::fmt::Display;
use std::str::FromStr;

pub const MAX_QUERIES: usize = u32::BITS as usize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    Absolute(usize),
    Relative(f64),
}

impl FromStr for Threshold {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(n) = s.parse::<usize>() {
            return if n > 0 {
                Ok(Self::Absolute(n))
            } else {
                Err("Absolute threshold must be ≥ 1".into())
            };
        }
        match s.parse::<f64>() {
            Ok(f) if f.is_finite() && f > 0. && f <= 1. => Ok(Self::Relative(f)),
            Ok(_) => Err("Relative threshold must be in (0, 1]".into()),
            Err(_) => Err("Invalid threshold format, pass an int or a float".into()),
        }
    }
}

impl Display for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute(n) => write!(f, "{n}"),
            Self::Relative(r) => write!(f, "{r}"),
        }
    }
}

impl Threshold {
    pub(crate) fn required(self, kmers: usize, reference_count: usize) -> usize {
        let required = match self {
            Self::Absolute(n) => n,
            Self::Relative(f) => (kmers as f64 * f).ceil() as usize,
        };
        // Preserve K2Rmini's cap, including its behavior for an empty reference.
        if reference_count > 0 {
            required.min(reference_count)
        } else {
            required
        }
    }
}

/// Options shared verbatim by both CLIs and by the filtering implementation.
#[derive(Args, Debug, Clone)]
pub struct Options {
    /// Output file for filtered sequences [default: stdout]
    #[arg(short)]
    pub output: Option<String>,
    /// K-mer size
    #[arg(short, default_value_t = 31)]
    pub k: usize,
    /// Minimizer size, must be ≤ k, up to 29
    #[arg(short, default_value_t = 21)]
    pub m: usize,
    /// Approximate filtering: only index and filter by minimizers (no k-mers indexed)
    #[arg(short = 'a', long)]
    pub approx: bool,
    /// Use a global 64-bit set: exact k-mer keys for k <= 32, hashes for larger k
    #[arg(long, conflicts_with = "approx")]
    pub high_precision: bool,
    /// Number of threads [default: all]
    #[arg(short = 'T', long)]
    pub threads: Option<usize>,
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("K-mer size must be ≥ 1".into());
        }
        if self.m == 0 || self.m > 29 {
            return Err("Minimizer size must be in 1..=29".into());
        }
        if self.m > self.k {
            return Err("Minimizer size must be ≤ k".into());
        }
        if self.threads == Some(0) {
            return Err("Number of threads must be ≥ 1".into());
        }
        if self.approx && self.high_precision {
            return Err("--approx cannot be combined with --high-precision".into());
        }
        Ok(())
    }

    pub(crate) fn num_threads(&self) -> usize {
        self.threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
    }
}

#[derive(Debug, Clone)]
pub struct QuerySpec {
    pub patterns: String,
    pub threshold: Threshold,
}

pub fn parse_constraints(values: &[String]) -> Result<Vec<QuerySpec>, String> {
    if values.is_empty() {
        return Err("At least one -c/--constraint is required".into());
    }
    if !values.len().is_multiple_of(2) {
        return Err("Each -c/--constraint needs both PATTERNS and THRESHOLD".into());
    }
    if values.len() / 2 > MAX_QUERIES {
        return Err(format!(
            "k2rminimulti supports at most {MAX_QUERIES} query files"
        ));
    }
    values
        .chunks_exact(2)
        .map(|pair| {
            Ok(QuerySpec {
                patterns: pair[0].clone(),
                threshold: pair[1]
                    .parse()
                    .map_err(|err| format!("Invalid threshold for {}: {err}", pair[0]))?,
            })
        })
        .collect()
}

pub fn value_error<C: CommandFactory>(message: String) -> ! {
    C::command()
        .error(clap::error::ErrorKind::ValueValidation, message)
        .exit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_constraints_and_validates_limits_and_thresholds() {
        let values = ["a", "2", "b", "0.5"].map(str::to_owned);
        let queries = parse_constraints(&values).unwrap();
        assert_eq!(queries[0].patterns, "a");
        assert_eq!(queries[0].threshold, Threshold::Absolute(2));
        assert_eq!(queries[1].threshold, Threshold::Relative(0.5));
        assert!(parse_constraints(&[]).is_err());
        assert!(parse_constraints(&values[..3]).is_err());
        let pairs = |n| {
            ["a", "1"]
                .repeat(n)
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert!(parse_constraints(&pairs(33)).is_err());
        assert_eq!(parse_constraints(&pairs(32)).unwrap().len(), 32);
        for value in ["0", "0.0", "-1", "NaN", "inf", "1.1", "bad"] {
            assert!(value.parse::<Threshold>().is_err(), "{value}");
        }
        assert_eq!("1".parse(), Ok(Threshold::Absolute(1)));
        assert_eq!("1.0".parse(), Ok(Threshold::Relative(1.0)));
    }
}

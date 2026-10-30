use clap::{ArgAction, Parser};
use k2rmini::cli::{Options, QuerySpec, Threshold, parse_constraints, value_error};
use std::io;

#[derive(Parser, Debug)]
#[command(author, version, about = "Filter reads by multiple k-mer query constraints", long_about = None)]
struct Args {
    /// FASTA/Q file to filter (possibly compressed)
    file: String,
    /// FASTA/Q file containing k-mers of interest (possibly compressed)
    #[arg(
        short,
        conflicts_with = "constraints",
        required_unless_present = "constraints"
    )]
    patterns: Option<String>,
    /// K-mer threshold for -p, either relative (float) or absolute (int)
    #[arg(short, long, default_value_t = Threshold::Relative(0.5), requires = "patterns")]
    threshold: Threshold,
    /// FASTA/Q file containing k-mers of interest and its threshold; may be repeated
    #[arg(short = 'c', long = "constraint", value_names = ["PATTERNS", "THRESHOLD"],
        num_args = 2, action = ArgAction::Append, required_unless_present = "patterns")]
    constraints: Vec<String>,
    #[command(flatten)]
    options: Options,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    args.options
        .validate()
        .unwrap_or_else(|err| value_error::<Args>(err));
    let queries = if let Some(patterns) = args.patterns {
        vec![QuerySpec {
            patterns,
            threshold: args.threshold,
        }]
    } else {
        parse_constraints(&args.constraints).unwrap_or_else(|err| value_error::<Args>(err))
    };
    k2rmini::run(&args.file, &queries, &args.options)
}

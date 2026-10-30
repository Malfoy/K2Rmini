use clap::Parser;
use k2rmini::cli::{Options, QuerySpec, Threshold, value_error};
use std::io;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// FASTA/Q file to filter (possibly compressed)
    file: String,
    /// FASTA/Q file containing k-mers of interest (possibly compressed)
    #[arg(short)]
    patterns: String,
    /// K-mer threshold, either relative (float) or absolute (int)
    #[arg(short, long, default_value_t = Threshold::Relative(0.5))]
    threshold: Threshold,
    #[command(flatten)]
    options: Options,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    args.options
        .validate()
        .unwrap_or_else(|err| value_error::<Args>(err));
    k2rmini::run(
        &args.file,
        &[QuerySpec {
            patterns: args.patterns,
            threshold: args.threshold,
        }],
        &args.options,
    )
}

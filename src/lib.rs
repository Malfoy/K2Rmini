//! Shared CLI options, reference indexing, and filtering for K2Rmini and k2rminimulti.
pub mod ambiguity;
pub mod cli;
mod filter;
pub mod high_precision;
mod index;
mod sharded_hashset;
mod sparse_bitset;

use cli::{MAX_QUERIES, Options, QuerySpec, Threshold};
use index::{FilterIndex, MultiIndex, index_reference};
use std::io;
use std::time::Instant;

fn mem_usage_gb() -> f64 {
    let mut usage = std::mem::MaybeUninit::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0.;
    }
    let maxrss = unsafe { usage.assume_init() }.ru_maxrss as f64;
    if cfg!(target_os = "macos") {
        maxrss / 1_000_000_000.
    } else {
        maxrss / 1_000_000.
    }
}

pub fn run(file: &str, queries: &[QuerySpec], options: &Options) -> io::Result<()> {
    options
        .validate()
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
    if queries.is_empty() || queries.len() > MAX_QUERIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Expected 1..=32 query constraints",
        ));
    }
    if queries.len() == 1 {
        eprintln!(
            "Running with k={}, m={} and {} threshold of {}",
            options.k,
            options.m,
            match queries[0].threshold {
                Threshold::Absolute(_) => "an absolute",
                Threshold::Relative(_) => "a relative",
            },
            queries[0].threshold
        );
    } else {
        eprintln!(
            "Running with k={}, m={} and {} query constraints",
            options.k,
            options.m,
            queries.len()
        );
    }
    if options.approx {
        eprintln!("Indexing minimizers of interest (approximate mode)...");
    } else if options.high_precision && options.k <= 32 {
        eprintln!("Indexing exact packed k-mers and minimizers (high precision mode)...");
    } else if options.high_precision {
        eprintln!("Indexing 64-bit k-mer hashes and minimizers (high precision mode)...");
    } else {
        eprintln!("Indexing k-mers and minimizers of interest...");
    }

    let start = Instant::now();
    if queries.len() == 1 {
        let index = index_reference(&queries[0].patterns, options)?;
        finish(file, queries, options, index, start)
    } else {
        let mut index = MultiIndex::new(options);
        for query in queries {
            index.add(index_reference(&query.patterns, options)?);
        }
        finish(file, queries, options, index, start)
    }
}

fn finish<I: FilterIndex>(
    file: &str,
    queries: &[QuerySpec],
    options: &Options,
    index: I,
    start: Instant,
) -> io::Result<()> {
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    if options.approx {
        eprintln!("Indexed {} minimizers.", index.minimizer_len());
    } else {
        eprintln!(
            "Indexed {} k-mer keys and {} minimizers.",
            index.kmer_len(),
            index.minimizer_len()
        );
    }
    let thresholds: Vec<_> = queries.iter().map(|query| query.threshold).collect();
    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    filter::process_query_streaming(file, options, &thresholds, &index)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    Ok(())
}

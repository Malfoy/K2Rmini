use clap::Parser as ClapParser;
use crossbeam_channel::bounded;
use helicase::input::*;
use helicase::*;
use rustc_hash::FxHashSet;
use simd_minimizers::minimizers;
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};

use core::fmt::Display;
use core::str::FromStr;
use std::fs::File;
use std::io::{self, BufWriter, Write, stdout};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

type MinIndex = FxHashSet<u64>;
type KmerIndex = FxHashSet<u32>;

const CONFIG_INDEX: Config = ParserOptions::default()
    .ignore_headers()
    .dna_packed()
    .keep_non_actg()
    .config();
const CONFIG_FILTER: Config = ParserOptions::default()
    .dna_string()
    .and_dna_packed()
    .keep_non_actg()
    .config();

#[derive(Debug, Clone, Copy)]
enum Threshold {
    Absolute(usize),
    Relative(f64),
}

impl FromStr for Threshold {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(val) = s.parse::<usize>() {
            if val == 0 {
                Err("Absolute threshold must be ≥ 1".to_string())
            } else {
                Ok(Self::Absolute(val))
            }
        } else if let Ok(val) = s.parse::<f64>() {
            if val.is_nan() || val.is_sign_negative() || val == 0. || val > 1. {
                Err("Relative threshold must in (0, 1]".to_string())
            } else {
                Ok(Self::Relative(val))
            }
        } else {
            Err("Invalid threshold format, pass an int or a float".to_string())
        }
    }
}

impl Display for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute(x) => write!(f, "{x}"),
            Self::Relative(x) => write!(f, "{x}"),
        }
    }
}

#[derive(ClapParser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// FASTA/Q file to filter (possibly compressed)
    #[arg()]
    file: String,
    /// FASTA/Q file containing k-mers of interest (possibly compressed)
    #[arg(short)]
    patterns: String,
    /// K-mer threshold, either relative (float) or absolute (int)
    #[arg(short, long, default_value_t = Threshold::Relative(0.5))]
    threshold: Threshold,
    /// Output file for filtered sequences [default: stdout]
    #[arg(short)]
    output: Option<String>,
    /// K-mer size
    #[arg(short, default_value_t = 31)]
    k: usize,
    /// Minimizer size, must be ≤ k, up to 29
    #[arg(short, default_value_t = 21)]
    m: usize,
    /// Number of threads [default: all]
    #[arg(short = 'T', long)]
    threads: Option<usize>,
}

// https://github.com/Daniel-Liu-c0deb0t/simple-saca/blob/main/src/main.rs#L96
fn mem_usage_gb() -> f64 {
    let rusage = unsafe {
        let mut rusage = std::mem::MaybeUninit::uninit();
        libc::getrusage(libc::RUSAGE_SELF, rusage.as_mut_ptr());
        rusage.assume_init()
    };
    let maxrss = rusage.ru_maxrss as f64;
    if cfg!(target_os = "macos") {
        maxrss / 1_000_000_000.
    } else {
        maxrss / 1_000_000.
    }
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    assert!(args.m <= args.k, "Minimizer size must be ≤ k");
    eprintln!(
        "Running with k={}, m={} and {} threshold of {}",
        args.k,
        args.m,
        match args.threshold {
            Threshold::Absolute(_) => "an absolute",
            Threshold::Relative(_) => "a relative",
        },
        args.threshold
    );

    eprintln!("Indexing k-mers and minimizers of interest...");
    let start = Instant::now();
    let (min_dict, kmer_dict) = index_reference(&args)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    let ref_min_dict = Arc::new(min_dict);
    let ref_kmer_dict = Arc::new(kmer_dict);
    eprintln!(
        "Indexed {} k-mers and {} minimizers.",
        ref_kmer_dict.len(),
        ref_min_dict.len()
    );

    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    process_query_streaming(&args, Arc::clone(&ref_kmer_dict), Arc::clone(&ref_min_dict))?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );

    Ok(())
}

fn index_reference(args: &Args) -> io::Result<(MinIndex, KmerIndex)> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let mini_builder = minimizers(minimizer_size, window_size);
    let hasher = NtHasher::<false>::new(kmer_size);
    let mut parser = FastxParser::<CONFIG_INDEX>::from_file(&args.patterns)
        .expect("Failed to parse file containing patterns");
    let mut dict_mini = MinIndex::default();
    let mut dict_kmer = KmerIndex::default();
    let mut mini_pos = Vec::new();
    let mut kmer_hashes = Vec::new();

    while let Some(event) = parser.next() {
        event?;
        let packed_seq = parser.get_packed_seq();
        mini_pos.clear();
        kmer_hashes.clear();
        let out = mini_builder.run(packed_seq, &mut mini_pos);
        hasher
            .hash_kmers_simd(packed_seq, 1)
            .collect_into(&mut kmer_hashes);
        dict_mini.extend(out.values_u64());
        dict_kmer.extend(&kmer_hashes);
    }
    dict_mini.reserve(dict_mini.len() / 2);
    Ok((dict_mini, dict_kmer))
}

fn process_query_streaming(
    args: &Args,
    ref_kmer_dict: Arc<KmerIndex>,
    ref_min_dict: Arc<MinIndex>,
) -> io::Result<()> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let threshold = args.threshold;
    let output = args.output.clone();
    let num_threads = args.threads.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let path = &args.file;
    let (result_tx, result_rx) = bounded::<(Vec<u8>, Vec<u8>)>(4 * num_threads);

    let printer_handle = thread::spawn(move || {
        if let Some(out) = output {
            let file = File::create(out).expect("Failed to open output file");
            let mut writer = BufWriter::new(file);
            for (id, seq) in result_rx.iter() {
                writer.write_all(b">")?;
                writer.write_all(&id)?;
                writer.write_all(b"\n")?;
                writer.write_all(&seq)?;
                writer.write_all(b"\n")?;
            }
        } else {
            for (id, seq) in result_rx.iter() {
                stdout().write_all(b">")?;
                stdout().write_all(&id)?;
                stdout().write_all(b"\n")?;
                stdout().write_all(&seq)?;
                stdout().write_all(b"\n")?;
            }
        }
        io::Result::Ok(())
    });

    ParallelFastxParser::<CONFIG_FILTER>::from_file(path)
        .expect("Failed to parse file to filter")
        .threads(num_threads)
        .for_each(|mut parser| {
            let mini_builder = minimizers(minimizer_size, window_size);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut mini_pos = Vec::new();
            let mut kmer_hashes = Vec::new();

            while let Some(event) = parser.next() {
                event?;
                let seq_len = parser.get_dna_len();
                if seq_len < kmer_size {
                    continue;
                }

                let kmer_threshold: usize = match threshold {
                    Threshold::Absolute(n) => n,
                    Threshold::Relative(f) => {
                        (((seq_len - kmer_size + 1) as f64) * f).ceil() as usize
                    }
                }
                .min(ref_kmer_dict.len());
                let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

                let packed_seq = parser.get_packed_seq();
                mini_pos.clear();
                let out = mini_builder.run(packed_seq, &mut mini_pos);
                let shared_min_count = out
                    .values_u64()
                    .filter(|word| ref_min_dict.contains(word))
                    .count();

                if shared_min_count < minimizer_threshold {
                    continue;
                }

                kmer_hashes.clear();
                hasher
                    .hash_kmers_simd(packed_seq, 1)
                    .collect_into(&mut kmer_hashes);
                let kmer_match_count = kmer_hashes
                    .iter()
                    .copied()
                    .filter(|hash| ref_kmer_dict.contains(hash))
                    .count();

                if kmer_match_count >= kmer_threshold {
                    let id = parser.get_header_owned();
                    let seq = parser.get_dna_string_owned();
                    let _ = result_tx.send((id, seq));
                }
            }
            Ok(())
        })
        .expect("ParallelFastxParser failed");

    drop(result_tx);
    let _ = printer_handle.join().expect("Printer thread panicked");

    Ok(())
}

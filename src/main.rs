use clap::Parser as ClapParser;
use crossbeam_channel::bounded;
use helicase::*;
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::Seq;
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};

use core::fmt::Display;
use core::str::FromStr;
use std::fs::File;
use std::io::{self, BufWriter, Write, stdout};
use std::thread;
use std::time::Instant;

mod sharded_hashset;
mod sparse_bitset;
use sharded_hashset::{ConcurrentShardedSet, ShardedHashSet};
use sparse_bitset::ConcurrentBlockSparseBitSet;

type MinIndex = ShardedHashSet;
type KmerIndex = ConcurrentBlockSparseBitSet;

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
    let (ref_min_dict, ref_kmer_dict) = index_reference(&args)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    let ref_kmer_dict_len = ref_kmer_dict.len();
    eprintln!(
        "Indexed {} k-mers and {} minimizers.",
        ref_kmer_dict_len,
        ref_min_dict.len()
    );

    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    process_query_streaming(&args, &ref_kmer_dict, ref_kmer_dict_len, &ref_min_dict)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );

    Ok(())
}

fn index_reference(args: &Args) -> io::Result<(MinIndex, KmerIndex)> {
    const BATCH_KMERS: usize = 2_000_000;

    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let window_span_bp = window_size + kmer_size - 1;
    let num_threads = args.threads.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let kmer_dict = KmerIndex::new();
    let mini_dict = ConcurrentShardedSet::new();

    ParallelFastxParser::<CONFIG_INDEX>::from_file(&args.patterns)
        .expect("Failed to parse file containing patterns")
        .threads(num_threads)
        .for_each(|mut parser| {
            let mini_hasher = NtHasher::<false>::new(minimizer_size);
            let mini_builder = minimizers(minimizer_size, window_size).hasher(&mini_hasher);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut local_mini = mini_dict.local_writer();
            let mut mini_pos = Vec::new();
            let mut super_kmers = Vec::new();
            let mut kmer_hashes = Vec::new();

            while let Some(event) = parser.next() {
                event?;
                let packed_seq = parser.get_packed_seq();
                let num_kmers = parser.get_dna_len().saturating_sub(kmer_size - 1);
                if num_kmers == 0 {
                    continue;
                }

                if num_kmers <= BATCH_KMERS {
                    mini_pos.clear();
                    super_kmers.clear();
                    let out = mini_builder
                        .super_kmers(&mut super_kmers)
                        .run(packed_seq, &mut mini_pos);
                    super_kmers.push(num_kmers as u32);

                    kmer_hashes.clear();
                    hasher
                        .hash_kmers_simd(packed_seq, 1)
                        .collect_into(&mut kmer_hashes);

                    for (min_val, bounds) in out.values_u64().zip(super_kmers.windows(2)) {
                        local_mini.insert(min_val);
                        kmer_dict.insert_batch(
                            min_val,
                            &kmer_hashes[bounds[0] as usize..bounds[1] as usize],
                        );
                    }
                } else {
                    for k_start in (0..num_kmers).step_by(BATCH_KMERS) {
                        let k_len = (num_kmers - k_start).min(BATCH_KMERS);
                        let chunk_end_bp =
                            (k_start + k_len - 1 + window_span_bp).min(packed_seq.len());
                        let sub_seq = packed_seq.slice(k_start..chunk_end_bp);
                        let sub_num_kmers = sub_seq.len().saturating_sub(kmer_size - 1);

                        mini_pos.clear();
                        super_kmers.clear();
                        kmer_hashes.clear();

                        let out = mini_builder
                            .super_kmers(&mut super_kmers)
                            .run(sub_seq, &mut mini_pos);
                        super_kmers.push(sub_num_kmers as u32);

                        hasher
                            .hash_kmers_simd(sub_seq, 1)
                            .collect_into(&mut kmer_hashes);

                        for (min_val, bounds) in out.values_u64().zip(super_kmers.windows(2)) {
                            let start_w = bounds[0] as usize;
                            if start_w >= k_len {
                                break;
                            }
                            let end_w = (bounds[1] as usize).min(k_len);
                            local_mini.insert(min_val);
                            kmer_dict.insert_batch(min_val, &kmer_hashes[start_w..end_w]);
                        }
                    }
                }
            }
            Ok(())
        })
        .expect("ParallelFastxParser failed on patterns");

    Ok((mini_dict.freeze(), kmer_dict))
}

fn process_query_streaming(
    args: &Args,
    ref_kmer_dict: &KmerIndex,
    ref_kmer_dict_len: usize,
    ref_min_dict: &MinIndex,
) -> io::Result<()> {
    const BATCH_SIZE: usize = 64 * 1024;

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
    let (result_tx, result_rx) = bounded::<Vec<u8>>(4 * num_threads);

    let printer_handle = thread::spawn(move || {
        if let Some(out) = output {
            let file = File::create(out).expect("Failed to open output file");
            let mut writer = BufWriter::new(file);
            for chunk in result_rx.iter() {
                writer.write_all(&chunk)?;
            }
        } else {
            for chunk in result_rx.iter() {
                stdout().write_all(&chunk)?;
            }
        }
        io::Result::Ok(())
    });

    ParallelFastxParser::<CONFIG_FILTER>::from_file(path)
        .expect("Failed to parse file to filter")
        .threads(num_threads)
        .for_each(|mut parser| {
            let mini_hasher = NtHasher::<false>::new(minimizer_size);
            let mini_builder = minimizers(minimizer_size, window_size).hasher(&mini_hasher);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut mini_pos = Vec::new();
            let mut super_kmers = Vec::new();
            let mut matched_sk: Vec<(u64, usize, usize)> = Vec::new();
            let mut kmer_hashes = Vec::new();
            let mut local_buf = Vec::with_capacity(BATCH_SIZE + 4096);

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
                .min(ref_kmer_dict_len);
                let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

                let packed_seq = parser.get_packed_seq();
                let num_kmers = seq_len - kmer_size + 1;
                mini_pos.clear();
                super_kmers.clear();
                let out = mini_builder
                    .super_kmers(&mut super_kmers)
                    .run(packed_seq, &mut mini_pos);
                super_kmers.push(num_kmers as u32);

                matched_sk.clear();
                let mut possible_kmers: usize = 0;
                for (min_val, bounds) in out.values_u64().zip(super_kmers.windows(2)) {
                    if ref_min_dict.contains(&min_val) {
                        let span = (bounds[1] - bounds[0]) as usize;
                        possible_kmers += span;
                        matched_sk.push((min_val, bounds[0] as usize, bounds[1] as usize));
                    }
                }

                if matched_sk.len() < minimizer_threshold || possible_kmers < kmer_threshold {
                    continue;
                }

                kmer_hashes.clear();
                hasher
                    .hash_kmers_simd(packed_seq, 1)
                    .collect_into(&mut kmer_hashes);

                let mut kmer_match_count = 0;
                let mut remaining_possible = possible_kmers;
                for &(min_val, start, end) in &matched_sk {
                    let span = end - start;
                    if let Some(page) = ref_kmer_dict.get_page(min_val) {
                        for &h in &kmer_hashes[start..end] {
                            if ConcurrentBlockSparseBitSet::page_contains(page, h) {
                                kmer_match_count += 1;
                            }
                        }
                    }
                    remaining_possible -= span;
                    if kmer_match_count >= kmer_threshold
                        || kmer_match_count + remaining_possible < kmer_threshold
                    {
                        break;
                    }
                }

                if kmer_match_count >= kmer_threshold {
                    local_buf.push(b'>');
                    local_buf.extend_from_slice(parser.get_header());
                    local_buf.push(b'\n');
                    local_buf.extend_from_slice(parser.get_dna_string());
                    local_buf.push(b'\n');

                    if local_buf.len() >= BATCH_SIZE {
                        let mut send_buf = Vec::with_capacity(BATCH_SIZE + 4096);
                        std::mem::swap(&mut local_buf, &mut send_buf);
                        let _ = result_tx.send(send_buf);
                    }
                }
            }
            if !local_buf.is_empty() {
                let _ = result_tx.send(local_buf);
            }
            Ok(())
        })
        .expect("ParallelFastxParser failed");

    drop(result_tx);
    let _ = printer_handle.join().expect("Printer thread panicked");

    Ok(())
}

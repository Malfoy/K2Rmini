use clap::Parser as ClapParser;
use crossbeam_channel::bounded;
use helicase::input::*;
use helicase::*;
use parking_lot::Mutex;
use rayon::prelude::*;
use regex::bytes::{Regex, RegexBuilder};
use rustc_hash::FxHashSet;
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::{PackedSeqVec, Seq, SeqVec};
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};

use core::fmt::Display;
use core::mem::swap;
use core::str::FromStr;
use std::array::from_fn;
use std::fs::File;
use std::io::{self, BufWriter, Write, stdout};
use std::sync::{Arc, LazyLock};
use std::thread;
use std::time::Instant;

const SHARD_BITS: usize = 9;
const SHARDS: usize = 1 << SHARD_BITS;
const SHARD_MASK: usize = SHARDS - 1;

type MinIndex = FxHashSet<u64>;
type KmerParIndex = [Mutex<FxHashSet<u32>>; SHARDS];
type KmerIndex = [FxHashSet<u32>; SHARDS];

const MSG_LEN_THRESHOLD: usize = 8000; // small enough for long reads

const CONFIG_INDEX: Config = ParserOptions::default()
    .ignore_headers()
    .dna_packed()
    .keep_non_actg()
    .config();
const CONFIG_FILTER: Config = ParserOptions::default().config();

static MATCH_N: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(r"[N]+")
        .case_insensitive(true)
        .unicode(false)
        .build()
        .unwrap()
});

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
    let threads = if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .unwrap();
        t
    } else {
        rayon::current_num_threads()
    };
    eprintln!(
        "Running with k={}, m={} and {} threshold of {} using {} thread{}",
        args.k,
        args.m,
        match args.threshold {
            Threshold::Absolute(_) => "an absolute",
            Threshold::Relative(_) => "a relative",
        },
        args.threshold,
        threads,
        if threads > 1 { "s" } else { "" }
    );

    eprintln!("Indexing k-mers and minimizers of interest...");
    let start = Instant::now();
    let (min_dict, kmer_dict) = index_reference(&args)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    let total_kmers: usize = kmer_dict.iter().map(|s| s.len()).sum();
    let ref_min_dict = Arc::new(min_dict);
    let ref_kmer_dict = Arc::new(kmer_dict);
    eprintln!(
        "Indexed {} k-mers and {} minimizers.",
        total_kmers,
        ref_min_dict.len()
    );

    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    process_query_streaming(
        &args,
        Arc::clone(&ref_kmer_dict),
        Arc::clone(&ref_min_dict),
        total_kmers,
    )?;
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
    let mut parser = FastxParser::<CONFIG_INDEX>::from_file_in_ram(&args.patterns)
        .expect("Failed to parse file containing patterns");
    let mut dict_mini = MinIndex::default();
    let dict_kmer: KmerParIndex = from_fn(|_| Mutex::new(FxHashSet::default()));
    let mut sk_pos = Vec::new();
    let mut mini_pos = Vec::new();
    let mut mini_words = Vec::new();
    let mut kmer_hashes = Vec::new();

    while let Some(_) = parser.next() {
        let packed_seq = parser.get_packed_seq();
        sk_pos.clear();
        mini_pos.clear();
        mini_words.clear();
        kmer_hashes.clear();
        mini_builder
            .super_kmers(&mut sk_pos)
            .run(packed_seq, &mut mini_pos);
        hasher
            .hash_kmers_simd(packed_seq, 1)
            .collect_into(&mut kmer_hashes);
        mini_words.extend(mini_pos.iter().copied().map(|pos| {
            packed_seq
                .slice((pos as usize)..(pos as usize + minimizer_size))
                .as_u64()
        }));
        dict_mini.extend(mini_words.iter().copied());
        // Assign each superkmer's k-mers to the shard of its minimizer
        sk_pos.push(kmer_hashes.len() as u32);
        sk_pos
            .par_windows(2)
            .zip(mini_words.par_iter().copied())
            .for_each(|(window, mini_word)| {
                let (sk_start, sk_end) = (window[0] as usize, window[1] as usize);
                dict_kmer[(mini_word as usize) & SHARD_MASK]
                    .lock()
                    .extend(&kmer_hashes[sk_start..sk_end]);
            });
    }
    dict_mini.reserve(dict_mini.len() / 2);
    let dict_kmer = dict_kmer.map(|bucket| {
        let mut dict = bucket.into_inner();
        dict.reserve(dict.len() / 2);
        dict
    });
    Ok((dict_mini, dict_kmer))
}

fn process_query_streaming(
    args: &Args,
    ref_kmer_dict: Arc<KmerIndex>,
    ref_min_dict: Arc<MinIndex>,
    total_kmers: usize,
) -> io::Result<()> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let threshold = args.threshold;
    let output = args.output.clone();
    let num_consumers = rayon::current_num_threads();

    let path = args.file.clone();
    let (record_tx, record_rx) = bounded(2 * num_consumers);
    let (result_tx, result_rx) = bounded(4 * num_consumers);

    let producer_handle = thread::spawn(move || {
        let mut ids = Vec::new();
        let mut seqs = Vec::new();
        let mut ends = Vec::new();
        let mut parser =
            FastxParser::<CONFIG_FILTER>::from_file(&path).expect("Failed to parse file to filter");
        while let Some(_) = parser.next() {
            let id = parser.get_header();
            let seq = parser.get_dna_string();
            if seq.len() < kmer_size {
                continue;
            }
            ids.extend_from_slice(id);
            seqs.extend_from_slice(seq);
            ends.push((ids.len(), seqs.len()));
            if seqs.len() >= MSG_LEN_THRESHOLD {
                let mut tmp_ids = Vec::new();
                let mut tmp_seqs = Vec::new();
                let mut tmp_ends = Vec::new();
                swap(&mut ids, &mut tmp_ids);
                swap(&mut seqs, &mut tmp_seqs);
                swap(&mut ends, &mut tmp_ends);
                if record_tx.send((tmp_ids, tmp_seqs, tmp_ends)).is_err() {
                    break;
                }
            }
        }
        if !seqs.is_empty() {
            record_tx.send((ids, seqs, ends)).unwrap();
        }
    });

    let mut consumer_handles = Vec::with_capacity(num_consumers);
    for _ in 0..num_consumers {
        let record_rx_clone = record_rx.clone();
        let result_tx_clone = result_tx.clone();
        let ref_min_dict_clone = Arc::clone(&ref_min_dict);
        let ref_kmer_dict_clone = Arc::clone(&ref_kmer_dict);

        let handle = thread::spawn(move || {
            let mini_builder = minimizers(minimizer_size, window_size);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut sk_pos = Vec::new();
            let mut mini_pos = Vec::new();
            let mut mini_words = Vec::new();
            let mut kmer_hashes = Vec::new();
            while let Ok((ids, seqs, ends)) = record_rx_clone.recv() {
                if ends.len() == 1 {
                    // a single long seq
                    let id = &ids;
                    let seq = &seqs;

                    let kmer_threshold: usize = match threshold {
                        Threshold::Absolute(n) => n,
                        Threshold::Relative(f) => {
                            (((seq.len().saturating_sub(kmer_size) + 1) as f64) * f).ceil() as usize
                        }
                    }
                    .min(total_kmers);
                    let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

                    let mut packed_seq = PackedSeqVec::default();
                    MATCH_N
                        .split(seq)
                        .filter(|&seq| seq.len() >= kmer_size)
                        .for_each(|seq| {
                            packed_seq.push_ascii(seq);
                        });

                    sk_pos.clear();
                    mini_pos.clear();
                    mini_words.clear();
                    mini_builder
                        .super_kmers(&mut sk_pos)
                        .run(packed_seq.as_slice(), &mut mini_pos);
                    let mut shared_min_count = 0;
                    mini_words.extend(mini_pos.iter().copied().map(|pos| {
                        let word = packed_seq
                            .slice((pos as usize)..(pos as usize + minimizer_size))
                            .as_u64();
                        shared_min_count += ref_min_dict_clone.contains(&word) as usize;
                        word
                    }));

                    if shared_min_count < minimizer_threshold {
                        continue;
                    }

                    kmer_hashes.clear();
                    hasher
                        .hash_kmers_simd(packed_seq.as_slice(), 1)
                        .collect_into(&mut kmer_hashes);
                    // Sharded kmer lookup using superkmer structure
                    sk_pos.push(kmer_hashes.len() as u32);
                    let mut kmer_match_count = 0;
                    let mut sk_start = 0;
                    for (sk_end, mini_word) in sk_pos
                        .iter()
                        .copied()
                        .skip(1)
                        .zip(mini_words.iter().copied())
                    {
                        let shard = (mini_word as usize) & SHARD_MASK;
                        kmer_match_count += kmer_hashes[sk_start..sk_end as usize]
                            .iter()
                            .copied()
                            .filter(|hash| ref_kmer_dict_clone[shard].contains(hash))
                            .count();
                        sk_start = sk_end as usize;
                    }

                    if kmer_match_count >= kmer_threshold {
                        let _ = result_tx_clone.send((id.clone(), seq.clone()));
                    }

                    continue;
                } else {
                    // multiple short seqs
                    let mut packed_seqs = PackedSeqVec::default();
                    let mut packed_ends = Vec::with_capacity(ends.len());
                    let mut seq_start = 0;
                    for (_, seq_end) in ends.iter().copied() {
                        let seq = &seqs[seq_start..seq_end];
                        MATCH_N
                            .split(seq)
                            .filter(|&seq| seq.len() >= kmer_size)
                            .for_each(|seq| {
                                packed_seqs.push_ascii(seq);
                            });
                        packed_ends.push(packed_seqs.len());
                        seq_start = seq_end;
                    }

                    sk_pos.clear();
                    mini_pos.clear();
                    mini_builder
                        .super_kmers(&mut sk_pos)
                        .run(packed_seqs.as_slice(), &mut mini_pos);
                    kmer_hashes.clear();

                    // Precompute minimizer words for all superkmers
                    mini_words.clear();
                    mini_words.extend(mini_pos.iter().copied().map(|pos| {
                        packed_seqs
                            .slice((pos as usize)..(pos as usize + minimizer_size))
                            .as_u64()
                    }));

                    let mut id_start = 0;
                    let mut seq_start = 0;
                    let mut packed_start = 0;
                    let mut mini_idx = 0;
                    for ((id_end, seq_end), packed_end) in ends.iter().copied().zip(packed_ends) {
                        let id = &ids[id_start..id_end];
                        let seq = &seqs[seq_start..seq_end];
                        let kmer_last = packed_end - kmer_size + 1;

                        let kmer_threshold: usize = match threshold {
                            Threshold::Absolute(n) => n,
                            Threshold::Relative(f) => {
                                (((seq.len().saturating_sub(kmer_size) + 1) as f64) * f).ceil()
                                    as usize
                            }
                        }
                        .min(total_kmers);
                        let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

                        let mini_start = mini_idx;
                        let mut shared_min_count = 0;
                        while mini_idx < sk_pos.len() && sk_pos[mini_idx] < kmer_last as u32 {
                            shared_min_count +=
                                ref_min_dict_clone.contains(&mini_words[mini_idx]) as usize;
                            mini_idx += 1;
                        }
                        let mini_end = mini_idx;
                        while mini_idx + 1 < sk_pos.len()
                            && sk_pos[mini_idx + 1] <= packed_end as u32
                        {
                            mini_idx += 1;
                        }

                        if shared_min_count < minimizer_threshold {
                            id_start = id_end;
                            seq_start = seq_end;
                            packed_start = packed_end;
                            continue;
                        }

                        if kmer_hashes.is_empty() {
                            hasher
                                .hash_kmers_simd(packed_seqs.as_slice(), 1)
                                .collect_into(&mut kmer_hashes);
                        }

                        // Sharded kmer lookup: iterate superkmers to use the right shard
                        let mut kmer_match_count = 0;
                        let mut kh_cursor = packed_start;
                        for i in mini_start..mini_end {
                            let sk_end = if i + 1 < mini_end {
                                sk_pos[i + 1] as usize
                            } else {
                                kmer_last
                            };
                            let shard = (mini_words[i] as usize) & SHARD_MASK;
                            kmer_match_count += kmer_hashes[kh_cursor..sk_end]
                                .iter()
                                .copied()
                                .filter(|hash| ref_kmer_dict_clone[shard].contains(hash))
                                .count();
                            kh_cursor = sk_end;
                        }

                        if kmer_match_count >= kmer_threshold {
                            let _ = result_tx_clone.send((id.to_vec(), seq.to_vec()));
                        }

                        id_start = id_end;
                        seq_start = seq_end;
                        packed_start = packed_end;
                    }
                }
            }
        });

        consumer_handles.push(handle);
    }

    drop(result_tx);

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

    producer_handle.join().expect("Producer thread panicked");
    for handle in consumer_handles {
        handle.join().expect("Consumer thread panicked");
    }
    let _ = printer_handle.join().expect("Printer thread panicked");

    Ok(())
}

use clap::Parser as ClapParser;
use crossbeam_channel::bounded;
use helicase::input::*;
use helicase::*;
use regex::bytes::{Regex, RegexBuilder};
use rustc_hash::FxHashSet;
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::{PackedSeqVec, Seq, SeqVec};
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};

use core::fmt::Display;
use core::mem::swap;
use core::str::FromStr;
use std::fs::File;
use std::io::{self, BufWriter, Write, stdout};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::Instant;

type MinIndex = FxHashSet<u64>;
type KmerIndex = FxHashSet<u32>;
type IndexShards = Vec<Mutex<IndexShard>>;
type FinalIndexShards = Vec<IndexShard>;

const MSG_LEN_THRESHOLD: usize = 8000; // small enough for long reads
const INDEX_BATCH_BASES: usize = 1 << 20;
const INDEX_NUM_SHARDS: usize = 1 << 16;

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

#[derive(Default)]
struct IndexShard {
    mins: MinIndex,
    kmers: KmerIndex,
}

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
    let index_shards = index_reference(&args)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    let ref_index_shards = Arc::new(index_shards);
    let total_kmers: usize = ref_index_shards.iter().map(|s| s.kmers.len()).sum();
    let total_minimizers: usize = ref_index_shards.iter().map(|s| s.mins.len()).sum();
    eprintln!(
        "Indexed {} k-mers and {} minimizers across {} shards.",
        total_kmers, total_minimizers, INDEX_NUM_SHARDS
    );

    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    process_query_streaming(&args, Arc::clone(&ref_index_shards), total_kmers)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );

    Ok(())
}

fn default_threads(threads: Option<usize>) -> usize {
    threads.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    })
}

#[inline(always)]
fn minimizer_shard(minimizer: u64, shard_mask: usize) -> usize {
    let mut x = minimizer;
    // splitmix64 finalizer: fast and well-distributed for shard partitioning.
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x as usize) & shard_mask
}

fn index_reference(args: &Args) -> io::Result<FinalIndexShards> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let num_workers = default_threads(args.threads).max(1);
    let num_shards = INDEX_NUM_SHARDS;
    let shard_mask = num_shards - 1;
    let pattern_path = args.patterns.clone();

    let shards: Arc<IndexShards> = Arc::new(
        (0..num_shards)
            .map(|_| Mutex::new(IndexShard::default()))
            .collect(),
    );

    let (batch_tx, batch_rx) = bounded::<Vec<PackedSeqVec>>(2 * num_workers);
    let producer_handle = thread::spawn(move || {
        let mut parser = FastxParser::<CONFIG_INDEX>::from_file_in_ram(&pattern_path)
            .expect("Failed to parse file containing patterns");
        let mut batch = Vec::with_capacity(4096);
        let mut batch_bases = 0usize;

        while let Some(_) = parser.next() {
            let packed_seq = parser.get_packed_seq();
            if packed_seq.len() < kmer_size {
                continue;
            }

            batch_bases += packed_seq.len();
            batch.push(packed_seq.to_vec());

            if batch_bases >= INDEX_BATCH_BASES {
                if batch_tx.send(batch).is_err() {
                    return;
                }
                batch = Vec::with_capacity(4096);
                batch_bases = 0;
            }
        }

        if !batch.is_empty() {
            let _ = batch_tx.send(batch);
        }
    });

    let mut worker_handles = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let batch_rx = batch_rx.clone();
        let shards = Arc::clone(&shards);
        let handle = thread::spawn(move || {
            let mini_builder = minimizers(minimizer_size, window_size);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut sk_pos = Vec::new();
            let mut mini_pos = Vec::new();
            let mut kmer_hashes = Vec::new();

            while let Ok(batch) = batch_rx.recv() {
                for packed_seq in batch {
                    let packed_seq = packed_seq.as_slice();
                    sk_pos.clear();
                    mini_pos.clear();
                    mini_builder
                        .super_kmers(&mut sk_pos)
                        .run(packed_seq, &mut mini_pos);

                    if mini_pos.is_empty() {
                        continue;
                    }

                    kmer_hashes.clear();
                    hasher
                        .hash_kmers_simd(packed_seq, 1)
                        .collect_into(&mut kmer_hashes);
                    if kmer_hashes.is_empty() {
                        continue;
                    }

                    let last_kmer = kmer_hashes.len();
                    for idx in 0..mini_pos.len() {
                        let run_start = sk_pos[idx] as usize;
                        let run_end = if idx + 1 < sk_pos.len() {
                            (sk_pos[idx + 1] as usize).min(last_kmer)
                        } else {
                            last_kmer
                        };
                        if run_start >= run_end {
                            continue;
                        }

                        let min_start = mini_pos[idx] as usize;
                        let minimizer = packed_seq
                            .slice(min_start..(min_start + minimizer_size))
                            .as_u64();
                        let shard_idx = minimizer_shard(minimizer, shard_mask);
                        let mut shard = shards[shard_idx].lock().unwrap();
                        shard.mins.insert(minimizer);
                        shard
                            .kmers
                            .extend(kmer_hashes[run_start..run_end].iter().copied());
                    }
                }
            }
        });
        worker_handles.push(handle);
    }

    producer_handle.join().expect("Producer thread panicked");
    for handle in worker_handles {
        handle.join().expect("Index worker thread panicked");
    }

    let shards = match Arc::try_unwrap(shards) {
        Ok(shards) => shards,
        Err(_) => panic!("Could not acquire ownership of index shards"),
    };

    let mut out = Vec::with_capacity(num_shards);
    for shard in shards {
        out.push(shard.into_inner().unwrap());
    }
    Ok(out)
}

fn process_query_streaming(
    args: &Args,
    ref_index_shards: Arc<FinalIndexShards>,
    total_kmers: usize,
) -> io::Result<()> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let threshold = args.threshold;
    let output = args.output.clone();
    let num_consumers = default_threads(args.threads).max(1);

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
        let ref_index_shards_clone = Arc::clone(&ref_index_shards);

        let handle = thread::spawn(move || {
            let mini_builder = minimizers(minimizer_size, window_size);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut sk_pos = Vec::new();
            let mut mini_pos = Vec::new();
            let mut kmer_hashes = Vec::new();
            let mut run_min_words = Vec::new();
            let mut run_shards = Vec::new();
            let shard_mask = INDEX_NUM_SHARDS - 1;
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
                    mini_builder
                        .super_kmers(&mut sk_pos)
                        .run(packed_seq.as_slice(), &mut mini_pos);
                    run_min_words.clear();
                    run_shards.clear();
                    run_min_words.reserve(mini_pos.len());
                    run_shards.reserve(mini_pos.len());
                    for &pos in &mini_pos {
                        let word = packed_seq
                            .slice((pos as usize)..(pos as usize + minimizer_size))
                            .as_u64();
                        let shard_idx = minimizer_shard(word, shard_mask);
                        run_min_words.push(word);
                        run_shards.push(shard_idx);
                    }
                    let shared_min_count = run_min_words
                        .iter()
                        .zip(run_shards.iter().copied())
                        .filter(|(word, shard_idx)| {
                            ref_index_shards_clone[*shard_idx].mins.contains(word)
                        })
                        .count();

                    if shared_min_count < minimizer_threshold {
                        continue;
                    }

                    kmer_hashes.clear();
                    hasher
                        .hash_kmers_simd(packed_seq.as_slice(), 1)
                        .collect_into(&mut kmer_hashes);
                    let last_kmer = kmer_hashes.len();
                    let mut kmer_match_count = 0usize;
                    for idx in 0..mini_pos.len() {
                        let run_start = sk_pos[idx] as usize;
                        let run_end = if idx + 1 < sk_pos.len() {
                            (sk_pos[idx + 1] as usize).min(last_kmer)
                        } else {
                            last_kmer
                        };
                        if run_start >= run_end {
                            continue;
                        }
                        let shard_idx = run_shards[idx];
                        let shard = &ref_index_shards_clone[shard_idx];
                        kmer_match_count += kmer_hashes[run_start..run_end]
                            .iter()
                            .copied()
                            .filter(|hash| shard.kmers.contains(hash))
                            .count();
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
                    run_min_words.clear();
                    run_shards.clear();
                    run_min_words.reserve(mini_pos.len());
                    run_shards.reserve(mini_pos.len());
                    for &pos in &mini_pos {
                        let word = packed_seqs
                            .slice((pos as usize)..(pos as usize + minimizer_size))
                            .as_u64();
                        run_min_words.push(word);
                        run_shards.push(minimizer_shard(word, shard_mask));
                    }
                    kmer_hashes.clear();

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

                        let seq_mini_start = mini_idx;
                        let mut shared_min_count = 0;
                        while mini_idx < sk_pos.len() && sk_pos[mini_idx] < kmer_last as u32 {
                            let run_pos = sk_pos[mini_idx] as usize;
                            if run_pos < packed_start {
                                mini_idx += 1;
                                continue;
                            }
                            let word = run_min_words[mini_idx];
                            let shard_idx = run_shards[mini_idx];
                            shared_min_count +=
                                if ref_index_shards_clone[shard_idx].mins.contains(&word) {
                                    1
                                } else {
                                    0
                                };
                            mini_idx += 1;
                        }
                        let seq_mini_stop = mini_idx;
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

                        let mut kmer_match_count = 0usize;
                        for idx in seq_mini_start..seq_mini_stop {
                            let run_start = sk_pos[idx] as usize;
                            if run_start < packed_start {
                                continue;
                            }
                            let run_end = if idx + 1 < sk_pos.len() {
                                (sk_pos[idx + 1] as usize).min(kmer_last)
                            } else {
                                kmer_last
                            };
                            if run_start >= run_end {
                                continue;
                            }
                            let shard_idx = run_shards[idx];
                            let shard = &ref_index_shards_clone[shard_idx];
                            kmer_match_count += kmer_hashes[run_start..run_end]
                                .iter()
                                .copied()
                                .filter(|hash| shard.kmers.contains(hash))
                                .count();
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

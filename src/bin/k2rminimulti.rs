use clap::{ArgAction, CommandFactory, Parser as ClapParser};
use crossbeam_channel::bounded;
use helicase::input::*;
use helicase::*;
use regex::bytes::{Regex, RegexBuilder};
use rustc_hash::{FxHashMap, FxHashSet};
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::{PackedSeqVec, Seq, SeqVec};
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};

use core::fmt::Display;
use core::mem::swap;
use core::str::FromStr;
use std::fs::File;
use std::io::{self, BufWriter, Write, stdout};
use std::sync::{Arc, LazyLock};
use std::thread;
use std::time::Instant;

type QueryMask = u32;
type MinIndex = FxHashMap<u32, QueryMask>;
type KmerIndex = FxHashMap<u32, QueryMask>;
type LocalMinIndex = FxHashSet<u32>;
type LocalKmerIndex = FxHashSet<u32>;

const MAX_QUERIES: usize = QueryMask::BITS as usize;
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
                Err("Absolute threshold must be >= 1".to_string())
            } else {
                Ok(Self::Absolute(val))
            }
        } else if let Ok(val) = s.parse::<f64>() {
            if val.is_nan() || val.is_sign_negative() || val == 0. || val > 1. {
                Err("Relative threshold must be in (0, 1]".to_string())
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

#[derive(Debug, Clone)]
struct QuerySpec {
    patterns: String,
    threshold: Threshold,
}

#[derive(ClapParser, Debug)]
#[command(
    author,
    version,
    about = "Filter reads by multiple k-mer query constraints",
    long_about = None
)]
struct Args {
    /// FASTA/Q file to filter (possibly compressed)
    #[arg()]
    file: String,
    /// FASTA/Q file containing k-mers of interest and its threshold; may be repeated
    #[arg(
        short = 'c',
        long = "constraint",
        value_names = ["PATTERNS", "THRESHOLD"],
        num_args = 2,
        action = ArgAction::Append,
        required = true
    )]
    constraints: Vec<String>,
    /// Output file for filtered sequences [default: stdout]
    #[arg(short)]
    output: Option<String>,
    /// K-mer size
    #[arg(short, default_value_t = 31)]
    k: usize,
    /// Minimizer size, must be <= k, up to 29
    #[arg(short, default_value_t = 21)]
    m: usize,
    /// Number of threads [default: all]
    #[arg(short = 'T', long)]
    threads: Option<usize>,
}

fn parse_constraints(values: &[String]) -> Result<Vec<QuerySpec>, String> {
    if values.is_empty() {
        return Err("At least one -c/--constraint is required".to_string());
    }
    if values.len() % 2 != 0 {
        return Err("Each -c/--constraint needs both PATTERNS and THRESHOLD".to_string());
    }

    let query_count = values.len() / 2;
    if query_count > MAX_QUERIES {
        return Err(format!(
            "k2rminimulti supports at most {MAX_QUERIES} query files"
        ));
    }

    values
        .chunks_exact(2)
        .map(|chunk| {
            let threshold = chunk[1]
                .parse::<Threshold>()
                .map_err(|err| format!("Invalid threshold for {}: {err}", chunk[0]))?;
            Ok(QuerySpec {
                patterns: chunk[0].clone(),
                threshold,
            })
        })
        .collect()
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

fn query_bit(query_id: usize) -> QueryMask {
    1u32 << query_id
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    if args.m > args.k {
        Args::command()
            .error(
                clap::error::ErrorKind::ValueValidation,
                "Minimizer size must be <= k",
            )
            .exit();
    }

    let queries = parse_constraints(&args.constraints).unwrap_or_else(|err| {
        Args::command()
            .error(clap::error::ErrorKind::ValueValidation, err)
            .exit()
    });

    eprintln!(
        "Running with k={}, m={} and {} query constraints",
        args.k,
        args.m,
        queries.len()
    );

    eprintln!("Indexing k-mers and minimizers of interest...");
    let start = Instant::now();
    let (min_dict, kmer_dict, query_kmer_counts) = index_references(&args, &queries)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    let ref_min_dict = Arc::new(min_dict);
    let ref_kmer_dict = Arc::new(kmer_dict);
    let query_kmer_counts = Arc::new(query_kmer_counts);
    eprintln!(
        "Indexed {} distinct k-mer hashes and {} distinct minimizers across {} query files.",
        ref_kmer_dict.len(),
        ref_min_dict.len(),
        queries.len()
    );

    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    let thresholds = Arc::new(
        queries
            .iter()
            .map(|query| query.threshold)
            .collect::<Vec<_>>(),
    );
    process_query_streaming(
        &args,
        Arc::clone(&thresholds),
        Arc::clone(&query_kmer_counts),
        Arc::clone(&ref_kmer_dict),
        Arc::clone(&ref_min_dict),
    )?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );

    Ok(())
}

fn index_references(
    args: &Args,
    queries: &[QuerySpec],
) -> io::Result<(MinIndex, KmerIndex, Vec<usize>)> {
    let mut dict_mini = MinIndex::default();
    let mut dict_kmer = KmerIndex::default();
    let mut query_kmer_counts = Vec::with_capacity(queries.len());

    for (query_id, query) in queries.iter().enumerate() {
        let mask = query_bit(query_id);
        let (local_mini, local_kmer) = index_one_reference(args, &query.patterns)?;
        let local_kmer_count = local_kmer.len();

        for minimizer in local_mini {
            *dict_mini.entry(minimizer).or_insert(0) |= mask;
        }
        for kmer in local_kmer {
            *dict_kmer.entry(kmer).or_insert(0) |= mask;
        }

        query_kmer_counts.push(local_kmer_count);
    }

    Ok((dict_mini, dict_kmer, query_kmer_counts))
}

fn index_one_reference(args: &Args, patterns: &str) -> io::Result<(LocalMinIndex, LocalKmerIndex)> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let mini_builder = minimizers(minimizer_size, window_size);
    let hasher = NtHasher::<false>::new(kmer_size);
    let mut parser = FastxParser::<CONFIG_INDEX>::from_file_in_ram(patterns)
        .expect("Failed to parse file containing patterns");
    let mut dict_mini = LocalMinIndex::default();
    let mut dict_kmer = LocalKmerIndex::default();
    let mut mini_pos = Vec::new();
    let mut kmer_hashes = Vec::new();

    while let Some(_) = parser.next() {
        let packed_seq = parser.get_packed_seq();
        mini_pos.clear();
        kmer_hashes.clear();
        mini_builder.run(packed_seq, &mut mini_pos);
        hasher
            .hash_kmers_simd(packed_seq, 1)
            .collect_into(&mut kmer_hashes);
        let mini_iter = mini_pos.iter().copied().map(|pos| {
            packed_seq
                .slice((pos as usize)..(pos as usize + minimizer_size))
                .as_u64() as u32
        });
        dict_mini.extend(mini_iter);
        dict_kmer.extend(&kmer_hashes);
    }

    Ok((dict_mini, dict_kmer))
}

fn fill_kmer_thresholds(
    thresholds: &[Threshold],
    query_kmer_counts: &[usize],
    seq_len: usize,
    kmer_size: usize,
    out: &mut Vec<usize>,
) -> QueryMask {
    out.clear();
    out.reserve(thresholds.len());
    let read_kmer_count = seq_len.saturating_sub(kmer_size) + 1;
    let mut needed_mask = 0;

    for (query_id, threshold) in thresholds.iter().copied().enumerate() {
        let kmer_threshold = match threshold {
            Threshold::Absolute(n) => n,
            Threshold::Relative(f) => ((read_kmer_count as f64) * f).ceil() as usize,
        }
        .min(query_kmer_counts[query_id]);

        out.push(kmer_threshold);
        if kmer_threshold > 0 {
            needed_mask |= query_bit(query_id);
        }
    }

    needed_mask
}

fn fill_minimizer_thresholds(kmer_thresholds: &[usize], window_size: usize, out: &mut Vec<usize>) {
    out.clear();
    out.reserve(kmer_thresholds.len());
    out.extend(
        kmer_thresholds
            .iter()
            .copied()
            .map(|threshold| threshold.div_ceil(window_size)),
    );
}

fn reset_counts(counts: &mut Vec<usize>, len: usize) {
    counts.clear();
    counts.resize(len, 0);
}

fn update_counts_from_mask(
    mut mask: QueryMask,
    counts: &mut [usize],
    thresholds: &[usize],
    needed_mask: &mut QueryMask,
) {
    while mask != 0 {
        let query_id = mask.trailing_zeros() as usize;
        counts[query_id] += 1;
        if counts[query_id] >= thresholds[query_id] {
            *needed_mask &= !query_bit(query_id);
        }
        mask &= mask - 1;
    }
}

fn process_query_streaming(
    args: &Args,
    thresholds: Arc<Vec<Threshold>>,
    query_kmer_counts: Arc<Vec<usize>>,
    ref_kmer_dict: Arc<KmerIndex>,
    ref_min_dict: Arc<MinIndex>,
) -> io::Result<()> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let output = args.output.clone();
    let num_consumers = args.threads.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

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
            let _ = record_tx.send((ids, seqs, ends));
        }
    });

    let mut consumer_handles = Vec::with_capacity(num_consumers);
    for _ in 0..num_consumers {
        let record_rx_clone = record_rx.clone();
        let result_tx_clone = result_tx.clone();
        let ref_min_dict_clone = Arc::clone(&ref_min_dict);
        let ref_kmer_dict_clone = Arc::clone(&ref_kmer_dict);
        let thresholds_clone = Arc::clone(&thresholds);
        let query_kmer_counts_clone = Arc::clone(&query_kmer_counts);

        let handle = thread::spawn(move || {
            let query_count = thresholds_clone.len();
            let mini_builder = minimizers(minimizer_size, window_size);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut sk_pos = Vec::new();
            let mut mini_pos = Vec::new();
            let mut kmer_hashes = Vec::new();
            let mut kmer_thresholds = Vec::with_capacity(query_count);
            let mut minimizer_thresholds = Vec::with_capacity(query_count);
            let mut kmer_counts = Vec::with_capacity(query_count);
            let mut minimizer_counts = Vec::with_capacity(query_count);

            while let Ok((ids, seqs, ends)) = record_rx_clone.recv() {
                if ends.len() == 1 {
                    // a single long seq
                    let id = &ids;
                    let seq = &seqs;

                    let mut kmer_needed_mask = fill_kmer_thresholds(
                        &thresholds_clone,
                        &query_kmer_counts_clone,
                        seq.len(),
                        kmer_size,
                        &mut kmer_thresholds,
                    );
                    if kmer_needed_mask == 0 {
                        let _ = result_tx_clone.send((id.clone(), seq.clone()));
                        continue;
                    }

                    fill_minimizer_thresholds(
                        &kmer_thresholds,
                        window_size,
                        &mut minimizer_thresholds,
                    );

                    let mut packed_seq = PackedSeqVec::default();
                    MATCH_N
                        .split(seq)
                        .filter(|&seq| seq.len() >= kmer_size)
                        .for_each(|seq| {
                            packed_seq.push_ascii(seq);
                        });

                    let mut minimizer_needed_mask = kmer_needed_mask;
                    reset_counts(&mut minimizer_counts, query_count);
                    mini_pos.clear();
                    mini_builder.run(packed_seq.as_slice(), &mut mini_pos);
                    for pos in mini_pos.iter().copied() {
                        let word = packed_seq
                            .slice((pos as usize)..(pos as usize + minimizer_size))
                            .as_u64() as u32;
                        let Some(mask) = ref_min_dict_clone.get(&word) else {
                            continue;
                        };
                        let relevant_mask = *mask & minimizer_needed_mask;
                        if relevant_mask != 0 {
                            update_counts_from_mask(
                                relevant_mask,
                                &mut minimizer_counts,
                                &minimizer_thresholds,
                                &mut minimizer_needed_mask,
                            );
                            if minimizer_needed_mask == 0 {
                                break;
                            }
                        }
                    }

                    if minimizer_needed_mask != 0 {
                        continue;
                    }

                    reset_counts(&mut kmer_counts, query_count);
                    kmer_hashes.clear();
                    hasher
                        .hash_kmers_simd(packed_seq.as_slice(), 1)
                        .collect_into(&mut kmer_hashes);
                    for hash in kmer_hashes.iter().copied() {
                        let Some(mask) = ref_kmer_dict_clone.get(&hash) else {
                            continue;
                        };
                        let relevant_mask = *mask & kmer_needed_mask;
                        if relevant_mask != 0 {
                            update_counts_from_mask(
                                relevant_mask,
                                &mut kmer_counts,
                                &kmer_thresholds,
                                &mut kmer_needed_mask,
                            );
                            if kmer_needed_mask == 0 {
                                break;
                            }
                        }
                    }

                    if kmer_needed_mask == 0 {
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
                    let mut kmer_hashes_ready = false;

                    let mut id_start = 0;
                    let mut seq_start = 0;
                    let mut packed_start = 0;
                    let mut mini_idx = 0;
                    for ((id_end, seq_end), packed_end) in ends.iter().copied().zip(packed_ends) {
                        let id = &ids[id_start..id_end];
                        let seq = &seqs[seq_start..seq_end];

                        let mut kmer_needed_mask = fill_kmer_thresholds(
                            &thresholds_clone,
                            &query_kmer_counts_clone,
                            seq.len(),
                            kmer_size,
                            &mut kmer_thresholds,
                        );
                        if kmer_needed_mask == 0 {
                            let _ = result_tx_clone.send((id.to_vec(), seq.to_vec()));
                            id_start = id_end;
                            seq_start = seq_end;
                            packed_start = packed_end;
                            continue;
                        }

                        if packed_end.saturating_sub(packed_start) < kmer_size {
                            id_start = id_end;
                            seq_start = seq_end;
                            packed_start = packed_end;
                            continue;
                        }

                        let kmer_last = packed_end - kmer_size + 1;
                        fill_minimizer_thresholds(
                            &kmer_thresholds,
                            window_size,
                            &mut minimizer_thresholds,
                        );

                        let mut minimizer_needed_mask = kmer_needed_mask;
                        reset_counts(&mut minimizer_counts, query_count);
                        while mini_idx < sk_pos.len() && sk_pos[mini_idx] < kmer_last as u32 {
                            if minimizer_needed_mask != 0 {
                                let pos = mini_pos[mini_idx] as usize;
                                let word =
                                    packed_seqs.slice(pos..(pos + minimizer_size)).as_u64() as u32;
                                if let Some(mask) = ref_min_dict_clone.get(&word) {
                                    let relevant_mask = *mask & minimizer_needed_mask;
                                    if relevant_mask != 0 {
                                        update_counts_from_mask(
                                            relevant_mask,
                                            &mut minimizer_counts,
                                            &minimizer_thresholds,
                                            &mut minimizer_needed_mask,
                                        );
                                    }
                                }
                            }
                            mini_idx += 1;
                        }
                        while mini_idx + 1 < sk_pos.len()
                            && sk_pos[mini_idx + 1] <= packed_end as u32
                        {
                            mini_idx += 1;
                        }

                        if minimizer_needed_mask != 0 {
                            id_start = id_end;
                            seq_start = seq_end;
                            packed_start = packed_end;
                            continue;
                        }

                        if !kmer_hashes_ready {
                            hasher
                                .hash_kmers_simd(packed_seqs.as_slice(), 1)
                                .collect_into(&mut kmer_hashes);
                            kmer_hashes_ready = true;
                        }

                        reset_counts(&mut kmer_counts, query_count);
                        for hash in kmer_hashes[packed_start..kmer_last].iter().copied() {
                            let Some(mask) = ref_kmer_dict_clone.get(&hash) else {
                                continue;
                            };
                            let relevant_mask = *mask & kmer_needed_mask;
                            if relevant_mask != 0 {
                                update_counts_from_mask(
                                    relevant_mask,
                                    &mut kmer_counts,
                                    &kmer_thresholds,
                                    &mut kmer_needed_mask,
                                );
                                if kmer_needed_mask == 0 {
                                    break;
                                }
                            }
                        }

                        if kmer_needed_mask == 0 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_constraints_as_pairs() {
        let values = vec![
            "q1.fa".to_string(),
            "2".to_string(),
            "q2.fa".to_string(),
            "0.5".to_string(),
        ];
        let parsed = parse_constraints(&values).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].patterns, "q1.fa");
        assert!(matches!(parsed[0].threshold, Threshold::Absolute(2)));
        assert!(matches!(parsed[1].threshold, Threshold::Relative(0.5)));
    }

    #[test]
    fn rejects_more_than_32_queries() {
        let mut values = Vec::new();
        for query_id in 0..33 {
            values.push(format!("q{query_id}.fa"));
            values.push("1".to_string());
        }
        assert!(parse_constraints(&values).is_err());
    }
}

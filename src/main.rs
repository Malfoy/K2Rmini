use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::LazyLock;
use std::thread;
use std::time::Instant;

use clap::Parser;
use crossbeam_channel::unbounded;
use needletail::parse_fastx_file;
use packed_seq::{PackedSeq, PackedSeqVec, Seq, SeqVec};
use regex::bytes::{Regex, RegexBuilder};
use rustc_hash::FxHashSet;
use simd_minimizers::minimizer_positions;
use simd_minimizers::private::collect::collect_into;
use simd_minimizers::private::nthash::{nthash_seq_scalar, nthash_seq_simd, NtHasher};
use simd_minimizers::scalar::minimizer_positions_scalar;

type MinIndex = FxHashSet<usize>;
type KmerIndex = FxHashSet<u32>;

const SIMD_LEN_THRESHOLD: usize = 1000;

static MATCH_N: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(r"[N]+")
        .case_insensitive(true)
        .unicode(false)
        .build()
        .unwrap()
});

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Reference file (FASTA, possibly compressed)
    #[arg(short, long)]
    reference: String,
    /// File to query (FASTA, possibly compressed)
    #[arg(short, long)]
    query: String,
    /// K-mer size
    #[arg(short, default_value_t = 31)]
    k: usize,
    /// Minimizer size
    #[arg(short, default_value_t = 21)]
    m: usize,
    /// K-mer threshold
    #[arg(short, default_value_t = 1000)]
    threshold: usize,
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
    let reference_path = &args.reference;
    let query_path = &args.query;
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    assert!(minimizer_size <= kmer_size);
    let kmer_threshold: usize = args.threshold;

    eprintln!("Indexing reference k-mers and minimizers...");
    let start = Instant::now();
    let (min_dict, kmer_dict) = index_reference(reference_path, kmer_size, minimizer_size)?;
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    let ref_min_dict = Arc::new(min_dict);
    let ref_kmer_dict = Arc::new(kmer_dict);
    eprintln!(
        "Reference k-mer index contains {} entries.",
        ref_kmer_dict.len()
    );
    eprintln!(
        "Reference minimizer index contains {} entries.",
        ref_min_dict.len()
    );

    eprintln!("Processing query sequences using a producer-consumer model...");
    let start = Instant::now();
    process_query_streaming(
        query_path,
        kmer_size,
        minimizer_size,
        kmer_threshold,
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

fn index_reference<P: AsRef<Path>>(
    ref_path: P,
    kmer_size: usize,
    minimizer_size: usize,
) -> io::Result<(MinIndex, KmerIndex)> {
    let window_size: usize = kmer_size - minimizer_size + 1;
    let mut reader = parse_fastx_file(ref_path).expect("Failed to parse reference file");
    let mut dict_mini = MinIndex::default();
    let mut dict_kmer = KmerIndex::default();
    let mut mini_pos = Vec::new();
    let mut kmer_hashes = Vec::new();

    while let Some(result) = reader.next() {
        let record = result.map_err(io::Error::other)?;
        let seq = &record.seq();
        MATCH_N
            .split(seq)
            .filter(|&seq| seq.len() >= kmer_size)
            .for_each(|seq| {
                let packed_seq = PackedSeqVec::from_ascii(seq);
                mini_pos.clear();
                kmer_hashes.clear();
                if seq.len() >= kmer_size + SIMD_LEN_THRESHOLD {
                    minimizer_positions(
                        packed_seq.as_slice(),
                        minimizer_size,
                        window_size,
                        &mut mini_pos,
                    );
                    let nthash_iter = nthash_seq_simd::<false, PackedSeq, NtHasher>(
                        packed_seq.as_slice(),
                        kmer_size,
                        1,
                    );
                    collect_into(nthash_iter, &mut kmer_hashes);
                } else {
                    minimizer_positions_scalar(
                        packed_seq.as_slice(),
                        minimizer_size,
                        window_size,
                        &mut mini_pos,
                    );
                    let nthash_iter =
                        nthash_seq_scalar::<false, NtHasher>(packed_seq.as_slice(), kmer_size);
                    kmer_hashes.extend(nthash_iter);
                }
                let mini_iter = mini_pos.iter().copied().map(|pos| {
                    packed_seq
                        .slice((pos as usize)..(pos as usize + minimizer_size))
                        .to_word()
                });
                dict_mini.extend(mini_iter);
                dict_kmer.extend(&kmer_hashes);
            });
    }
    Ok((dict_mini, dict_kmer))
}

fn process_query_streaming<P: AsRef<Path>>(
    query_path: P,
    kmer_size: usize,
    minimizer_size: usize,
    kmer_threshold: usize,
    ref_kmer_dict: Arc<KmerIndex>,
    ref_min_dict: Arc<MinIndex>,
) -> io::Result<()> {
    let window_size: usize = kmer_size - minimizer_size + 1;
    let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

    let mut parser = parse_fastx_file(query_path).expect("Failed to parse query file");
    let (record_tx, record_rx) = unbounded();
    let (result_tx, result_rx) = unbounded();

    let producer_handle = thread::spawn(move || {
        while let Some(result) = parser.next() {
            match result {
                Ok(record) => {
                    let id = record.id().to_vec(); // BOF
                    let seq = record.seq().to_vec(); // BOF
                    if record_tx.send((id, seq)).is_err() {
                        break;
                    }
                }
                Err(e) => eprintln!("Error reading record: {}", e),
            }
        }
    });

    let num_consumers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut consumer_handles = Vec::with_capacity(num_consumers);

    for _ in 0..num_consumers {
        let record_rx_clone = record_rx.clone();
        let result_tx_clone = result_tx.clone();
        let ref_min_dict_clone = Arc::clone(&ref_min_dict);
        let ref_kmer_dict_clone = Arc::clone(&ref_kmer_dict);

        let handle = thread::spawn(move || {
            let mut mini_pos = Vec::new();
            let mut kmer_hashes = Vec::new();
            while let Ok((id, seq)) = record_rx_clone.recv() {
                let mut shared_min_count = 0;
                let seqs: Vec<_> = MATCH_N
                    .split(&seq)
                    .filter(|&seq| seq.len() >= kmer_size)
                    .map(|seq| {
                        let packed_seq = PackedSeqVec::from_ascii(seq);
                        mini_pos.clear();
                        if seq.len() >= kmer_size + SIMD_LEN_THRESHOLD {
                            minimizer_positions(
                                packed_seq.as_slice(),
                                minimizer_size,
                                window_size,
                                &mut mini_pos,
                            );
                        } else {
                            minimizer_positions_scalar(
                                packed_seq.as_slice(),
                                minimizer_size,
                                window_size,
                                &mut mini_pos,
                            );
                        }
                        shared_min_count += mini_pos
                            .iter()
                            .copied()
                            .map(|pos| {
                                packed_seq
                                    .slice((pos as usize)..(pos as usize + minimizer_size))
                                    .to_word()
                            })
                            .filter(|word| ref_min_dict_clone.contains(word))
                            .count();
                        packed_seq
                    })
                    .collect();

                if shared_min_count < minimizer_threshold {
                    continue;
                }

                let mut kmer_match_count = 0;
                for packed_seq in seqs {
                    kmer_hashes.clear();
                    let nthash_iter = nthash_seq_simd::<false, PackedSeq, NtHasher>(
                        packed_seq.as_slice(),
                        kmer_size,
                        1,
                    );
                    collect_into(nthash_iter, &mut kmer_hashes);
                    kmer_match_count += kmer_hashes
                        .iter()
                        .copied()
                        .filter(|hash| ref_kmer_dict_clone.contains(hash))
                        .count();
                }

                if kmer_match_count >= kmer_threshold {
                    let _ = result_tx_clone.send((id, seq));
                }
            }
        });

        consumer_handles.push(handle);
    }

    drop(result_tx);

    let printer_handle = thread::spawn(move || {
        for (id, seq) in result_rx.iter() {
            print!(">");
            std::io::stdout().write_all(&id).unwrap();
            println!();
            std::io::stdout().write_all(&seq).unwrap();
            println!();
        }
    });

    producer_handle.join().expect("Producer thread panicked");
    for handle in consumer_handles {
        handle.join().expect("Consumer thread panicked");
    }
    printer_handle.join().expect("Printer thread panicked");

    Ok(())
}

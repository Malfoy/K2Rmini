use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use clap::Parser;
use crossbeam_channel::unbounded;
use needletail::parse_fastx_file;
use packed_seq::{PackedSeqVec, Seq, SeqVec};
use regex::bytes::RegexBuilder;
use rustc_hash::FxHashSet as HashSet;

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

    let start = Instant::now();
    eprintln!("Indexing reference k-mers...");
    let ref_kmer_dict = Arc::new(index_reference_kmers(reference_path, kmer_size)?);
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    eprintln!(
        "Reference k-mer index contains {} entries.",
        ref_kmer_dict.len()
    );

    let start = Instant::now();
    eprintln!("Indexing reference minimizers...");
    let ref_min_dict = Arc::new(index_reference_minimizers(
        reference_path,
        kmer_size,
        minimizer_size,
    )?);
    eprintln!(
        "Took {:.02} s, RAM: {:.03} GB",
        start.elapsed().as_secs_f64(),
        mem_usage_gb()
    );
    eprintln!(
        "Reference minimizer index contains {} entries.",
        ref_min_dict.len()
    );

    let start = Instant::now();
    eprintln!("Processing query sequences using a producer-consumer model...");
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

fn index_reference_kmers<P: AsRef<Path>>(
    ref_path: P,
    kmer_size: usize,
) -> io::Result<HashSet<u64>> {
    let mut dict = HashSet::default();
    let mut reader = parse_fastx_file(ref_path).expect("Failed to parse reference file");
    let match_n = RegexBuilder::new(r"[N]+")
        .case_insensitive(true)
        .unicode(false)
        .build()
        .unwrap();

    while let Some(result) = reader.next() {
        let record = result.map_err(io::Error::other)?;
        let seq = &record.seq();
        match_n.split(seq).for_each(|seq| {
            // for hash in nthash::nthash(seq, kmer_size) {
            //     dict.insert(hash);
            // }
            let hasher = ahash::RandomState::with_seed(42);
            for kmer in seq.windows(kmer_size) {
                dict.insert(hasher.hash_one(kmer));
            }
        });
    }
    Ok(dict)
}

fn index_reference_minimizers<P: AsRef<Path>>(
    ref_path: P,
    kmer_size: usize,
    minimizer_size: usize,
) -> io::Result<HashSet<u64>> {
    let window_size: usize = kmer_size - minimizer_size + 1;
    let mut dict = HashSet::default();
    let mut reader = parse_fastx_file(ref_path).expect("Failed to parse reference file");
    let match_n = RegexBuilder::new(r"[N]+")
        .case_insensitive(true)
        .unicode(false)
        .build()
        .unwrap();

    while let Some(result) = reader.next() {
        let record = result.map_err(io::Error::other)?;
        let seq = &record.seq();
        match_n
            .split(seq)
            .filter(|&seq| seq.len() >= kmer_size)
            .for_each(|seq| {
                let packed_seq = PackedSeqVec::from_ascii(seq);
                let mut fwd_pos = Vec::new();
                simd_minimizers::minimizer_positions(
                    packed_seq.as_slice(),
                    minimizer_size,
                    window_size,
                    &mut fwd_pos,
                );
                for pos in fwd_pos {
                    let shard = packed_seq
                        .slice((pos as usize)..(pos as usize + minimizer_size))
                        .to_word();
                    dict.insert(shard as u64);
                }
            });
    }
    Ok(dict)
}

fn process_query_streaming<P: AsRef<Path>>(
    query_path: P,
    kmer_size: usize,
    minimizer_size: usize,
    kmer_threshold: usize,
    ref_kmer_dict: Arc<HashSet<u64>>,
    ref_min_dict: Arc<HashSet<u64>>,
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
    let match_n = Arc::new(
        RegexBuilder::new(r"[N]+")
            .case_insensitive(true)
            .unicode(false)
            .build()
            .unwrap(),
    );

    for _ in 0..num_consumers {
        let record_rx_clone = record_rx.clone();
        let result_tx_clone = result_tx.clone();
        let ref_min_dict_clone = Arc::clone(&ref_min_dict);
        let ref_kmer_dict_clone = Arc::clone(&ref_kmer_dict);
        let match_n_clone = Arc::clone(&match_n);

        let handle = thread::spawn(move || {
            while let Ok((id, seq)) = record_rx_clone.recv() {
                let mut shared_min_count = 0;
                let mut fwd_pos = Vec::new();
                match_n_clone
                    .split(&seq)
                    .filter(|&seq| seq.len() >= kmer_size)
                    .for_each(|seq| {
                        let packed_seq = PackedSeqVec::from_ascii(seq);
                        fwd_pos.clear();
                        simd_minimizers::minimizer_positions(
                            packed_seq.as_slice(),
                            minimizer_size,
                            window_size,
                            &mut fwd_pos,
                        );
                        for pos in &fwd_pos {
                            let minimizer = packed_seq
                                .slice((*pos as usize)..(*pos as usize + minimizer_size))
                                .to_word();
                            if ref_min_dict_clone.contains(&(minimizer as u64)) {
                                shared_min_count += 1;
                            }
                        }
                    });

                if shared_min_count < minimizer_threshold || seq.len() < kmer_size {
                    continue;
                }

                // let kmer_matches = nthash::nthash(&seq, kmer_size)
                //     .into_iter()
                //     .filter(|hash| ref_kmer_dict_clone.contains(hash))
                //     .count();
                let hasher = ahash::RandomState::with_seed(42);
                let kmer_matches = seq
                    .windows(kmer_size)
                    .map(|kmer| hasher.hash_one(kmer))
                    .filter(|hash| ref_kmer_dict_clone.contains(hash))
                    .count();

                if kmer_matches >= kmer_threshold {
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

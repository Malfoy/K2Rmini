use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::thread;

use ahash::AHashSet;
use clap::Parser;
use crossbeam_channel::unbounded;
use flate2::read::MultiGzDecoder;
use nthash::nthash;
use packed_seq::{PackedSeqVec, Seq, SeqVec};
use seq_io::fasta::Reader as FastaReader;
use seq_io::fasta::Record;
use zstd::stream::read::Decoder as ZstdDecoder;

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

fn main() -> io::Result<()> {
    let args = Args::parse();

    let reference_path = &args.reference;
    let query_path = &args.query;
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    assert!(minimizer_size <= kmer_size);
    let window_size: usize = kmer_size - minimizer_size + 1;
    let kmer_threshold: usize = args.threshold;
    let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

    eprintln!("Indexing reference k-mers...");
    let ref_kmer_dict = Arc::new(index_reference_kmers(reference_path, kmer_size)?);
    eprintln!(
        "Reference k-mer index contains {} entries.",
        ref_kmer_dict.len()
    );

    eprintln!("Indexing reference minimizers...");
    let ref_min_dict = Arc::new(index_reference_minimizers(
        reference_path,
        minimizer_size,
        window_size,
    )?);
    eprintln!(
        "Reference minimizer index contains {} entries.",
        ref_min_dict.len()
    );

    eprintln!("Processing query sequences using a producer–consumer model...");
    process_query_streaming(
        query_path,
        kmer_size,
        minimizer_size,
        window_size,
        minimizer_threshold,
        kmer_threshold,
        Arc::clone(&ref_kmer_dict),
        Arc::clone(&ref_min_dict),
    )?;

    Ok(())
}

fn index_reference_kmers<P: AsRef<Path>>(
    ref_path: P,
    kmer_size: usize,
) -> io::Result<AHashSet<u64>> {
    let mut dict = AHashSet::new();
    let file = File::open(ref_path)?;
    let mut reader = FastaReader::new(BufReader::new(file));

    while let Some(result) = reader.next() {
        let record = result.map_err(io::Error::other)?;
        let seq = record.seq();
        for hash in nthash(seq, kmer_size) {
            dict.insert(hash);
        }
    }
    Ok(dict)
}

fn index_reference_minimizers<P: AsRef<Path>>(
    ref_path: P,
    minimizer_size: usize,
    window_size: usize,
) -> io::Result<AHashSet<u64>> {
    let mut dict = AHashSet::new();
    let file = File::open(ref_path)?;
    let mut reader = FastaReader::new(BufReader::new(file));

    while let Some(result) = reader.next() {
        let record = result.map_err(io::Error::other)?;
        let seq = record.seq();
        // Pack the sequence for simd-minimizers
        let packed_seq = PackedSeqVec::from_ascii(seq);
        let mut fwd_pos = Vec::new();
        // Pass minimizer_size and window_size as usize
        simd_minimizers::minimizer_positions(
            packed_seq.as_slice(),
            minimizer_size,
            window_size,
            &mut fwd_pos,
        );
        for pos in fwd_pos {
            if (pos as usize + minimizer_size) <= seq.len() {
                let shard = packed_seq
                    .slice((pos as usize)..(pos as usize + minimizer_size))
                    .to_word();
                dict.insert(shard as u64);
            }
        }
    }
    Ok(dict)
}

fn process_query_streaming<P: AsRef<Path>>(
    query_path: P,
    kmer_size: usize,
    minimizer_size: usize,
    window_size: usize,
    minimizer_threshold: usize,
    kmer_threshold: usize,
    ref_kmer_dict: Arc<AHashSet<u64>>,
    ref_min_dict: Arc<AHashSet<u64>>,
) -> io::Result<()> {
    let reader = open_compressed_file(query_path)?;
    let mut parser = FastaReader::new(reader);

    let (record_tx, record_rx) = unbounded();
    let (result_tx, result_rx) = unbounded();

    let producer_handle = thread::spawn(move || {
        while let Some(result) = parser.next() {
            match result {
                Ok(record) => {
                    let id = record.id().unwrap_or("<unknown>").to_string();
                    let seq = record.seq().to_vec();
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
            while let Ok((id, seq)) = record_rx_clone.recv() {
                // Compute minimizers for the query sequence using simd-minimizers.
                let packed_seq = PackedSeqVec::from_ascii(&seq);
                let mut fwd_pos = Vec::new();
                simd_minimizers::minimizer_positions(
                    packed_seq.as_slice(),
                    minimizer_size,
                    window_size,
                    &mut fwd_pos,
                );

                let mut shared_min_count = 0;
                for pos in &fwd_pos {
                    if (*pos as usize + minimizer_size) <= seq.len() {
                        let minimizer = packed_seq
                            .slice((*pos as usize)..(*pos as usize + minimizer_size))
                            .to_word();
                        if ref_min_dict_clone.contains(&(minimizer as u64)) {
                            shared_min_count += 1;
                        }
                    }
                }

                if shared_min_count < minimizer_threshold || seq.len() < kmer_size {
                    continue;
                }

                let kmer_matches = nthash(&seq, kmer_size)
                    .into_iter()
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
            println!(">{}", id);
            println!("{}", String::from_utf8_lossy(&seq));
        }
    });

    producer_handle.join().expect("Producer thread panicked");
    for handle in consumer_handles {
        handle.join().expect("Consumer thread panicked");
    }
    printer_handle.join().expect("Printer thread panicked");

    Ok(())
}

fn open_compressed_file<P: AsRef<Path>>(path: P) -> io::Result<Box<dyn Read + Send>> {
    let file = File::open(&path)?;
    let buf_reader = BufReader::new(file);
    let extension = path
        .as_ref()
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    match extension {
        "gz" => Ok(Box::new(MultiGzDecoder::new(buf_reader))),
        "zst" => Ok(Box::new(ZstdDecoder::new(buf_reader)?)),
        _ => Ok(Box::new(buf_reader)),
    }
}

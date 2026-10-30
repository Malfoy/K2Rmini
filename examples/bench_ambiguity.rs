//! In-memory parsing + minimizer/k-mer kernel benchmark; no index or output I/O.
//! cargo run --release --example bench_ambiguity -- mask reads.fa 31 5
use helicase::*;
use k2rmini::{ambiguity, high_precision};
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::Seq;
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};
use std::ops::Range;
use std::time::Instant;

const MASK: Config = ParserOptions::default()
    .dna_string()
    .and_dna_packed()
    .keep_non_actg()
    .compute_mask_non_actg()
    .config();
const TEXT: Config = ParserOptions::default()
    .dna_string()
    .and_dna_packed()
    .keep_non_actg()
    .config();
const SPLIT: Config = ParserOptions::default()
    .dna_string()
    .and_dna_packed()
    .split_non_actg()
    .config();

fn text_spans(dna: &[u8], k: usize, out: &mut Vec<Range<usize>>) {
    out.clear();
    let mut start = 0;
    for (i, &base) in dna.iter().enumerate() {
        if !matches!(base.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T') {
            if i - start >= k {
                out.push(start..i);
            }
            start = i + 1;
        }
    }
    if dna.len() - start >= k {
        out.push(start..dna.len());
    }
}

fn run<const C: Config>(
    input: &[u8],
    threads: usize,
    k: usize,
    precision: bool,
    locate: impl Fn(&FastxParser<'_, C>, usize, &mut Vec<Range<usize>>) + Sync,
) -> (u64, u64, u64) {
    let totals = ParallelFastxParser::<C>::from_slice(input)
        .threads(threads)
        .for_each_with(
            || (0u64, 0u64, 0u64),
            |total, mut parser| {
                let hasher = NtHasher::<false>::new(k);
                let mini_hasher = NtHasher::<false>::new(21);
                let builder = minimizers(21, k - 21 + 1).hasher(&mini_hasher);
                let (mut spans, mut positions, mut hashes, mut keys) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                while let Some(event) = parser.next() {
                    event?;
                    let seq = parser.get_packed_seq();
                    locate(&parser, k, &mut spans);
                    for range in &spans {
                        let span = seq.slice(range.clone());
                        total.0 += (span.len() - k + 1) as u64;
                        positions.clear();
                        let out = builder.run(span, &mut positions);
                        total.1 = total.1.wrapping_add(out.values_u64().sum::<u64>());
                        if precision {
                            high_precision::kmer_keys_64(span, k, &mut keys);
                            for &hash in &keys {
                                total.2 = total.2.wrapping_add(hash);
                            }
                        } else {
                            hashes.clear();
                            hasher.hash_kmers_simd(span, 1).collect_into(&mut hashes);
                            total.2 = total
                                .2
                                .wrapping_add(hashes.iter().map(|&x| u64::from(x)).sum::<u64>());
                        }
                    }
                }
                Ok(())
            },
        )
        .unwrap();
    totals.into_iter().fold((0, 0, 0), |a, b| {
        (a.0 + b.0, a.1.wrapping_add(b.1), a.2.wrapping_add(b.2))
    })
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let data = std::fs::read(&args[2]).unwrap();
    let k = args.get(3).map_or(31, |x| x.parse().unwrap());
    let repeats = args.get(4).map_or(5, |x| x.parse().unwrap());
    let precision = args.get(5).is_some_and(|x| x == "high");
    let threads = std::thread::available_parallelism().unwrap().get();
    for round in 0..=repeats {
        let start = Instant::now();
        let total = match args[1].as_str() {
            "mask" => run::<MASK>(&data, threads, k, precision, |p, k, spans| {
                ambiguity::valid_spans(p.get_mask_non_actg(), k, spans)
            }),
            "text" => run::<TEXT>(&data, threads, k, precision, |p, k, spans| {
                text_spans(p.get_dna_string(), k, spans)
            }),
            "split" => run::<SPLIT>(&data, threads, k, precision, |p, k, spans| {
                spans.clear();
                let len = p.get_packed_seq().len();
                if len >= k {
                    spans.push(0..len);
                }
            }),
            _ => panic!("expected mask, text, or split"),
        };
        if round > 0 {
            println!(
                "{},{threads},{round},{:.9},{},{},{}",
                args[1],
                start.elapsed().as_secs_f64(),
                total.0,
                total.1,
                total.2
            );
        }
        std::hint::black_box(total);
    }
}

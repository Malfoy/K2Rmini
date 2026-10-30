use crate::ambiguity::valid_spans;
use crate::cli::{MAX_QUERIES, Options, Threshold};
use crate::high_precision::kmer_keys_64;
use crate::index::{FilterIndex, QueryMask, SparseMembership};
use crossbeam_channel::bounded;
use helicase::*;
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::{PackedSeq, Seq};
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::Range;
use std::thread;

const CONFIG_FILTER: Config = ParserOptions::default()
    .dna_string()
    .and_dna_packed()
    .keep_non_actg()
    .compute_mask_non_actg()
    .config();
const BATCH_SIZE: usize = 64 * 1024;

fn query_ids(mut mask: QueryMask) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let id = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(id)
    })
}

fn count_matches(
    mask: QueryMask,
    counts: &mut [usize],
    thresholds: &[usize],
    needed: &mut QueryMask,
) {
    for id in query_ids(mask & *needed) {
        counts[id] += 1;
        if counts[id] >= thresholds[id] {
            *needed &= !(1 << id);
        }
    }
}

struct Candidate {
    minimizer: u64,
    start: usize,
    end: usize,
    queries: QueryMask,
}

struct Worker {
    mini_hasher: NtHasher<false>,
    hasher: NtHasher<false>,
    spans: Vec<Range<usize>>,
    positions: Vec<u32>,
    super_kmers: Vec<u32>,
    candidates: Vec<Candidate>,
    hashes: Vec<u32>,
    keys: Vec<u64>,
}

impl Worker {
    fn new(args: &Options) -> Self {
        Self {
            mini_hasher: NtHasher::new(args.m),
            hasher: NtHasher::new(args.k),
            spans: Vec::new(),
            positions: Vec::new(),
            super_kmers: Vec::new(),
            candidates: Vec::new(),
            hashes: Vec::new(),
            keys: Vec::new(),
        }
    }

    fn matches<I: FilterIndex>(
        &mut self,
        seq: PackedSeq<'_>,
        mask: &dna_format::BitMask,
        args: &Options,
        thresholds: &[Threshold],
        index: &I,
    ) -> bool {
        if seq.len() < args.k {
            return false;
        }
        let window = args.k - args.m + 1;
        let builder = minimizers(args.m, window).hasher(&self.mini_hasher);
        let mut required = [0; MAX_QUERIES];
        let mut mini_required = [0; MAX_QUERIES];
        for (id, &threshold) in thresholds.iter().enumerate() {
            required[id] = threshold.required(seq.len() - args.k + 1, index.counts()[id]);
            mini_required[id] = required[id].div_ceil(window);
        }
        let mut needed = QueryMask::MAX >> (MAX_QUERIES - thresholds.len());
        valid_spans(mask, args.k, &mut self.spans);
        let mut mini_counts = [0; MAX_QUERIES];
        if args.approx {
            if thresholds.len() == 1 && self.spans.len() == 1 {
                self.positions.clear();
                let out = builder.run(seq.slice(self.spans[0].clone()), &mut self.positions);
                if let Some(result) =
                    index.single_minimizer_threshold(out.values_u64(), mini_required[0])
                {
                    return result;
                }
            }
            for range in &self.spans {
                self.positions.clear();
                let out = builder.run(seq.slice(range.clone()), &mut self.positions);
                for value in out.values_u64() {
                    count_matches(
                        index.minimizer_mask(value),
                        &mut mini_counts,
                        &mini_required,
                        &mut needed,
                    );
                    if needed == 0 {
                        return true;
                    }
                }
            }
            return false;
        }

        self.candidates.clear();
        let mut possible = [0; MAX_QUERIES];
        for range in &self.spans {
            self.positions.clear();
            self.super_kmers.clear();
            let span = seq.slice(range.clone());
            let out = builder
                .super_kmers(&mut self.super_kmers)
                .run(span, &mut self.positions);
            self.super_kmers.push((span.len() - args.k + 1) as u32);
            for (value, bounds) in out.values_u64().zip(self.super_kmers.windows(2)) {
                let queries = index.minimizer_mask(value);
                if queries == 0 {
                    continue;
                }
                let start = range.start + bounds[0] as usize;
                let end = range.start + bounds[1] as usize;
                for id in query_ids(queries) {
                    mini_counts[id] += 1;
                    possible[id] += end - start;
                }
                self.candidates.push(Candidate {
                    minimizer: value,
                    start,
                    end,
                    queries,
                });
            }
        }
        if (0..thresholds.len())
            .any(|id| mini_counts[id] < mini_required[id] || possible[id] < required[id])
        {
            return false;
        }

        if args.high_precision {
            kmer_keys_64(seq, args.k, &mut self.keys);
        } else {
            self.hashes.clear();
            self.hasher
                .hash_kmers_simd(seq, 1)
                .collect_into(&mut self.hashes);
        }
        let mut counts = [0; MAX_QUERIES];
        for candidate in &self.candidates {
            if candidate.queries & needed == 0 {
                continue;
            }
            if args.high_precision {
                for &key in &self.keys[candidate.start..candidate.end] {
                    // The minimizer must belong to the SAME query as the key.
                    count_matches(
                        index.high_mask(key) & candidate.queries,
                        &mut counts,
                        &required,
                        &mut needed,
                    );
                    if needed == 0 {
                        return true;
                    }
                }
            } else {
                let page = index.sparse_page(candidate.minimizer);
                for &hash in &self.hashes[candidate.start..candidate.end] {
                    count_matches(
                        page.mask(hash) & candidate.queries,
                        &mut counts,
                        &required,
                        &mut needed,
                    );
                    if needed == 0 {
                        return true;
                    }
                }
            }
            for id in query_ids(candidate.queries & needed) {
                possible[id] -= candidate.end - candidate.start;
                if counts[id] + possible[id] < required[id] {
                    return false;
                }
            }
        }
        false
    }
}

pub(crate) fn process_query_streaming<I: FilterIndex>(
    file: &str,
    args: &Options,
    thresholds: &[Threshold],
    index: &I,
) -> io::Result<()> {
    let parser = ParallelFastxParser::<CONFIG_FILTER>::from_file(file)
        .map_err(|err| io::Error::other(err.to_string()))?;
    let destination: Box<dyn Write + Send> = if let Some(path) = &args.output {
        Box::new(File::create(path)?)
    } else {
        Box::new(io::stdout())
    };
    let (tx, rx) = bounded::<Vec<u8>>(4 * args.num_threads());
    let printer = thread::spawn(move || -> io::Result<()> {
        let mut writer = BufWriter::new(destination);
        for chunk in rx {
            writer.write_all(&chunk)?;
        }
        writer.flush()
    });
    let processing = parser.threads(args.num_threads()).for_each(|mut parser| {
        let mut worker = Worker::new(args);
        let mut output = Vec::with_capacity(BATCH_SIZE + 4096);
        while let Some(event) = parser.next() {
            event?;
            if !worker.matches(
                parser.get_packed_seq(),
                parser.get_mask_non_actg(),
                args,
                thresholds,
                index,
            ) {
                continue;
            }
            output.push(b'>');
            output.extend_from_slice(parser.get_header());
            output.push(b'\n');
            output.extend_from_slice(parser.get_dna_string());
            output.push(b'\n');
            if output.len() >= BATCH_SIZE {
                let mut batch = Vec::with_capacity(BATCH_SIZE + 4096);
                std::mem::swap(&mut output, &mut batch);
                if tx.send(batch).is_err() {
                    break;
                }
            }
        }
        if !output.is_empty() {
            let _ = tx.send(output);
        }
        Ok(())
    });
    drop(tx);
    printer
        .join()
        .map_err(|_| io::Error::other("Printer thread panicked"))??;
    processing.map_err(|err| io::Error::other(err.to_string()))
}

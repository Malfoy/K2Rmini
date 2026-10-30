use crate::ambiguity::valid_spans;
use crate::cli::Options;
use crate::high_precision::kmer_keys_64;
use crate::sharded_hashset::{ConcurrentShardedSet, ShardedHashSet};
use crate::sparse_bitset::{ConcurrentBlockSparseBitSet, WORDS_PER_PAGE, minimizer_page};
use helicase::*;
use rustc_hash::{FxHashMap, FxHashSet};
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::Seq;
use simd_minimizers::seq_hash::{KmerHasher, NtHasher};
use std::hash::Hash;
use std::io;
use std::sync::{Mutex, atomic::AtomicU64};

pub(crate) const CONFIG_INDEX: Config = ParserOptions::default()
    .ignore_headers()
    .dna_packed()
    .keep_non_actg()
    .compute_mask_non_actg()
    .config();
type MinIndex = ShardedHashSet;
pub(crate) type QueryMask = u32;

enum KmerIndex {
    Sparse(ConcurrentBlockSparseBitSet),
    HighPrecision(FxHashSet<u64>),
}

impl KmerIndex {
    fn len(&self) -> usize {
        match self {
            Self::Sparse(index) => index.len(),
            Self::HighPrecision(index) => index.len(),
        }
    }
}

pub(crate) struct SingleIndex {
    minimizers: MinIndex,
    kmers: KmerIndex,
    counts: [usize; 1],
}

impl SingleIndex {
    fn new(minimizers: MinIndex, kmers: KmerIndex) -> Self {
        let counts = [kmers.len()];
        Self {
            minimizers,
            kmers,
            counts,
        }
    }
}

pub(crate) fn index_reference(patterns: &str, args: &Options) -> io::Result<SingleIndex> {
    if args.high_precision {
        return index_reference_high_precision(patterns, args);
    }
    const BATCH_KMERS: usize = 2_000_000;

    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let window_span_bp = window_size + kmer_size - 1;
    let num_threads = args.num_threads();
    let is_approx = args.approx;

    let kmer_dict = ConcurrentBlockSparseBitSet::new();
    let mini_dict = ConcurrentShardedSet::new();

    ParallelFastxParser::<CONFIG_INDEX>::from_file(patterns)
        .map_err(|err| io::Error::other(err.to_string()))?
        .threads(num_threads)
        .for_each(|mut parser| {
            let mini_hasher = NtHasher::<false>::new(minimizer_size);
            let mini_builder = minimizers(minimizer_size, window_size).hasher(&mini_hasher);
            let hasher = NtHasher::<false>::new(kmer_size);
            let mut local_mini = mini_dict.local_writer();
            let mut mini_pos = Vec::new();
            let mut spans = Vec::new();
            let mut super_kmers = Vec::new();
            let mut kmer_hashes = Vec::new();

            while let Some(event) = parser.next() {
                event?;
                let record = parser.get_packed_seq();
                valid_spans(parser.get_mask_non_actg(), args.k, &mut spans);
                for range in &spans {
                    let packed_seq = record.slice(range.clone());
                    let num_kmers = packed_seq.len().saturating_sub(kmer_size - 1);
                    if num_kmers == 0 {
                        continue;
                    }

                    if num_kmers <= BATCH_KMERS {
                        mini_pos.clear();
                        if is_approx {
                            let out = mini_builder.run(packed_seq, &mut mini_pos);
                            local_mini.extend(out.values_u64());
                        } else {
                            super_kmers.clear();
                            kmer_hashes.clear();
                            let out = mini_builder
                                .super_kmers(&mut super_kmers)
                                .run(packed_seq, &mut mini_pos);
                            super_kmers.push(num_kmers as u32);
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
                        }
                    } else {
                        for k_start in (0..num_kmers).step_by(BATCH_KMERS) {
                            let k_len = (num_kmers - k_start).min(BATCH_KMERS);
                            let chunk_end =
                                (k_start + k_len - 1 + window_span_bp).min(packed_seq.len());
                            let sub_seq = packed_seq.slice(k_start..chunk_end);
                            let sub_num_kmers = sub_seq.len().saturating_sub(kmer_size - 1);

                            mini_pos.clear();
                            if is_approx {
                                let out = mini_builder.run(sub_seq, &mut mini_pos);
                                local_mini.extend(out.values_u64());
                            } else {
                                super_kmers.clear();
                                kmer_hashes.clear();
                                let out = mini_builder
                                    .super_kmers(&mut super_kmers)
                                    .run(sub_seq, &mut mini_pos);
                                super_kmers.push(sub_num_kmers as u32);
                                hasher
                                    .hash_kmers_simd(sub_seq, 1)
                                    .collect_into(&mut kmer_hashes);
                                for (min_val, bounds) in
                                    out.values_u64().zip(super_kmers.windows(2))
                                {
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
                }
            }
            Ok(())
        })
        .map_err(|err| io::Error::other(err.to_string()))?;

    Ok(SingleIndex::new(
        mini_dict.freeze(),
        KmerIndex::Sparse(kmer_dict),
    ))
}

fn index_reference_high_precision(patterns: &str, args: &Options) -> io::Result<SingleIndex> {
    // Bound per-worker buffers even when a reference contains very long records.
    const BATCH_KMERS: usize = 65_536;
    let num_threads = args.num_threads();
    let mini_dict = ConcurrentShardedSet::new();
    let kmer_dict = Mutex::new(FxHashSet::default());

    ParallelFastxParser::<CONFIG_INDEX>::from_file(patterns)
        .map_err(|err| io::Error::other(err.to_string()))?
        .threads(num_threads)
        .for_each(|mut parser| {
            let mini_hasher = NtHasher::<false>::new(args.m);
            let mini_builder = minimizers(args.m, args.k - args.m + 1).hasher(&mini_hasher);
            let mut local_mini = mini_dict.local_writer();
            let mut local_kmers = FxHashSet::default();
            let mut mini_pos = Vec::new();
            let mut spans = Vec::new();
            let mut keys = Vec::new();

            while let Some(event) = parser.next() {
                event?;
                let record = parser.get_packed_seq();
                valid_spans(parser.get_mask_non_actg(), args.k, &mut spans);
                for range in &spans {
                    let seq = record.slice(range.clone());
                    let num_kmers = seq.len().saturating_sub(args.k - 1);
                    for start in (0..num_kmers).step_by(BATCH_KMERS) {
                        let count = (num_kmers - start).min(BATCH_KMERS);
                        // Overlap by k-1 bases so every k-mer is indexed exactly once.
                        let chunk = seq.slice(start..start + count + args.k - 1);
                        mini_pos.clear();
                        let out = mini_builder.run(chunk, &mut mini_pos);
                        local_mini.extend(out.values_u64());
                        kmer_keys_64(chunk, args.k, &mut keys);
                        local_kmers.extend(keys.iter().copied());
                        if local_kmers.len() >= BATCH_KMERS {
                            kmer_dict.lock().unwrap().extend(local_kmers.drain());
                        }
                    }
                }
            }
            kmer_dict.lock().unwrap().extend(local_kmers.drain());
            Ok(())
        })
        .map_err(|err| io::Error::other(err.to_string()))?;

    Ok(SingleIndex::new(
        mini_dict.freeze(),
        KmerIndex::HighPrecision(kmer_dict.into_inner().unwrap()),
    ))
}

/// One logical membership index, specialized at compile time for one or many queries.
pub(crate) trait FilterIndex: Sync {
    type Page<'a>: SparseMembership
    where
        Self: 'a;
    fn counts(&self) -> &[usize];
    fn minimizer_mask(&self, value: u64) -> QueryMask;
    fn sparse_page(&self, minimizer: u64) -> Self::Page<'_>;
    fn high_mask(&self, key: u64) -> QueryMask;
    /// Optional optimized early-accept/reject path for a single constraint.
    fn single_minimizer_threshold(
        &self,
        _values: impl ExactSizeIterator<Item = u64>,
        _threshold: usize,
    ) -> Option<bool> {
        None
    }
    fn minimizer_len(&self) -> usize;
    fn kmer_len(&self) -> usize;
}

pub(crate) trait SparseMembership {
    fn mask(&self, hash: u32) -> QueryMask;
}

pub(crate) struct SinglePage<'a>(Option<&'a [AtomicU64; WORDS_PER_PAGE]>);
impl SparseMembership for SinglePage<'_> {
    #[inline(always)]
    fn mask(&self, hash: u32) -> QueryMask {
        u32::from(
            self.0
                .is_some_and(|page| ConcurrentBlockSparseBitSet::page_contains(page, hash)),
        )
    }
}

impl FilterIndex for SingleIndex {
    type Page<'a> = SinglePage<'a>;
    fn counts(&self) -> &[usize] {
        &self.counts
    }
    fn single_minimizer_threshold(
        &self,
        values: impl ExactSizeIterator<Item = u64>,
        threshold: usize,
    ) -> Option<bool> {
        Some(self.minimizers.matches_threshold(values, threshold))
    }
    #[inline(always)]
    fn minimizer_mask(&self, value: u64) -> QueryMask {
        u32::from(self.minimizers.contains(&value))
    }
    fn sparse_page(&self, minimizer: u64) -> SinglePage<'_> {
        SinglePage(match &self.kmers {
            KmerIndex::Sparse(index) => index.get_page(minimizer),
            KmerIndex::HighPrecision(_) => None,
        })
    }
    #[inline(always)]
    fn high_mask(&self, key: u64) -> QueryMask {
        match &self.kmers {
            KmerIndex::HighPrecision(index) => u32::from(index.contains(&key)),
            KmerIndex::Sparse(_) => 0,
        }
    }
    fn minimizer_len(&self) -> usize {
        self.minimizers.len()
    }
    fn kmer_len(&self) -> usize {
        self.counts[0]
    }
}

/// A globally shared map, partitioned like the single-query minimizer set.
struct MaskMap<K> {
    shards: Box<[FxHashMap<K, QueryMask>]>,
}

impl<K: Copy + Eq + Hash + Into<u64>> MaskMap<K> {
    fn new() -> Self {
        Self {
            shards: (0..crate::sharded_hashset::NUM_SHARDS)
                .map(|_| FxHashMap::default())
                .collect(),
        }
    }
    fn insert(&mut self, key: K, bit: QueryMask) {
        *self.shards[crate::sharded_hashset::shard_index(key.into())]
            .entry(key)
            .or_default() |= bit;
    }
    #[inline(always)]
    fn mask(&self, key: K) -> QueryMask {
        self.shards[crate::sharded_hashset::shard_index(key.into())]
            .get(&key)
            .copied()
            .unwrap_or(0)
    }
    fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.len()).sum()
    }
}

enum KmerMasks {
    Sparse(MaskMap<u32>),
    HighPrecision(MaskMap<u64>),
    Approximate,
}

pub(crate) struct MultiIndex {
    minimizers: MaskMap<u64>,
    kmers: KmerMasks,
    counts: Vec<usize>,
}

impl MultiIndex {
    pub(crate) fn new(options: &Options) -> Self {
        Self {
            minimizers: MaskMap::new(),
            counts: Vec::new(),
            kmers: if options.approx {
                KmerMasks::Approximate
            } else if options.high_precision {
                KmerMasks::HighPrecision(MaskMap::new())
            } else {
                KmerMasks::Sparse(MaskMap::new())
            },
        }
    }

    /// Build each reference with exactly the same parallel indexer as K2Rmini,
    /// then attach its query bit. Only one temporary single index is held at once.
    pub(crate) fn add(&mut self, reference: SingleIndex) {
        let bit = 1u32 << self.counts.len();
        self.counts.push(reference.counts[0]);
        for minimizer in reference.minimizers.into_values() {
            self.minimizers.insert(minimizer, bit);
        }
        match (&mut self.kmers, reference.kmers) {
            (KmerMasks::Sparse(merged), KmerIndex::Sparse(index)) => {
                for address in index.addresses() {
                    merged.insert(address, bit);
                }
            }
            (KmerMasks::HighPrecision(merged), KmerIndex::HighPrecision(index)) => {
                for key in index {
                    merged.insert(key, bit);
                }
            }
            (KmerMasks::Approximate, _) => {}
            _ => unreachable!("Reference and merged index modes must agree"),
        }
    }
}

pub(crate) struct MultiPage<'a> {
    prefix: u32,
    map: Option<&'a MaskMap<u32>>,
}

impl SparseMembership for MultiPage<'_> {
    #[inline(always)]
    fn mask(&self, hash: u32) -> QueryMask {
        self.map
            .map_or(0, |map| map.mask(self.prefix | (hash & 0xffff)))
    }
}

impl FilterIndex for MultiIndex {
    type Page<'a> = MultiPage<'a>;
    fn counts(&self) -> &[usize] {
        &self.counts
    }
    #[inline(always)]
    fn minimizer_mask(&self, value: u64) -> QueryMask {
        self.minimizers.mask(value)
    }
    fn sparse_page(&self, minimizer: u64) -> MultiPage<'_> {
        MultiPage {
            prefix: (minimizer_page(minimizer) as u32) << 16,
            map: match &self.kmers {
                KmerMasks::Sparse(map) => Some(map),
                _ => None,
            },
        }
    }
    #[inline(always)]
    fn high_mask(&self, key: u64) -> QueryMask {
        match &self.kmers {
            KmerMasks::HighPrecision(map) => map.mask(key),
            _ => 0,
        }
    }
    fn minimizer_len(&self) -> usize {
        self.minimizers.len()
    }
    fn kmer_len(&self) -> usize {
        match &self.kmers {
            KmerMasks::Sparse(map) => map.len(),
            KmerMasks::HighPrecision(map) => map.len(),
            KmerMasks::Approximate => 0,
        }
    }
}

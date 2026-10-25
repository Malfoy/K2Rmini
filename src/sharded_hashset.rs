use rustc_hash::FxHashSet;
use std::sync::{Arc, Mutex};

pub const NUM_SHARDS: usize = 64;

/// Partition hash function mapping a 64-bit value to a shard index `[0, NUM_SHARDS)`.
#[inline(always)]
pub fn shard_index(val: u64) -> usize {
    (val.wrapping_mul(0x517c_c1b7_2722_0a95) >> 58) as usize
}

/// An immutable sharded hash set for high-throughput concurrent lookups.
///
/// Lookups are partitioned across 64 independent `FxHashSet<u64>` sub-tables.
/// This significantly improves CPU cache locality during query filtering
/// because individual shards are much smaller than a monolithic 100M+ entry table.
pub struct ShardedHashSet {
    shards: Box<[FxHashSet<u64>]>,
}

impl Default for ShardedHashSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardedHashSet {
    pub fn new() -> Self {
        Self {
            shards: (0..NUM_SHARDS)
                .map(|_| FxHashSet::default())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn insert(&mut self, val: u64) -> bool {
        self.shards[shard_index(val)].insert(val)
    }

    #[inline(always)]
    pub fn contains(&self, val: &u64) -> bool {
        self.shards[shard_index(*val)].contains(val)
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }
}

/// Backwards-compatible alias for minimizer indexing.
#[allow(dead_code)]
pub type ShardedMinIndex = ShardedHashSet;

/// A concurrent builder for constructing a `ShardedHashSet` in parallel without
/// a costly serial single-threaded merge phase.
#[derive(Clone)]
pub struct ConcurrentShardedSet {
    shards: Arc<Vec<Mutex<FxHashSet<u64>>>>,
}

impl Default for ConcurrentShardedSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ConcurrentShardedSet {
    pub fn new() -> Self {
        Self {
            shards: Arc::new(
                (0..NUM_SHARDS)
                    .map(|_| Mutex::new(FxHashSet::default()))
                    .collect(),
            ),
        }
    }

    /// Creates a thread-local writer with batch buffering (512 entries/shard).
    /// Batches amortize mutex acquisition overhead to virtually zero.
    pub fn local_writer(&self) -> LocalWriter<'_> {
        LocalWriter {
            shared: self,
            buffers: core::array::from_fn(|_| Vec::with_capacity(512)),
        }
    }

    /// Consumes the concurrent builder and freezes it into an immutable `ShardedHashSet`.
    /// Each shard reserves extra capacity to guarantee a `< 50%` load factor for
    /// single-probe negative lookups.
    pub fn freeze(self) -> ShardedHashSet {
        let shards = Arc::try_unwrap(self.shards)
            .ok()
            .expect("All concurrent writers must be dropped before freeze")
            .into_iter()
            .map(|m| {
                let mut s = m.into_inner().unwrap();
                s.reserve(s.len() / 2);
                s
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        ShardedHashSet { shards }
    }
}

/// Thread-local writer that batches insertions before acquiring shard mutexes.
pub struct LocalWriter<'a> {
    shared: &'a ConcurrentShardedSet,
    buffers: [Vec<u64>; NUM_SHARDS],
}

impl LocalWriter<'_> {
    #[inline(always)]
    pub fn insert(&mut self, val: u64) {
        let shard = shard_index(val);
        let buf = &mut self.buffers[shard];
        buf.push(val);
        if buf.len() >= 512 {
            self.flush_shard(shard);
        }
    }

    #[cold]
    fn flush_shard(&mut self, shard: usize) {
        let buf = &mut self.buffers[shard];
        if !buf.is_empty() {
            let mut guard = self.shared.shards[shard].lock().unwrap();
            guard.extend(buf.drain(..));
        }
    }

    pub fn flush(&mut self) {
        for shard in 0..NUM_SHARDS {
            self.flush_shard(shard);
        }
    }
}

impl Drop for LocalWriter<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

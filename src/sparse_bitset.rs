use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

const NUM_PAGES: usize = 65_536;
pub(crate) const WORDS_PER_PAGE: usize = 1_024; // 1,024 * 64 = 65,536 bits = 8,192 bytes = 8 KB

/// A concurrent block-sparse bitset for 32-bit keys (`u32`).
///
/// Divides the $2^{32}$ key space into 65,536 pages of 8 KB each.
/// Pages are allocated lazily on demand when a key falls within that page.
///
/// Memory scaling:
/// - Initial: 65,536 * 8 bytes = 512 KB (directory of pointers).
/// - Dynamically adds 8 KB per touched page.
/// - Hard ceiling: <= 512.5 MB when completely saturated.
pub struct ConcurrentBlockSparseBitSet {
    pages: Box<[AtomicPtr<[AtomicU64; WORDS_PER_PAGE]>]>,
}

unsafe impl Send for ConcurrentBlockSparseBitSet {}
unsafe impl Sync for ConcurrentBlockSparseBitSet {}

impl Default for ConcurrentBlockSparseBitSet {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
pub fn minimizer_page(minimizer: u64) -> usize {
    ((minimizer.wrapping_mul(0x517c_c1b7_2722_0a95)) as usize) & (NUM_PAGES - 1)
}

impl ConcurrentBlockSparseBitSet {
    pub fn new() -> Self {
        let mut pages = Vec::with_capacity(NUM_PAGES);
        for _ in 0..NUM_PAGES {
            pages.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        Self {
            pages: pages.into_boxed_slice(),
        }
    }

    #[cold]
    fn alloc_page(&self, page_idx: usize) -> *mut [AtomicU64; WORDS_PER_PAGE] {
        let new_page = Box::into_raw(Box::new(core::array::from_fn(|_| AtomicU64::new(0))));
        match self.pages[page_idx].compare_exchange(
            std::ptr::null_mut(),
            new_page,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => new_page,
            Err(actual) => {
                unsafe { drop(Box::from_raw(new_page)) };
                actual
            }
        }
    }

    /// Inserts a slice of k-mer hashes into the page determined by their shared minimizer.
    #[inline(always)]
    pub fn insert_batch(&self, minimizer: u64, kmer_hashes: &[u32]) {
        let page_idx = minimizer_page(minimizer);
        let mut ptr = self.pages[page_idx].load(Ordering::Acquire);
        if ptr.is_null() {
            ptr = self.alloc_page(page_idx);
        }
        let page = unsafe { &*ptr };
        for &kmer_hash in kmer_hashes {
            let word_idx = ((kmer_hash >> 6) as usize) & (WORDS_PER_PAGE - 1);
            let bit_mask = 1u64 << (kmer_hash & 63);
            let word = &page[word_idx];
            if (word.load(Ordering::Relaxed) & bit_mask) == 0 {
                word.fetch_or(bit_mask, Ordering::Relaxed);
            }
        }
    }

    /// Retrieves the page associated with `minimizer`, if allocated.
    #[inline(always)]
    pub fn get_page(&self, minimizer: u64) -> Option<&[AtomicU64; WORDS_PER_PAGE]> {
        let page_idx = minimizer_page(minimizer);
        let ptr = self.pages[page_idx].load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(&*ptr) }
        }
    }

    /// Tests if a k-mer hash is present inside a loaded page.
    #[inline(always)]
    pub fn page_contains(page: &[AtomicU64; WORDS_PER_PAGE], kmer_hash: u32) -> bool {
        let word_idx = ((kmer_hash >> 6) as usize) & (WORDS_PER_PAGE - 1);
        let bit_mask = 1u64 << (kmer_hash & 63);
        (page[word_idx].load(Ordering::Relaxed) & bit_mask) != 0
    }

    /// Iterates occupied composite addresses after indexing has finished.
    pub(crate) fn addresses(&self) -> impl Iterator<Item = u32> + '_ {
        self.pages
            .iter()
            .enumerate()
            .filter_map(|(page_idx, ptr)| {
                // Pages remain allocated until self is dropped. The returned borrow
                // cannot outlive self, and words are read atomically.
                unsafe { ptr.load(Ordering::Acquire).as_ref() }.map(|page| (page_idx, page))
            })
            .flat_map(|(page_idx, page)| {
                page.iter().enumerate().flat_map(move |(word_idx, word)| {
                    let mut bits = word.load(Ordering::Relaxed);
                    std::iter::from_fn(move || {
                        if bits == 0 {
                            return None;
                        }
                        let offset = word_idx * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        Some(((page_idx as u32) << 16) | offset as u32)
                    })
                })
            })
    }

    /// Returns the total number of set bits (cardinality).
    pub fn len(&self) -> usize {
        let mut total = 0;
        for atomic_ptr in self.pages.iter() {
            let ptr = atomic_ptr.load(Ordering::Relaxed);
            if !ptr.is_null() {
                unsafe {
                    for word in (*ptr).iter() {
                        total += word.load(Ordering::Relaxed).count_ones() as usize;
                    }
                }
            }
        }
        total
    }
}

impl Drop for ConcurrentBlockSparseBitSet {
    fn drop(&mut self) {
        for atomic_ptr in self.pages.iter_mut() {
            let ptr = *atomic_ptr.get_mut();
            if !ptr.is_null() {
                unsafe { drop(Box::from_raw(ptr)) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn exported_addresses_preserve_page_and_low_hash_bits() {
        let index = ConcurrentBlockSparseBitSet::new();
        let batches: &[(u64, &[u32])] = &[
            (0, &[0, 63, 64, 65_535, 65_536, u32::MAX]),
            (1, &[0, 63, 64, 65_535]),
            (u64::MAX, &[0, 64, u32::MAX]),
        ];
        let mut expected = HashSet::new();
        for &(minimizer, hashes) in batches {
            index.insert_batch(minimizer, hashes);
            for &hash in hashes {
                expected.insert(((minimizer_page(minimizer) as u32) << 16) | (hash & 0xffff));
            }
        }
        let actual: HashSet<_> = index.addresses().collect();
        assert_eq!(actual, expected);
        assert_eq!(index.len(), expected.len());
    }
}

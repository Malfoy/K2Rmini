use simd_minimizers::packed_seq::Seq;
use std::hash::{DefaultHasher, Hasher};

/// Produce global membership keys for forward k-mers: complete packed bases for
/// k <= 32, or a 64-bit hash of all packed blocks for larger k.
/// This deliberately does not widen or reuse the existing 32-bit ntHash values.
pub fn kmer_keys_64<'s>(seq: impl Seq<'s>, k: usize, out: &mut Vec<u64>) {
    assert!(k > 0);
    out.clear();
    if seq.len() < k {
        return;
    }
    if k <= 32 {
        // Two bits per base: retain every base, including all 64 bits at k=32.
        out.extend((0..=seq.len() - k).map(|pos| seq.read_kmer(k, pos)));
        return;
    }
    out.extend((0..=seq.len() - k).map(|pos| {
        let mut hasher = DefaultHasher::new();
        for offset in (0..k).step_by(32) {
            hasher.write_u64(seq.read_kmer((k - offset).min(32), pos + offset));
        }
        hasher.finish()
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use simd_minimizers::packed_seq::{PackedSeqVec, SeqVec};
    use simd_minimizers::seq_hash::{KmerHasher, NtHasher};

    #[test]
    fn short_kmer_keys_preserve_every_base_without_hashing() {
        for k in [1, 21, 31, 32] {
            let packed = PackedSeqVec::from_ascii(&vec![b'A'; k]);
            let mut keys = Vec::new();
            kmer_keys_64(packed.as_slice(), k, &mut keys);
            assert_eq!(keys, [0]);

            // Every position must retain its two bits, even the top bits at k=32.
            for pos in 0..k {
                let mut dna = vec![b'A'; k];
                dna[pos] = b'G';
                let packed = PackedSeqVec::from_ascii(&dna);
                kmer_keys_64(packed.as_slice(), k, &mut keys);
                assert_eq!(keys, [3u64 << (2 * pos)], "k={k}, pos={pos}");
            }
        }
        let packed = PackedSeqVec::from_ascii(&[b'G'; 32]);
        let mut keys = Vec::new();
        kmer_keys_64(packed.as_slice(), 32, &mut keys);
        assert_eq!(keys, [u64::MAX]);
    }

    #[test]
    fn distinguishes_known_32_bit_collision() {
        let a = PackedSeqVec::from_ascii(b"ATTGACTGTAATATCCGGATTATCAATATAT");
        let b = PackedSeqVec::from_ascii(b"ATTTGGAAACTGCGATAAATCATAGTACCAA");
        let old = NtHasher::<false>::new(31);
        assert_eq!(old.hash_seq(a.as_slice()), old.hash_seq(b.as_slice()));
        let (mut hashes_a, mut hashes_b) = (Vec::new(), Vec::new());
        kmer_keys_64(a.as_slice(), 31, &mut hashes_a);
        kmer_keys_64(b.as_slice(), 31, &mut hashes_b);
        assert_ne!(hashes_a, hashes_b);
        assert!(hashes_a[0] > u32::MAX as u64);
    }

    #[test]
    fn keys_are_independent_of_slice_alignment_and_include_all_blocks() {
        let dna = b"ACGTTGCAAGCTTAGCGATCCGTATGCAATGCTAGCTTACGATCGGTACAGTCCGATAGCTAACGTTACCGATGCTAGGCTAACGTTCAGATCGTACTGACCTAGCATGCA";
        let packed = PackedSeqVec::from_ascii(dna);
        for k in [1, 21, 31, 32, 33, 63, 64, 65, 96] {
            let mut hashes = Vec::new();
            kmer_keys_64(packed.as_slice(), k, &mut hashes);
            assert_eq!(hashes.len(), dna.len() - k + 1);
            for (pos, &hash) in hashes.iter().enumerate() {
                let isolated = PackedSeqVec::from_ascii(&dna[pos..pos + k]);
                let mut expected = Vec::new();
                kmer_keys_64(isolated.as_slice(), k, &mut expected);
                assert_eq!(hash, expected[0], "k={k}, pos={pos}");
            }
            let mut changed = dna[..k].to_vec();
            changed[k - 1] = if changed[k - 1] == b'A' { b'C' } else { b'A' };
            let changed = PackedSeqVec::from_ascii(&changed);
            let mut different = Vec::new();
            kmer_keys_64(changed.as_slice(), k, &mut different);
            assert_ne!(hashes[0], different[0], "k={k}");
        }
        let mut hashes = vec![42];
        kmer_keys_64(packed.slice(0..3), 31, &mut hashes);
        assert!(hashes.is_empty());
    }
}

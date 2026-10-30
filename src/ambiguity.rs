use helicase::dna_format::BitMask;
use std::ops::Range;

/// Locate A/C/G/T spans in the original coordinates without scanning DNA bytes.
/// Empty and short spans are discarded; ambiguous bases are never concatenated.
pub fn valid_spans(mask: &BitMask, k: usize, out: &mut Vec<Range<usize>>) {
    out.clear();
    let (words, tail) = mask.bits();
    let mut start = 0;
    for (word_idx, mut invalid) in words.iter().copied().chain([tail]).enumerate() {
        let base = word_idx * 64;
        while invalid != 0 {
            let first = invalid.trailing_zeros() as usize;
            let pos = base + first;
            if pos - start >= k {
                out.push(start..pos);
            }
            let end = first + (invalid >> first).trailing_ones() as usize;
            start = base + end;
            invalid = if end == 64 { 0 } else { invalid & (!0 << end) };
        }
    }
    if mask.len() - start >= k {
        out.push(start..mask.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_agree_with_a_byte_scan_at_word_boundaries() {
        for len in [0, 1, 31, 32, 63, 64, 65, 127, 128, 129, 512] {
            for step in [1, 2, 7, 31, 63, 64, 65, 1000] {
                let mut dna = vec![b'A'; len];
                let mut mask = BitMask::new();
                for (i, base) in dna.iter_mut().enumerate() {
                    let invalid = i % step == 0 || (60..70).contains(&i);
                    if invalid {
                        *base = b'N';
                    }
                    mask.append(u64::from(invalid), 1);
                }
                for k in [1, 5, 31, 32, 63] {
                    let mut actual = Vec::new();
                    valid_spans(&mask, k, &mut actual);
                    let mut pos = 0;
                    let expected: Vec<_> = dna
                        .split(|&b| b == b'N')
                        .filter_map(|span| {
                            let range = pos..pos + span.len();
                            pos += span.len() + 1;
                            (span.len() >= k).then_some(range)
                        })
                        .collect();
                    assert_eq!(actual, expected, "len={len}, step={step}, k={k}");
                }
            }
        }
    }
}

use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn run(patterns: &str, reads: &str, options: &[&str]) -> HashMap<String, String> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "k2rmini-ambiguity-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    fs::write(path.join("patterns.fa"), patterns).unwrap();
    fs::write(path.join("reads.fastx"), reads).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_K2Rmini"))
        .arg("-p")
        .arg(path.join("patterns.fa"))
        .args(options)
        .arg(path.join("reads.fastx"))
        .output()
        .unwrap();
    fs::remove_dir_all(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<_> = text.lines().collect();
    let mut result = HashMap::new();
    for record in lines.chunks_exact(2) {
        assert!(
            result
                .insert(
                    record[0].strip_prefix('>').unwrap().to_owned(),
                    record[1].to_owned()
                )
                .is_none(),
            "duplicate output read"
        );
    }
    result
}

#[test]
fn ambiguous_bases_cannot_become_real_bases_in_reads_or_references() {
    let gap = format!("{}N{}", "A".repeat(31), "C".repeat(31));
    let substitution = format!("{}G{}", "A".repeat(15), "C".repeat(15));
    let joined = format!("{}{}", "A".repeat(15), "C".repeat(16));
    for mode in ["", "--high-precision", "--approx"] {
        let mut opts = vec!["-t", "1", "-T", "3"];
        if !mode.is_empty() {
            opts.push(mode);
        }
        for ambiguous in [
            gap.clone(),
            gap.replace('N', "n"),
            gap.replace('N', "R"),
            gap.replace('N', "Y"),
        ] {
            for query in [&substitution, &joined] {
                assert!(
                    run(
                        &format!(">query\n{query}\n"),
                        &format!(">read\n{ambiguous}\n"),
                        &opts
                    )
                    .is_empty(),
                    "mode={mode}"
                );
                assert!(
                    run(
                        &format!(">query\n{ambiguous}\n"),
                        &format!(">read\n{query}\n"),
                        &opts
                    )
                    .is_empty(),
                    "reference mode={mode}"
                );
            }
        }
    }
}

#[test]
fn sums_spans_and_preserves_complete_original_records() {
    let a = "A".repeat(21);
    let c = "C".repeat(21);
    let patterns = format!(">a\n{a}\n>c\n{c}\n");
    for mode in ["", "--high-precision", "--approx"] {
        let mut opts = vec!["-k", "21", "-m", "21", "-t", "2", "-T", "3"];
        if !mode.is_empty() {
            opts.push(mode);
        }
        let mixed = format!("nn{}NRYSWKMBDHVNX-{}n", a.to_lowercase(), c);
        for reads in [
            format!(
                ">keep description\r\nnn{}N\r\nRYSWKMBDHVNX-{}n\r\n>reject\r\n{}NN\r\n>unknown\r\n{}\r\n",
                a.to_lowercase(),
                c,
                a,
                "N".repeat(100)
            ),
            format!(
                "@keep description\n{mixed}\n+\n{}\n@reject\n{a}NN\n+\n{}\n@unknown\n{}\n+\n{}\n",
                "I".repeat(mixed.len()),
                "I".repeat(a.len() + 2),
                "N".repeat(100),
                "I".repeat(100)
            ),
        ] {
            assert_eq!(
                run(&patterns, &reads, &opts),
                HashMap::from([("keep description".to_owned(), mixed.clone())]),
                "mode={mode}"
            );
        }
    }
}

#[test]
fn agrees_with_contiguous_sequence_oracle_and_preserves_relative_denominator() {
    use std::collections::HashSet;
    let mut state = 91u64;
    let mut reference: Vec<u8> = (0..3000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b"ACGT"[(state & 3) as usize]
        })
        .collect();
    for pos in [0, 63, 64, 127, 128, 129, 300, 650, 1000, 2000, 2999] {
        reference[pos] = b'N';
    }
    let reference = String::from_utf8(reference).unwrap();
    let mut reads: Vec<String> = (0..20)
        .map(|i| reference[i * 100..i * 100 + 200].to_owned())
        .collect();
    reads.push(reference[500..700].replace('N', "G"));
    reads.push(format!(
        "{}N{}",
        &reference[700..760],
        &reference[1300..1360]
    ));
    reads.push(format!(
        "{}n{}",
        reference[700..760].to_lowercase(),
        &reference[1300..1360]
    ));
    reads.push("N".repeat(200));
    reads.push("ACGT".to_owned());
    let patterns = format!(">reference\n{reference}\n");
    let fasta: String = reads
        .iter()
        .enumerate()
        .map(|(i, seq)| format!(">r{i}\n{seq}\n"))
        .collect();
    for k in [21, 31, 32, 33, 63] {
        let kmers: HashSet<&[u8]> = reference
            .as_bytes()
            .windows(k)
            .filter(|word| word.iter().all(|b| b"ACGT".contains(b)))
            .collect();
        for threshold in ["2", "0.5", "0.9", "1.0"] {
            let expected: HashMap<String, String> = reads
                .iter()
                .enumerate()
                .filter(|(_, seq)| {
                    if seq.len() < k {
                        return false;
                    }
                    let required = if threshold.contains('.') {
                        ((seq.len() - k + 1) as f64 * threshold.parse::<f64>().unwrap()).ceil()
                            as usize
                    } else {
                        threshold.parse().unwrap()
                    };
                    let upper = seq.to_ascii_uppercase();
                    upper
                        .as_bytes()
                        .windows(k)
                        .filter(|word| kmers.contains(word))
                        .count()
                        >= required
                })
                .map(|(i, seq)| (format!("r{i}"), seq.clone()))
                .collect();
            for threads in ["1", "3"] {
                for mode in ["default", "high"] {
                    let mut options = vec!["-k", "0", "-t", threshold, "-T", threads];
                    let k_string = k.to_string();
                    options[1] = &k_string;
                    if mode == "high" {
                        options.push("--high-precision");
                    }
                    assert_eq!(
                        run(&patterns, &fasta, &options),
                        expected,
                        "k={k}, threshold={threshold}, threads={threads}, mode={mode}"
                    );
                }
            }
        }
    }
}

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Inputs(PathBuf);

impl Inputs {
    fn new(patterns: &str, reads: &str) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "k2rmini-high-precision-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("patterns.fa"), patterns).unwrap();
        fs::write(path.join("reads.fastx"), reads).unwrap();
        Self(path)
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_K2Rmini"))
            .arg("-p")
            .arg(self.0.join("patterns.fa"))
            .args(args)
            .arg(self.0.join("reads.fastx"))
            .output()
            .unwrap()
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn selected(output: Output) -> HashSet<String> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix('>').map(str::to_owned))
        .collect()
}

#[test]
fn rejects_a_bitset_collision_and_keeps_the_actual_match() {
    // Same minimizer and same low 16 hash bits, but different complete k-mers.
    let inputs = Inputs::new(
        ">reference\nGGTTGTATGAAAAAAAAAAAAAAAAAAAAAA\n",
        ">collision\nTCCTTTCTTGAAAAAAAAAAAAAAAAAAAAA\n>exact\nGGTTGTATGAAAAAAAAAAAAAAAAAAAAAA\n",
    );
    assert_eq!(
        selected(inputs.run(&["-t", "1", "-T", "1"])),
        HashSet::from(["collision".into(), "exact".into()])
    );
    for threads in ["1", "3"] {
        assert_eq!(
            selected(inputs.run(&["--high-precision", "-t", "1", "-T", threads])),
            HashSet::from(["exact".into()])
        );
    }
}

#[test]
fn high_precision_conflicts_with_approx() {
    let inputs = Inputs::new(">reference\nACGT\n", ">read\nACGT\n");
    let output = inputs.run(&["--high-precision", "--approx"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
}

fn dna(len: usize, state: &mut u64) -> String {
    (0..len)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            b"ACGT"[(*state & 3) as usize] as char
        })
        .collect()
}

#[test]
fn agrees_with_exact_counts_across_chunk_boundaries_and_kmer_sizes() {
    let mut state = 42;
    // Cross the 65,536-k-mer indexing boundary and include multiple records.
    let reference = dna(70_000, &mut state);
    let other_reference = dna(700, &mut state);
    let reads = [
        reference[0..200].to_owned(),
        reference[65_500..65_700].to_owned(),
        reference[69_800..].to_owned(),
        other_reference[100..300].to_owned(),
        format!("{}{}", &reference[100..240], dna(60, &mut state)),
        dna(200, &mut state),
        "ACGT".to_owned(),
    ];
    // Exercise FASTQ input and FASTA output as well as the filtering decisions.
    let fastq: String = reads
        .iter()
        .enumerate()
        .map(|(i, read)| format!("@read{i}\n{read}\n+\n{}\n", "I".repeat(read.len())))
        .collect();
    let inputs = Inputs::new(
        &format!(">reference\n{reference}\n>other\n{other_reference}\n"),
        &fastq,
    );

    for k in [31, 32, 33, 63, 96] {
        let kmers: HashSet<&[u8]> = reference
            .as_bytes()
            .windows(k)
            .chain(other_reference.as_bytes().windows(k))
            .collect();
        for threshold in ["1", "3", "0.5", "1.0"] {
            let expected: HashSet<String> = reads
                .iter()
                .enumerate()
                .filter(|(_, read)| {
                    if read.len() < k {
                        return false;
                    }
                    let required = if threshold.contains('.') {
                        ((read.len() - k + 1) as f64 * threshold.parse::<f64>().unwrap()).ceil()
                            as usize
                    } else {
                        threshold.parse().unwrap()
                    };
                    read.as_bytes()
                        .windows(k)
                        .filter(|word| kmers.contains(word))
                        .count()
                        >= required
                })
                .map(|(i, _)| format!("read{i}"))
                .collect();
            for threads in ["1", "3"] {
                assert_eq!(
                    selected(inputs.run(&[
                        "--high-precision",
                        "-k",
                        &k.to_string(),
                        "-t",
                        threshold,
                        "-T",
                        threads
                    ])),
                    expected,
                    "k={k}, threshold={threshold}, threads={threads}"
                );
            }
        }
    }
}

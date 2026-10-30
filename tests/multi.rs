use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Inputs {
    root: PathBuf,
}
impl Inputs {
    fn new(references: &[String], reads: &str) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "k2rmini-parity-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        for (i, reference) in references.iter().enumerate() {
            fs::write(root.join(format!("q{i}.fa")), reference).unwrap();
        }
        fs::write(root.join("reads.fastx"), reads).unwrap();
        Self { root }
    }
    fn query(&self, id: usize) -> String {
        self.root
            .join(format!("q{id}.fa"))
            .to_str()
            .unwrap()
            .to_owned()
    }
    fn run(&self, multi: bool, args: &[String]) -> Output {
        Command::new(if multi {
            env!("CARGO_BIN_EXE_k2rminimulti")
        } else {
            env!("CARGO_BIN_EXE_K2Rmini")
        })
        .args(args)
        .arg(self.root.join("reads.fastx"))
        .output()
        .unwrap()
    }
    fn single(&self, query: usize, threshold: &str, options: &[String]) -> HashMap<String, String> {
        let mut args = vec![
            "-p".into(),
            self.query(query),
            "-t".into(),
            threshold.into(),
        ];
        args.extend_from_slice(options);
        records(self.run(false, &args))
    }
    fn multi(&self, constraints: &[(usize, &str)], options: &[String]) -> HashMap<String, String> {
        let mut args = Vec::new();
        for &(id, threshold) in constraints {
            args.extend(["-c".into(), self.query(id), threshold.into()]);
        }
        args.extend_from_slice(options);
        records(self.run(true, &args))
    }
}
impl Drop for Inputs {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn records(output: Output) -> HashMap<String, String> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len() % 2, 0);
    let mut records = HashMap::new();
    for pair in lines.chunks_exact(2) {
        assert!(
            records
                .insert(
                    pair[0].strip_prefix('>').unwrap().to_owned(),
                    pair[1].to_owned()
                )
                .is_none()
        );
    }
    records
}

fn options(mode: &str, k: usize, threads: &str) -> Vec<String> {
    let mut args = vec![
        "-k".into(),
        k.to_string(),
        "-m".into(),
        "21".into(),
        "-T".into(),
        threads.into(),
    ];
    if !mode.is_empty() {
        args.push(mode.into());
    }
    args
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
fn multi_is_the_intersection_of_single_runs_in_every_mode() {
    let mut state = 42;
    let first = dna(70_000, &mut state);
    let second = format!("{}N{}", &first[65_000..66_000], dna(1500, &mut state));
    let third = format!(
        "n{}R{}n",
        first[65_300..65_700].to_lowercase(),
        dna(1000, &mut state)
    );
    let reads = [
        first[0..200].to_owned(),
        first[65_480..65_680].to_owned(),
        first[69_800..].to_owned(),
        format!("{}N{}", &first[65_350..65_450], &first[65_500..65_600]),
        format!(
            "{}RYSWKMBDHVNX-{}",
            first[65_300..65_400].to_lowercase(),
            &first[65_500..65_600]
        ),
        dna(200, &mut state),
        "N".repeat(200),
        "ACGT".into(),
    ];
    let fasta: String = reads
        .iter()
        .enumerate()
        .map(|(i, seq)| {
            format!(
                ">r{i} description\r\n{}\r\n{}\r\n",
                &seq[..seq.len() / 2],
                &seq[seq.len() / 2..]
            )
        })
        .collect();
    let fastq: String = reads
        .iter()
        .enumerate()
        .map(|(i, seq)| format!("@r{i} description\n{seq}\n+\n{}\n", "I".repeat(seq.len())))
        .collect();
    for input in [fasta, fastq] {
        let data = Inputs::new(
            &[
                format!(">first\n{first}\n"),
                format!(">second\n{second}\n"),
                format!(">third\n{third}\n"),
            ],
            &input,
        );
        for mode in ["", "--high-precision", "--approx"] {
            for k in [21, 31, 32, 33, 63, 96] {
                for threads in ["1", "3"] {
                    let args = options(mode, k, threads);
                    let constraints = [(0, "0.5"), (1, "3"), (2, "0.25")];
                    let mut expected = data.single(0, "0.5", &args);
                    assert_eq!(data.multi(&constraints[..1], &args), expected);
                    let mut shorthand = vec!["-p".into(), data.query(0), "-t".into(), "0.5".into()];
                    shorthand.extend_from_slice(&args);
                    assert_eq!(records(data.run(true, &shorthand)), expected);
                    for &(id, threshold) in &constraints[1..] {
                        let selected = data.single(id, threshold, &args);
                        expected.retain(|id, seq| selected.get(id) == Some(seq));
                    }
                    assert_eq!(
                        data.multi(&constraints, &args),
                        expected,
                        "mode={mode}, k={k}, threads={threads}"
                    );
                }
            }
        }
    }
}

#[test]
fn default_collisions_match_single_and_cannot_borrow_another_querys_minimizer() {
    let reference = "GGTTGTATGAAAAAAAAAAAAAAAAAAAAAA";
    let collision = "TCCTTTCTTGAAAAAAAAAAAAAAAAAAAAA";
    let data = Inputs::new(
        &[format!(">q0\n{reference}\n"), format!(">q1\n{collision}\n")],
        &format!(">collision\n{collision}\n>exact\n{reference}\n"),
    );
    assert!(
        data.multi(&[(0, "1"), (1, "1")], &options("", 31, "3"))
            .contains_key("collision")
    );
    assert!(
        data.multi(&[(0, "1"), (1, "1")], &options("--high-precision", 31, "3"))
            .is_empty()
    );
    // Same composite address but different full minimizers: query 1's minimizer
    // must not authorize a key hit in query 0.
    let q0 = "AAAAAAAAACCTTCTCAAATGAGTATGTGCT";
    let q1 = "AAAAAAAACGGGAACGAAGGACTCGTCATGT";
    let data = Inputs::new(
        &[format!(">q0\n{q0}\n"), format!(">q1\n{q1}\n")],
        &format!(">read\n{q1}\n"),
    );
    assert!(data.single(0, "1", &options("", 31, "1")).is_empty());
    assert!(
        data.multi(&[(0, "1"), (1, "1")], &options("", 31, "3"))
            .is_empty()
    );
}

#[test]
fn supports_all_32_queries_and_rejects_empty_constraints_like_single() {
    let a = "A".repeat(21);
    let c = "C".repeat(21);
    let data = Inputs::new(
        &[
            format!(">a\n{a}\n"),
            format!(">c\n{c}\n"),
            String::new(),
            format!(">unknown\n{}\n", "N".repeat(100)),
        ],
        &format!(">a\n{a}\n"),
    );
    for mode in ["", "--high-precision", "--approx"] {
        let args = options(mode, 21, "3");
        let mut constraints = vec![(0, "1"); 32];
        assert_eq!(
            data.multi(&constraints, &args),
            HashMap::from([("a".into(), a.clone())])
        );
        constraints[31] = (1, "1");
        assert!(data.multi(&constraints, &args).is_empty());
        for id in [2, 3] {
            assert!(data.single(id, "1", &args).is_empty());
            assert!(data.multi(&[(0, "1"), (id, "1")], &args).is_empty());
        }
    }
}

#[test]
fn common_options_validate_identically_and_multi_constraints_remain_required() {
    let data = Inputs::new(&[">q\nACGT\n".into()], ">read\nACGT\n");
    for bad in [
        vec!["--approx", "--high-precision"],
        vec!["-k", "0"],
        vec!["-m", "0"],
        vec!["-m", "30"],
        vec!["-k", "20"],
        vec!["-T", "0"],
        vec!["-t", "0"],
        vec!["-t", "NaN"],
    ] {
        for multi in [false, true] {
            let mut args = vec!["-p".into(), data.query(0)];
            args.extend(bad.iter().map(|arg| (*arg).into()));
            assert_eq!(
                data.run(multi, &args).status.code(),
                Some(2),
                "multi={multi}, args={args:?}"
            );
        }
    }
    assert_eq!(data.run(true, &[]).status.code(), Some(2));
    for constraints in [
        vec!["-c".into(), data.query(0)],
        vec!["-c".into(), data.query(0), "bad".into()],
        vec![
            "-p".into(),
            data.query(0),
            "-c".into(),
            data.query(0),
            "1".into(),
        ],
    ] {
        assert_eq!(data.run(true, &constraints).status.code(), Some(2));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn output_errors_are_reported_by_both_binaries() {
    let seq = "A".repeat(31);
    let data = Inputs::new(&[format!(">q\n{seq}\n")], &format!(">read\n{seq}\n"));
    for multi in [false, true] {
        let args = vec![
            "-p".into(),
            data.query(0),
            "-t".into(),
            "1".into(),
            "-o".into(),
            "/dev/full".into(),
        ];
        assert!(!data.run(multi, &args).status.success());
    }
}

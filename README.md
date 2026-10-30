# K2Rmini: filter a set of reads using k-mers

**K2Rmini** (or *K-mer to Reads* mini) is a tool to filter the reads contained in a FASTA/Q file based on a set of *k*-mers of interest.

Under the hood, it uses [simd-minimizers](https://github.com/rust-seq/simd-minimizers) to quickly prefilter reads based on their minimizers, and filters the remaining candidates using the *k*-mer set. On an Apple M1, `K2Rmini` is able to filter long reads at ~2 Gbp/s.

## Installation

If you have not installed Rust yet, please visit [rustup.rs](https://rustup.rs/) to install it.

```sh
git clone https://github.com/Malfoy/K2Rmini.git
cd K2Rmini
RUSTFLAGS="-C target-cpu=native" cargo install --path .
```

This will compile the `K2Rmini` and `k2rminimulti` binaries and add them to your path.

## Usage

```
Usage: K2Rmini [OPTIONS] -p <PATTERNS> <FILE>

Arguments:
  <FILE>  FASTA/Q file to filter (possibly compressed)

Options:
  -p <PATTERNS>                FASTA/Q file containing k-mers of interest (possibly compressed)
  -t, --threshold <THRESHOLD>  K-mer threshold, either relative (float) or absolute (int) [default: 0.5]
  -o <OUTPUT>                  Output file for filtered sequences [default: stdout]
  -k <K>                       K-mer size [default: 31]
  -m <M>                       Minimizer size, must be ≤ k, up to 29 [default: 21]
      --high-precision         Use a global 64-bit set: exact k-mer keys for k <= 32, hashes for larger k
  -a, --approx                 Approximate filtering using minimizers only
  -T, --threads <THREADS>      Number of threads [default: all]
  -h, --help                   Print help
  -V, --version                Print version
```

`K2Rmini` has 3 main arguments:
- a FASTA/Q file containing the sequences that you want to filter, this file can be compressed using `gzip` / `xz` / `zstd`
- a FASTA/Q file (flagged with `-p`) containing the *k*-mers of interest used for filtering: sequences containing enough of these *k*-mers will be outputed, while others will be discarded
- a selection threshold (flagged with `-t`): a sequence is discarded if its number of desired *k*-mers is below this threshold, the threshold can be relative (e.g. at least 90% of desired *k*-mers) or absolute (e.g. at least 2 desired *k*-mers)

It also provides options to write the output to a file (`-o`), set the *k*-mer size (`-k`) or set the number of threads (`-T`).
You shouldn't need to change the minimizer size (`-m`), excepted if `k` is smaller than 25.

### Example: selecting reads with ≥90% of desired *k*-mers

Let's say we want to filter the reads in `reads.fa` to only keep those that share at least 90% of their *k*-mers with the reference in `reference.fa`, this can be achieved with:
```sh
K2Rmini -p reference.fa -t 0.9 reads.fa
```

### Example: selecting reads with ≥2 desired *k*-mers

Let's say this time we have a list of *k*-mers of size 63 stored in `patterns.fa` and we want to select the reads in `reads.fa` that contain at least two of them, this can be achieved with:
```sh
K2Rmini -p patterns.fa -k 63 -t 2 reads.fa
```

### High precision mode

Use `--high-precision` to reduce false positives caused by collisions in the default k-mer bitset:

```sh
K2Rmini --high-precision -p reference.fa -t 0.9 reads.fa
```

For a single query, this mode replaces the k-mer bitset with one global `FxHashSet<u64>`; multiple queries attach presence masks to the same keys. For `k <= 32`, each key is the complete packed k-mer (two bits per base), giving exact membership for A/C/G/T k-mers without fingerprint collisions. For `k > 32`, each key is a full 64-bit hash computed from all packed blocks using Rust's `DefaultHasher`, giving high precision while remaining probabilistic. These hashes are computed from the bases, rather than widening the existing 32-bit hashes. The existing minimizer prefilter is retained.

Memory grows with the number of distinct keys, and indexing and filtering can be slower than the default mode. Fingerprint collisions remain possible for `k > 32`. Threshold calculation and strand specificity are unchanged. Both binaries support this option; it cannot be combined with `--approx`.

### Ambiguous bases

`K2Rmini` treats any non-A/C/G/T character (including `N`, lowercase `n`, and other ambiguity symbols) as a boundary in both reference sequences and reads. Only k-mers wholly inside a contiguous A/C/G/T span can match; spans are never joined. Lowercase A/C/G/T bases are supported. Matches are accumulated over the original read, which is output once with its original sequence and ambiguous characters preserved.

Relative thresholds retain the denominator `read_length - k + 1`, including positions overlapping ambiguous bases. Those positions contribute no matches. Existing threshold capping by reference index cardinality is retained, independently for each constraint. An empty reference, or one without valid k-mers, cannot satisfy a constraint. This behavior applies to default, high precision, and approximate filtering in both binaries.

The reproducible all-core RAM benchmark is `python3 benchmarks/ambiguity.py --phase all` (requires NumPy). It compares mask-based spans, sequence-text scanning, and native Helicase splitting; the kernel benchmark checks identical results across methods. Full CLI timings also compare native split indexing with masked read filtering. Inputs are generated under `/dev/shm` and benchmark builds and results under `/tmp/k2rmini-ambiguity-bench`.

[Benchmark results](benchmarks/ambiguity-results.json), recorded before the shared-engine refactor, include 832 timed samples on an i9-13950HX with 32 threads, synthetic 150-base and 10,000-base reads, and clean, scattered-N, and clustered-N inputs. Mask-based processing was the best overall choice: full CLI runs were approximately 5–8% faster than sequence-text scanning, while native split indexing with masked read filtering was effectively tied overall. Native splitting alone was slightly faster on some clean-input kernels; those kernel timings exclude preserving and outputting whole reads. Clean-read filtering overhead versus the unfixed implementation was approximately 2–7% on the larger inputs. These measurements use `k=31` and do not establish a universal winner for all datasets or hardware.

## K2Rminimulti

`k2rminimulti` filters reads using several query files at once. It is useful when a read must share enough *k*-mers with several independent query sets, for example at least `X` *k*-mers with `Q1.fa` and at least `Y` *k*-mers with `Q2.fa`.

A read is kept only if it satisfies every constraint. In other words, constraints are combined with AND semantics.

```
Usage: k2rminimulti [OPTIONS] <FILE>

Arguments:
  <FILE>  FASTA/Q file to filter (possibly compressed)

Options:
  -p <PATTERNS>                           Single-query shorthand, as in K2Rmini
  -t, --threshold <THRESHOLD>              Threshold for -p [default: 0.5]
  -c, --constraint <PATTERNS> <THRESHOLD>  FASTA/Q file containing k-mers of interest and its threshold; may be repeated
  -o <OUTPUT>                              Output file for filtered sequences [default: stdout]
  -k <K>                                   K-mer size [default: 31]
  -m <M>                                   Minimizer size, must be <= k, up to 29 [default: 21]
  -a, --approx                             Approximate filtering using minimizers only
      --high-precision                     Exact k-mer keys for k <= 32, 64-bit hashes for larger k
  -T, --threads <THREADS>                  Number of threads [default: all]
  -h, --help                               Print help
  -V, --version                            Print version
```

`k2rminimulti` has 2 main arguments:
- a FASTA/Q file containing the sequences that you want to filter, this file can be compressed using `gzip` / `xz` / `zstd`
- one or more constraints, each flagged with `-c` / `--constraint`, made of a FASTA/Q query file and the threshold associated with that query

The threshold syntax is the same as `K2Rmini`: an integer is interpreted as an absolute number of shared *k*-mers, while a float in `(0, 1]` is interpreted as a fraction of the read's *k*-mers. The `-p PATTERNS -t THRESHOLD` shorthand selects one query, with a default threshold of `0.5`. Use either `-p` or repeated `-c` constraints; they cannot be mixed, and `-t` applies only to `-p`.

All common options, defaults, validation, precision modes, and ambiguity handling are shared with `K2Rmini`:

```sh
k2rminimulti --high-precision -c Q1.fa 10 -c Q2.fa 0.25 reads.fa
k2rminimulti --approx -c Q1.fa 10 -c Q2.fa 0.25 reads.fa
k2rminimulti --high-precision -p Q1.fa -t 0.9 reads.fa
```

### Example: selecting reads matching two query files

Let's say we want to keep reads from `reads.fa` only when they share at least 10 *k*-mers with `Q1.fa` and at least 5 *k*-mers with `Q2.fa`:
```sh
k2rminimulti -c Q1.fa 10 -c Q2.fa 5 reads.fa
```

This is equivalent to the logical condition:
```text
shared_kmers(read, Q1.fa) >= 10 AND shared_kmers(read, Q2.fa) >= 5
```

### Example: mixing absolute and relative thresholds

```sh
k2rminimulti -c Q1.fa 10 -c Q2.fa 5 -c Q3.fa 0.25 reads.fa
```

This keeps reads sharing at least 10 *k*-mers with `Q1.fa`, at least 5 with `Q2.fa`, and at least 25% of their *k*-mers with `Q3.fa`. The constraints are combined with AND semantics.

### Implementation notes

Both binaries call the same library for parallel reference indexing, ambiguity masks, minimizer prefiltering, k-mer checks, thresholds, parallel read parsing, and buffered FASTA output. With one constraint, `k2rminimulti` uses the same index backend as `K2Rmini`.

With multiple constraints, each reference is indexed using the same parallel indexer, then merged into shared sharded maps. Each map value is a 32-bit query-presence mask, supporting up to 32 constraints:

- Minimizer keys retain the complete 64-bit representation used by `K2Rmini`.
- Default k-mer keys are the same 32-bit composite bitset addresses: 16 bits from the minimizer-derived page and 16 bits from the k-mer hash.
- High-precision keys retain the complete packed k-mer for `k <= 32`, or a full 64-bit hash for larger k.
- Approximate mode indexes only minimizers.

Reads are parsed once. Minimizers and k-mer keys are computed once and shared across constraints, with matches counted independently for each query. A minimizer and k-mer hit must belong to the same query. Output order is unspecified, as in `K2Rmini`.

Default filtering remains probabilistic because composite addresses can collide. High precision removes fingerprint collisions for `k <= 32` and reduces them for larger k. Multi-query storage uses presence maps rather than allocating a dense query mask for every possible bitset address.

## Benchmarks

Benchmarks and plots against other sequence filtering tools are available in the [experiments repository](https://github.com/imartayan/K2Rmini_experiments).

## Citation

> Accelerating *k*-mer based sequence filtering. I. Martayan, L. Vandamme, B. Constantinides, B. Cazaux, C. Paperman and A. Limasset. https://doi.org/10.1101/2025.06.16.659853

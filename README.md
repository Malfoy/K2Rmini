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

## K2Rminimulti

`k2rminimulti` filters reads using several query files at once. It is useful when a read must share enough *k*-mers with several independent query sets, for example at least `X` *k*-mers with `Q1.fa` and at least `Y` *k*-mers with `Q2.fa`.

A read is kept only if it satisfies every constraint. In other words, constraints are combined with AND semantics.

```
Usage: k2rminimulti [OPTIONS] --constraint <PATTERNS> <THRESHOLD> <FILE>

Arguments:
  <FILE>  FASTA/Q file to filter (possibly compressed)

Options:
  -c, --constraint <PATTERNS> <THRESHOLD>  FASTA/Q file containing k-mers of interest and its threshold; may be repeated
  -o <OUTPUT>                              Output file for filtered sequences [default: stdout]
  -k <K>                                   K-mer size [default: 31]
  -m <M>                                   Minimizer size, must be <= k, up to 29 [default: 21]
  -T, --threads <THREADS>                  Number of threads [default: all]
  -h, --help                               Print help
  -V, --version                            Print version
```

`k2rminimulti` has 2 main arguments:
- a FASTA/Q file containing the sequences that you want to filter, this file can be compressed using `gzip` / `xz` / `zstd`
- one or more constraints, each flagged with `-c` / `--constraint`, made of a FASTA/Q query file and the threshold associated with that query

The threshold syntax is the same as `K2Rmini`: an integer is interpreted as an absolute number of shared *k*-mers, while a float in `(0, 1]` is interpreted as a fraction of the read's *k*-mers.

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

`k2rminimulti` indexes all query files into shared maps:
- the *k*-mer map stores a 32-bit *k*-mer hash as key and a 32-bit query presence mask as value
- the minimizer map stores a 32-bit minimizer key and a 32-bit query presence mask as value

Each bit in the presence mask corresponds to one query file, so `k2rminimulti` supports up to 32 query files. During filtering, a read is first prefiltered using minimizers, then candidate reads are checked using their *k*-mer hashes.

Because both *k*-mers and minimizers are represented with 32-bit keys, hash/key collisions can create false positives. This matches the current 32-bit *k*-mer hash behavior of `K2Rmini`, while also applying 32-bit keys to minimizers in `k2rminimulti`.

## Benchmarks

Benchmarks and plots against other sequence filtering tools are available in the [experiments repository](https://github.com/imartayan/K2Rmini_experiments).

## Citation

> Accelerating *k*-mer based sequence filtering. I. Martayan, L. Vandamme, B. Constantinides, B. Cazaux, C. Paperman and A. Limasset. https://doi.org/10.1101/2025.06.16.659853

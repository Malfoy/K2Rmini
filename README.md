# K2Rmini: filter a set of reads using k-mers

**K2Rmini** (or *K-mer to Reads* mini) is a tool to filter the reads contained in a FASTA/Q file based on a set of *k*-mers of interest.

Under the hood, it uses [simd-minimizers](https://github.com/rust-seq/simd-minimizers) to quickly prefilter reads based on their minimizers, and filters the remaining candidates using the *k*-mer set. On an Apple M1, `K2Rmini` is able to filter long reads at 450Mbp/s.

## Installation

If you have not installed Rust yet, please visit [rustup.rs](https://rustup.rs/) to install it.

```sh
git clone https://github.com/Malfoy/K2Rmini.git
cd K2Rmini
RUSTFLAGS="-C target-cpu=native" cargo install --path .
```

This will compile a binary called `K2Rmini` and add it to your path.

## Usage

```
Usage: K2Rmini [OPTIONS] -p <PATTERNS> <FILE>

Arguments:
  <FILE>  FASTA/Q file to filter (possibly compressed)

Options:
  -p <PATTERNS>                FASTA/Q file containing k-mers of interest (possibly compressed)
  -o <OUTPUT>                  Output file for filtered sequences [default: stdout]
  -k <K>                       K-mer size [default: 31]
  -m <M>                       Minimizer size [default: 21]
  -t, --threshold <THRESHOLD>  K-mer threshold, either absolute (int) or relative (float) [default: 0.5]
  -T, --threads <THREADS>      Number of threads [default: all]
  -h, --help                   Print help
  -V, --version                Print version
```

## Citation

A preprint is coming soon.
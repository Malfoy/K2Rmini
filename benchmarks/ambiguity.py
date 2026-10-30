#!/usr/bin/env python3
"""Reproduce the all-core RAM ambiguity benchmarks (Python + NumPy).

Run from the repository: python3 benchmarks/ambiguity.py --phase all
Inputs live on tmpfs; results and isolated candidate builds live under /tmp.
Kernel timings load input once before timing and validate checksums across methods.
Full CLI timings include indexing, filtering, and output to /dev/null.
Native split is measured as a kernel only: its chunk API does not preserve the
original full read for output. Hybrid uses native split indexing + masked reads.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import resource
import shutil
import statistics
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/tmp/k2rmini-ambiguity-bench')
DATA = Path('/dev/shm/k2rmini-ambiguity-bench')


def generate(short_mb, long_mb):
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261030)
    alphabet = np.frombuffer(b'ACGT', dtype=np.uint8)
    reference = alphabet[rng.integers(0, 4, 1_000_000)].tobytes()
    unrelated = alphabet[rng.integers(0, 4, 1_000_000)].tobytes()
    (DATA / 'patterns.fa').write_bytes(b'>reference\n' + reference + b'\n')
    ambiguous_reference = bytearray(reference)
    ambiguous_reference[500::1000] = b'N' * len(ambiguous_reference[500::1000])
    (DATA / 'patterns_N.fa').write_bytes(b'>reference\n' + ambiguous_reference + b'\n')
    cases = []
    for kind, length, size_mb in [('short', 150, short_mb), ('long', 10_000, long_mb)]:
        records = max(1, 8_000_000 // length)
        clean = np.empty((records, length), dtype=np.uint8)
        for i in range(records):
            offset = int(rng.integers(0, len(reference) - length))
            pool = reference if i % 2 == 0 else unrelated
            clean[i] = np.frombuffer(pool[offset:offset + length], dtype=np.uint8)
        for ambiguity in ['clean', 'sparse', 'dense', 'cluster']:
            seqs = clean.copy()
            if ambiguity in ['sparse', 'dense']:
                seqs[rng.random(seqs.shape) < (0.001 if ambiguity == 'sparse' else 0.01)] = ord('N')
            elif ambiguity == 'cluster':
                seqs[:, length // 2:length // 2 + max(1, length // 100)] = ord('N')
            block = b''.join(b'>r%d\n' % i + row.tobytes() + b'\n' for i, row in enumerate(seqs))
            repeats = max(1, round(size_mb * 1_000_000 / seqs.size))
            name = f'{kind}_{ambiguity}'
            path = DATA / f'{name}.fa'
            with path.open('wb') as f:
                for _ in range(repeats):
                    f.write(block)
            cases.append(dict(name=name, path=str(path), bases=int(seqs.size) * repeats,
                              bytes=path.stat().st_size,
                              patterns=str(DATA / ('patterns.fa' if ambiguity == 'clean' else 'patterns_N.fa'))))
            print('generated', name, cases[-1]['bases'], flush=True)
    (WORK / 'datasets.json').write_text(json.dumps(cases, indent=2))


def build():
    sources = {path.name: path.read_text() for path in (ROOT / 'src').glob('*.rs')}
    # Preserve candidate snapshots; building never rewrites the working source.
    variants = {'mask': dict(sources)}
    text = dict(sources)
    for module in ['index.rs', 'filter.rs']:
        source = text[module].replace('.compute_mask_non_actg()', '')
        source = source.replace('.ignore_headers()\n    .dna_packed()', '.dna_string()\n    .and_dna_packed()')
        source = source.replace('use crate::ambiguity::valid_spans;', 'use crate::ambiguity::text_spans;')
        source = source.replace('valid_spans(parser.get_mask_non_actg(),', 'text_spans(parser.get_dna_string(),')
        if module == 'filter.rs':
            source = source.replace('mask: &dna_format::BitMask', 'dna: &[u8]')
            source = source.replace('valid_spans(mask,', 'text_spans(dna,')
            source = source.replace('parser.get_mask_non_actg()', 'parser.get_dna_string()')
        text[module] = source
    text['ambiguity.rs'] += '''
pub fn text_spans(dna: &[u8], k: usize, out: &mut Vec<std::ops::Range<usize>>) {
    out.clear(); let mut start = 0;
    for (i, &base) in dna.iter().enumerate() {
        if !matches!(base.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T') {
            if i - start >= k { out.push(start..i); } start = i + 1;
        }
    }
    if dna.len() - start >= k { out.push(start..dna.len()); }
}
'''
    variants['text'] = text
    hybrid = dict(sources)
    hybrid['index.rs'] = hybrid['index.rs'].replace('    .ignore_headers()\n    .dna_packed()\n    .keep_non_actg()\n    .compute_mask_non_actg()', '    .ignore_headers()\n    .dna_packed()\n    .split_non_actg()')
    hybrid['index.rs'] = hybrid['index.rs'].replace('valid_spans(parser.get_mask_non_actg(), args.k, &mut spans);',
                            'spans.clear(); if record.len() >= args.k { spans.push(0..record.len()); }')
    hybrid['index.rs'] = hybrid['index.rs'].replace('use crate::ambiguity::valid_spans;\n', '')
    variants['hybrid'] = hybrid
    baseline = WORK / 'baseline-main.rs'
    if baseline.exists():
        variants['baseline'] = dict(sources, **{'main.rs': baseline.read_text()})
    for name, modules in variants.items():
        project = WORK / name
        (project / 'src').mkdir(parents=True, exist_ok=True)
        # Unique package/binary names prevent candidates overwriting the user's
        # normal target/release/K2Rmini executable in the shared Cargo cache.
        binary_name = f'K2Rmini-ambiguity-{name}'
        manifest = (ROOT / 'Cargo.toml').read_text().replace(
            'name = "K2Rmini"', f'name = "{binary_name}"', 1)
        (project / 'Cargo.toml').write_text(manifest)
        shutil.copy(ROOT / 'Cargo.lock', project / 'Cargo.lock')
        # Finer timings in isolated benchmark builds only.
        for filename, source in modules.items():
            (project / 'src' / filename).write_text(source.replace('{:.02}', '{:.06}'))
        subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(project / 'Cargo.toml'),
                        '--target-dir', str(ROOT / 'target'), '--bin', binary_name], cwd=ROOT, check=True)
        shutil.copy(ROOT / 'target/release' / binary_name, WORK / f'K2Rmini-{name}')
        print('built', name, flush=True)
    subprocess.run(['cargo', 'build', '--release', '--example', 'bench_ambiguity'], cwd=ROOT, check=True)


def kernel(cases, repeats):
    rows, checksums = [], {}
    rng = random.Random(42)
    jobs = [(case, precision, method) for case in cases for precision in ['default', 'high']
            for method in ['mask', 'text', 'split']]
    rng.shuffle(jobs)
    for case, precision, method in jobs:
        output = subprocess.check_output([str(ROOT / 'target/release/examples/bench_ambiguity'), method,
            case['path'], '31', str(repeats), precision], text=True)
        for line in output.splitlines():
            _, threads, repeat, seconds, *checksum = line.split(',')
            key = (case['name'], precision)
            checksum = tuple(checksum)
            assert checksums.setdefault(key, checksum) == checksum, (key, method, checksum, checksums[key])
            rows.append(dict(case=case['name'], precision=precision, method=method, threads=int(threads),
                             repeat=int(repeat), seconds=float(seconds), checksum=checksum,
                             gbps=case['bases'] / float(seconds) / 1e9))
        print('kernel', case['name'], precision, method, statistics.median(r['gbps'] for r in rows
            if r['case'] == case['name'] and r['precision'] == precision and r['method'] == method), flush=True)
        (WORK / 'kernel.json').write_text(json.dumps(rows, indent=2))


def full(cases, repeats, reference_layout):
    if reference_layout == 'multiple':
        adapted = []
        for case in cases:
            source = Path(case['patterns'])
            destination = source.with_suffix('.parallel.fa')
            if not destination.exists():
                seq = b''.join(source.read_bytes().splitlines()[1:])
                destination.write_bytes(b''.join(b'>ref%d\n' % offset + seq[offset:offset+10_000] + b'\n'
                    for offset in range(0, len(seq), 10_000)))
            adapted.append(dict(case, patterns=str(destination)))
        cases = adapted
    output_path = WORK / ('full-parallel-reference.json' if reference_layout == 'multiple' else 'full.json')
    rows = []
    methods = [name for name in ['mask', 'text', 'hybrid', 'baseline'] if (WORK / f'K2Rmini-{name}').exists()]
    rng = random.Random(84)
    for repeat in range(-1, repeats):
        # The unfixed baseline is meaningful only for clean sequences.
        jobs = [(case, precision, method) for case in cases for precision in ['default', 'high']
                for method in methods if method != 'baseline' or case['name'].endswith('_clean')]
        rng.shuffle(jobs)
        for case, precision, method in jobs:
            command = [str(WORK / f'K2Rmini-{method}'), '-p', case['patterns'], '-T', str(len(os.sched_getaffinity(0))),
                       '-t', '0.5', '-o', '/dev/null', case['path']]
            if precision == 'high': command.append('--high-precision')
            cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.perf_counter()
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            elapsed = time.perf_counter() - start
            cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_s = cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime
            times = re.findall(r'Took ([\d.]+) s, RAM: ([\d.]+) GB', result.stderr)
            assert len(times) == 2, result.stderr
            if repeat >= 0:
                rows.append(dict(case=case['name'], precision=precision, method=method, repeat=repeat,
                    threads=len(os.sched_getaffinity(0)), total_s=elapsed, index_s=float(times[0][0]),
                    filter_s=float(times[1][0]), peak_gb=float(times[1][1]),
                    cpu_s=cpu_s, effective_cores=cpu_s / elapsed,
                    gbps=case['bases'] / float(times[1][0]) / 1e9))
        print('full completed round', repeat, flush=True)
        output_path.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'generate', 'build', 'kernel', 'full'], default='all')
    parser.add_argument('--short-mb', type=int, default=128)
    parser.add_argument('--long-mb', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--reference-layout', choices=['multiple', 'single'], default='multiple',
                        help='multiple uses 100 reference records so indexing can use all cores')
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if args.phase in ['all', 'generate']: generate(args.short_mb, args.long_mb)
    if args.phase in ['all', 'build']: build()
    if args.phase in ['all', 'kernel', 'full']:
        cases = json.loads((WORK / 'datasets.json').read_text())
        if args.phase in ['all', 'kernel']: kernel(cases, args.repeats)
        if args.phase in ['all', 'full']: full(cases, args.repeats, args.reference_layout)

import os
import argparse
import subprocess
from multiprocessing import Pool, cpu_count

def compute_metrics(bam):
    sample = os.path.splitext(os.path.basename(bam))[0]
    print('Running process.')
    cmds = [
        (['samtools', 'view', '-c', '-f', '1', bam],             'total'),
        (['samtools', 'view', '-c', '-f', '3', '-F', '780', bam],    'both_mapped'),
        (['samtools', 'view', '-c', '-f', '3', '-F', '516', bam],  'no_pcr'),
        (['samtools', 'view', '-c', '-f', '67', '-F', '516', bam], 'no_pcr_pp'),
        (['samtools', 'view', '-c', '-f', '67', '-F', '516', '-q', '30', bam], 'no_pcr_pp_mapq30'),
    ]
    out = {'sample': sample}
    for cmd, key in cmds:
        count = subprocess.check_output(cmd).strip()
        out[key] = int(count)
    return out

def main():
    p = argparse.ArgumentParser(description="Compute cfDNA quality metrics on BAMs in parallel")
    p.add_argument('bam_dir', help="Directory containing .bam files")
    p.add_argument('-o','--output', default='cfDNA_quality_metrics.tsv',
                   help="Output TSV file (default: cfDNA_quality_metrics.tsv)")
    p.add_argument('-p','--processes', type=int, default=cpu_count(),
                   help="Number of worker processes (default: all CPUs)")
    args = p.parse_args()

    # collect BAMs
    bams = []
    for root, _, files in os.walk(args.bam_dir):
        for fn in files:
            if fn.endswith('.bam'):
                bams.append(os.path.join(root, fn))

    # parallel compute
    with Pool(args.processes) as pool:
        results = pool.map(compute_metrics, bams)

    # write TSV
    header = (
        "Sample\t"
        "Total Fragments\t"
        "Both Ends Unique Mapped\t"
        "No PCR Duplicates\t"
        "No PCR + Properly Paired\t"
        "No PCR + Properly Paired + MAPQ>30\n"
    )
    with open(args.output, 'w') as out:
        out.write(header)
        for r in results:
            out.write(
                f"{r['sample']}\t"
                f"{r['total']}\t"
                f"{r['both_mapped']}\t"
                f"{r['no_pcr']}\t"
                f"{r['no_pcr_pp']}\t"
                f"{r['no_pcr_pp_mapq30']}\n"
            )

if __name__ == "__main__":
    main()

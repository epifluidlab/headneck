import pysam
import py2bit
from tqdm import tqdm

def compute_gc_content(seq):
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

def process_bin(bin_chr, bin_start, bin_end, frag_file, tb):
    gc_contents = []
    for record in frag_file.fetch(bin_chr, bin_start, bin_end):
        fields = record.split()
        chr, start, end, mapq, strand = fields[0], int(fields[1]), int(fields[2]), int(fields[3]), fields[4]
        if mapq > 30 and chr == bin_chr:
            overlap_start = max(bin_start, start)
            overlap_end = min(bin_end, end)
            if overlap_end > overlap_start:
                seq = tb.sequence(chr, overlap_start, overlap_end)
                gc_content = compute_gc_content(seq)
                gc_contents.append(gc_content)
    return sum(gc_contents) / len(gc_contents) if gc_contents else 0

def main(bins_file, frag_bed_gz, two_bit_file, output_file):
    tb = py2bit.open(two_bit_file)
    
    with open(bins_file, 'r') as bins_f, open(output_file, 'w') as out_f, pysam.TabixFile(frag_bed_gz) as frag_file:
        total_lines = sum(1 for _ in open(bins_file))
        bins_f.seek(0)
        
        for line in tqdm(bins_f, total=total_lines, desc="Processing bins"):
            chr, bin_start, bin_end = line.strip().split()
            bin_start, bin_end = int(bin_start), int(bin_end)
            avg_gc_content = process_bin(chr, bin_start, bin_end, frag_file, tb)
            out_f.write(f"{chr}\t{bin_start}\t{bin_end}\t{avg_gc_content:.4f}\n")
    
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main(snakemake.input.bins, snakemake.input.frag_bed, snakemake.input.two_bit, snakemake.output.gc_bins)
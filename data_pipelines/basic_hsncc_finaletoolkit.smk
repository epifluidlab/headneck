# Snakefile

import glob
import os

# Get all input files
INPUT_DIR = "/jet/home/rbandaru/ravi/headneck/data/hg38_frag/"
INPUT_FILES = glob.glob(os.path.join(INPUT_DIR, "*.hg38.frag.bed.gz"))

# Extract sample names
SAMPLES = [os.path.basename(f).replace(".hg38.frag.bed.gz", "") for f in INPUT_FILES]

# Output directory
OUTPUT_DIR = "./processed_data"

# Reference files
BINS_FILE = "./500kb.bins"
REF_GENOME = "./hg38.analysisSet.2bit"

rule all:
    input:
        expand(os.path.join(OUTPUT_DIR, "{sample}.mds.500kb.bed"), sample=SAMPLES),
        expand(os.path.join(OUTPUT_DIR, "{sample}.fraglen.500kb.bed"), sample=SAMPLES),
        expand(os.path.join(OUTPUT_DIR, "{sample}.coverage.500kb.bed"), sample=SAMPLES)

rule process_and_modify:
    input:
        os.path.join(INPUT_DIR, "{sample}.hg38.frag.bed.gz")
    output:
        modified = os.path.join(OUTPUT_DIR, "{sample}.hg38.frag.modified.bed.gz"),
        indexed = os.path.join(OUTPUT_DIR, "{sample}.hg38.frag.modified.bed.gz.tbi")
    shell:
        """
        zcat {input} | awk '{{ $4=""; $0=$1 OFS $2 OFS $3 OFS $5 OFS $6; if (1==2) print "chr"$0; else print $0 }}' OFS="\t" | bgzip > {output.modified}
        tabix -p bed {output.modified}
        """

rule frag_length_intervals:
    input:
        bed = rules.process_and_modify.output.modified,
        bins = BINS_FILE
    output:
        os.path.join(OUTPUT_DIR, "{sample}.fraglen.500kb.bed")
    shell:
        "finaletoolkit frag-length-intervals -w 1 -o {output} -v {input.bed} {input.bins}"

rule coverage:
    input:
        bed = rules.process_and_modify.output.modified,
        bins = BINS_FILE
    output:
        os.path.join(OUTPUT_DIR, "{sample}.coverage.500kb.bed")
    shell:
        "finaletoolkit coverage -w 1 -o {output} -v {input.bed} {input.bins}"

rule interval_end_motifs:
    input:
        bed = rules.process_and_modify.output.modified,
        ref = REF_GENOME,
        bins = BINS_FILE
    output:
        os.path.join(OUTPUT_DIR, "{sample}.endmotifs.500kb.bed")
    shell:
        "finaletoolkit interval-end-motifs -o {output} -q 30 -w 1 -v {input.bed} {input.ref} {input.bins}"

rule interval_mds:
    input:
        rules.interval_end_motifs.output
    output:
        os.path.join(OUTPUT_DIR, "{sample}.mds.500kb.bed")
    shell:
        "finaletoolkit interval-mds {input} {output}"

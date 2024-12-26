# Snakefile

import os
from glob import glob

# Configuration
config = {
    "bins_file": "/jet/home/rbandaru/ravi/headneck/500kb.bins",
    "frag_bed_gz_pattern": "/jet/home/rbandaru/ravi/headneck/processed_data/*.hg38.frag.modified.bed.gz",
    "two_bit_file": "/jet/home/rbandaru/ravi/headneck/hg38.analysisSet.2bit",
    "output_pattern": "/jet/home/rbandaru/ravi/headneck/processed_data/{sample}.500kb.gc.bins"
}

# Find all input files
SAMPLES = [os.path.basename(f).split('.')[0] for f in glob(config["frag_bed_gz_pattern"])]

rule all:
    input:
        expand(config["output_pattern"], sample=SAMPLES)

rule compute_gc_content:
    input:
        bins = config["bins_file"],
        frag_bed = lambda wildcards: glob(config["frag_bed_gz_pattern"].replace("*", wildcards.sample))[0],
        two_bit = config["two_bit_file"]
    output:
        gc_bins = config["output_pattern"]
    script:
        "gc_content_script.py"

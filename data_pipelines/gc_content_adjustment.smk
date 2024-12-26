# Snakefile

import os
from glob import glob

# Configuration
config = {
    "input_pattern": "/jet/home/rbandaru/ravi/headneck/processed_data/*.coverage.500kb.bed",
    "output_pattern": "/jet/home/rbandaru/ravi/headneck/processed_data/{sample}.coverage.500kb_GCadjusted.bed"
}

# Find all input files
SAMPLES = [os.path.basename(f).split('.')[0] for f in glob(config["input_pattern"])]

rule all:
    input:
        expand(config["output_pattern"], sample=SAMPLES)

rule adjust_gc_content:
    input:
        coverage_bed = "/jet/home/rbandaru/ravi/headneck/processed_data/{sample}.coverage.500kb.bed",
        gc_bins = "/jet/home/rbandaru/ravi/headneck/processed_data/{sample}.500kb.gc.bins"
    output:
        adjusted_bed = config["output_pattern"]
    script:
        "gc_content_adjustment_script.py"
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def combine_gc_content(coverage_bed, gc_bins):
    # Load the .bed file with no headers and specific column names
    df_bed = pd.read_csv(coverage_bed, header=None, sep='\t', names=['chr', 'start', 'end', 'name', 'coverage'])
    
    # Load the .bins file with no headers and specific column names
    df_bins = pd.read_csv(gc_bins, header=None, sep='\t', names=['chr', 'start', 'end', 'gc_content'])
    
    # Convert columns to the same type to avoid merge issues
    df_bed['start'] = df_bed['start'].astype(str)
    df_bed['end'] = df_bed['end'].astype(str)
    df_bins['start'] = df_bins['start'].astype(str)
    df_bins['end'] = df_bins['end'].astype(str)
    
    # Merge on the "chr", "start", and "end" columns
    merged_df = pd.merge(df_bed, df_bins, on=['chr', 'start', 'end'], how='left')
    
    # Combine the gc_content values (sum them in case of multiple entries)
    combined_gc_content = merged_df.groupby(['chr', 'start', 'end', 'name', 'coverage'])['gc_content'].sum().reset_index()
    
    # Convert 'coverage' and 'gc_content' columns to float
    combined_gc_content['coverage'] = combined_gc_content['coverage'].astype(float)
    combined_gc_content['gc_content'] = combined_gc_content['gc_content'].astype(float)
    
    return combined_gc_content

def apply_lowess_correction(df, span=0.75):
    # Sort the dataframe by GC content
    df_sorted = df.sort_values('gc_content')
    
    # Apply LOWESS regression
    lowess_result = lowess(df_sorted['coverage'], df_sorted['gc_content'], frac=span)
    
    # Extract the smoothed values
    smoothed_coverage = lowess_result[:, 1]
    
    # Calculate residuals
    residuals = df_sorted['coverage'] - smoothed_coverage
    
    # Add back the median coverage
    median_coverage = df['coverage'].median()
    adjusted_coverage = residuals + median_coverage
    
    # Create a new dataframe with the adjusted coverage
    df_adjusted = df_sorted.copy()
    df_adjusted['adjusted_coverage'] = adjusted_coverage
    
    return df_adjusted.sort_index()

def process_sample(coverage_bed, gc_bins, output_file):
    df = combine_gc_content(coverage_bed, gc_bins)
    df_adjusted = apply_lowess_correction(df)
    output_df = df_adjusted[['chr', 'start', 'end', 'name', 'adjusted_coverage']]
    output_df = output_df.rename(columns={'adjusted_coverage': 'coverage'})
    output_df.to_csv(output_file, sep='\t', header=False, index=False)
    
    print(f"Adjusted BED file written to: {output_file}")

if __name__ == "__main__":
    process_sample(snakemake.input.coverage_bed, snakemake.input.gc_bins, snakemake.output.adjusted_bed)
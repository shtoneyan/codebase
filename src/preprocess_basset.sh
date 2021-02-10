#!/bin/sh

o_prefix=$1
sample_beds_path=$2
genome_path=$3
seq_size=$4
chrom_size_file=$5
output_dir=$6

act_path="${o_prefix}_act.txt"
headers_path="${o_prefix}_headers.txt"
mkdir $6
echo ***
echo Starting preprocess_features.py
echo Making DNA sequence chunks of length $seq_size
/home/shush/codebase/src/preprocess_features.py -y -m 200 -s $seq_size -o $o_prefix -c $chrom_size_file $sample_beds_path
echo Done with running preprocess_features.py
echo ***
echo Starting bedtools to convert bed to fa
#genome at /home/shush/prelim_analysis/hg38.fa
bedtools getfasta -fi $genome_path -bed "${o_prefix}.bed" -s -fo "${o_prefix}.fa"
echo Done converting to fa
echo ***
echo Running seq_hdf5.py
/home/shush/codebase/src/seq_hdf5.py -t 0.1 -v 0.1 $o_prefix.fa $act_path $o_prefix.h5 $headers_path
echo Done converting to fa to h5
echo ***
echo PIPELINE COMPLETED

#!/bin/sh

/home/shush/codebase/src/preprocess_basset.sh /home/shush/codebase/preprocess/basset/er \
      /home/shush/codebase/download_data/basset/sample_beds.txt \
      /home/shush/genomes/hg19.fa \
      1000 \
      /home/shush/genomes/human.hg19.genome \
      /home/shush/codebase/preprocess/basset/er


# o_prefix=$1
# sample_beds_path=$2
# genome_path=$3
# seq_size=$4
# chrom_size_file=$5
# output_dir=$6

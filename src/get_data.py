#!/usr/bin/env python

import os
import pandas as pd
import urllib.request
import sys

def make_directory(path):
    """Short summary.

    Parameters
    ----------
    path : Full path to the directory

    """

    if not os.path.isdir(path):
        os.mkdir(path)
        print("Making directory: " + path)
    else:
        print("Directory already exists!")



def make_label(df_row):

    '''
    Add label for each row selected:
    Assay_
    Experiment target (IF DNASE REPLACE nan WITH DNA)_
    Biosample term name_
    Experiment accession_
    File accession
    '''
    if df_row['Assay'].iloc[0]=='DNase-seq':
        df_row['Experiment target'] = 'DNA'
    label_list = [str(c.iloc[0]) for c in [df_row['Assay'], df_row['Experiment target'], df_row['Biosample term name'],
                         df_row['Experiment accession'], df_row['File accession']]]
    return('_'.join(label_list).replace(" ", "-"))

def download_bed(df_row, output_dir, download_on_fly):
    url = df_row['File download URL'].iloc[0]
    output_path = os.path.join(output_dir, url.split('/')[-1])
    if download_on_fly:
        urllib.request.urlretrieve(url, output_path)
    return(output_path, url)

def process(filtered_row, output_dir, download_on_fly):
    # get the label
    label = make_label(filtered_row)
    # get the output_dir & download bed
    output_path, url = download_bed(filtered_row, output_dir, download_on_fly)
    # return 'label \t output path' to be added to a txt file
    return('\t'.join([label,output_path]), url)

def process_priority(c_name, df, txt_lines, output_dir,url_txt, download_on_fly):
    """Process a df selection (1 or 2 rows only) basd on a set criterion"""
    c_true = df['Output type'] == c_name
    if any(c_true):
        found_c = True
        #preferentially take c_name output type files if any present
        if sum(c_true)==1:
            bed_text, url = process(df[c_true].iloc[[0]], output_dir, download_on_fly)
            txt_lines.append(bed_text)
            url_txt.append(url)
        elif sum(c_true)==2:
            bed_text, url = process(df[c_true].iloc[[0]], output_dir, download_on_fly)
            txt_lines.append(bed_text)
            url_txt.append(url)
            bed_text, url = process(df[c_true].iloc[[1]], output_dir, download_on_fly)
            txt_lines.append(bed_text)
            url_txt.append(url)
        else:
            pass
    else:
        found_c = False
    return found_c


def filter_dnase(df, txt_lines, output_dir, url_txt, download_on_fly):
    '''
    df = one DNase-seq experiment dataframe with bed files only and genome filtered
    txt_lines = list of lines of label and path of the corresponding file
    '''
    c1 = 'peaks'
    c1_found = process_priority(c1, df, txt_lines, output_dir,url_txt, download_on_fly)

def filter_hist(df, txt_lines, output_dir, url_txt, download_on_fly):
    c1 = 'replicated peaks'
    c2 = 'pseudo-replicated peaks'
    c1_found = process_priority(c1, df, txt_lines, output_dir, url_txt, download_on_fly)
    if not c1_found:
        c2_found = process_priority(c2, df, txt_lines, output_dir, url_txt, download_on_fly)

def filter_tf(df, txt_lines, output_dir, url_txt, download_on_fly):
    c1 = 'conservative IDR thresholded peaks'
    c2 = 'optimal IDR thresholded peaks'
    c3 = 'pseudoreplicated IDR thresholded peaks'
    c1_found = process_priority(c1, df, txt_lines, output_dir, url_txt, download_on_fly)
    if not c1_found:
        c2_found = process_priority(c2, df, txt_lines, output_dir, url_txt, download_on_fly)
        if not c2_found:
            process_priority(c3, df, txt_lines, output_dir, url_txt, download_on_fly)

def main():
    download_on_fly = False
    base_dir = sys.argv[1]
    files_path = os.path.join(base_dir, 'files.txt')
    with open(files_path, "r") as file:
        metadata_url = file.readline()[1:-2] #remove " before and after url
    print("Downloading metadata.tsv for the project")
    metadata_path = os.path.join(base_dir, 'metadata.tsv')
    urllib.request.urlretrieve(metadata_url, metadata_path)
    metadata = pd.read_csv(metadata_path ,sep='\t')
    metadata['Experiment accession'] = metadata['Experiment accession']+'_'+metadata['Biological replicate(s)']

    assay_groups = metadata.groupby(by='Assay')

    assay_filter_dict = {'TF ChIP-seq':filter_tf, 'Histone ChIP-seq':filter_hist, 'DNase-seq':filter_dnase}
    all_lines = []
    url_txt = []
    out_dir = os.path.join(base_dir, "bedfiles")
    make_directory(out_dir)
    for assay, filter_func in assay_filter_dict.items():
        try:
            assay_df = assay_groups.get_group(assay)
            assay_df = assay_df[(assay_df['File assembly'] == 'GRCh38')
                     & (assay_df['File Status'] == 'released')
                     & (assay_df['File type'] == 'bed')
                     & (assay_df['File format type']!='bed3+')]
            ass_exp_groups = assay_df.groupby(by='Experiment accession')
            print("Processing {} {} experiments".format(len(ass_exp_groups), assay))
            for exp_name, df in ass_exp_groups:
                filter_func(df, all_lines, out_dir, url_txt, download_on_fly)
        except KeyError:
            print('No {} bed files found!'.format(assay))
    with open(os.path.join(base_dir, 'sample_beds.txt'), 'w') as f:
        for item in all_lines:
            f.write("%s\n" % item)
    if not download_on_fly:
        with open(os.path.join(base_dir, 'urls.txt'), 'w') as f:
            for item in url_txt:
                f.write("%s\n" % item.strip())
    print(len(url_txt))

# ################################################################################
# # __main__
################################################################################
if __name__ == '__main__':
    main()

#!/usr/bin/env python
import os, sys

def main():
    data_path = sys.argv[1] # dir to output files in
    filename = 'deepsea_train_bundle.v0.9.tar.gz'
    file_path = os.path.join(data_path, filename)
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    uncompressed_data_dir = os.path.join(data_path, 'deepsea_train')
    if not os.path.isdir(uncompressed_data_dir):
        if not os.path.isfile(file_path):
            print('Downloading DeepSea dataset')
            os.system('wget http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz -O {}'.format(file_path))
        print('decompressing DeepSea dataset')
        os.system('tar xzvf {} -C {} '.format(file_path, data_path))


if __name__ == '__main__':
  main()

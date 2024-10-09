import os
import numpy
import pandas as pd
from urllib import request
import shutil
import zipfile
import kaggle
import gzip


DATA_DIR = 'data'


NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    # 'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    # 'lending-club': 'wordsforthewise/lending-club',
    # 'gmc': 'GiveMeSomeCredit',
    # 'dutch': 'https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/dutch.csv',
    # 'bank': 'https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/bank-full.csv',
}

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):

    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')

def download_from_repo(name):

    print(f'Start processing dataset {name} from repo.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        url = NAME_URL_DICT_UCI[name]
        # request.urlretrieve(url, f'{save_dir}/{name}.csv')
        df = pd.read_csv(url)
        df.to_csv(f'{save_dir}/{name}.csv')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
    else:
        print('Aready downloaded.')


def download_kaggle(name):
    print(f'Start processing dataset {name} from Kaggle.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        kaggle.api.authenticate()

        if name == 'gmc':
            kaggle.api.competition_download_files(url, path=save_dir)
            print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
            unzip_file(f'{save_dir}/{url}.zip', save_dir)
        else:
            kaggle.api.dataset_download_files(url, path=save_dir, unzip=False)
            print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
            unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')


if __name__ == '__main__':
    for name in NAME_URL_DICT_UCI.keys():
        if name in ['lending-club', 'gmc']:
            download_kaggle(name)
        elif name in ['bank', 'dutch']:
            download_from_repo(name)
        else:
            download_from_uci(name)
    
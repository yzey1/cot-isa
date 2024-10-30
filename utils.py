# load data
import os
import yaml
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


config = yaml.load(open("config.yaml", 'r'), Loader=yaml.FullLoader)

def get_data(data_fname):
    data_path = os.path.join(config['data_path'], data_fname)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
        data.rename(columns={"raw_texts": "sentence", "raw_aspect_terms": "target", "labels": "label", "implicits": "implicit"}, inplace=True)
        data = data[["sentence", "target", "label", "implicit", "source"]]
    elif data_path.endswith('.pkl'):
        data = pkl.load(open(data_path, 'rb'))
        data = pd.DataFrame(data, columns=["sentence", "target", "label", "implicit"])
    return data


def evaluate_result(result):
    acc = accuracy_score(result['label'], result['pred'])
    f1 = f1_score(result['label'], result['pred'], average='macro')
    return acc, f1


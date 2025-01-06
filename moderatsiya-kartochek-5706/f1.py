import pandas as pd
import argparse
import json
from sklearn.metrics import f1_score


def compare_csv_files(csv_file1, csv_file2, sep="\t"):

    df1 = pd.read_csv(csv_file1, sep=sep)
    df2 = pd.read_csv(csv_file2, sep=sep)
    
    assert set(df1['image_name']) == set(df2['image_name'])
    
    df1 = df1.sort_values(by='image_name').reset_index(drop=True)
    df2 = df2.sort_values(by='image_name').reset_index(drop=True)
    
    labels1 = df1['label_id'].tolist()
    labels2 = df2['label_id'].tolist()

    f1 = f1_score(labels1, labels2)
    
    return f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--public_test_url', type=str, required=True)
    parser.add_argument('--public_prediction_url', type=str, required=True)
    parser.add_argument('--private_test_url', type=str, required=False)
    parser.add_argument('--private_prediction_url', type=str, required=False)
    args = parser.parse_args()

    public_score = compare_csv_files(args.public_test_url, args.public_prediction_url)

    private_score = None
    if args.private_test_url and args.private_prediction_url:
        private_score = compare_csv_files(args.private_test_url, args.private_prediction_url)
    print(json.dumps({
        "public_score": public_score,
        "private_score": private_score,
    }))

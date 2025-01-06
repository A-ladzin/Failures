import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def compare_csv_files(csv_file1, csv_file2, sep="\t"):

    df1 = pd.read_csv(csv_file1, sep=sep)
    df2 = pd.read_csv(csv_file2, sep=sep)
    
    assert set(df1['image_name']) == set(df2['image_name'])
    
    df1 = df1.sort_values(by='image_name').reset_index(drop=True)
    df2 = df2.sort_values(by='image_name').reset_index(drop=True)
    
    labels1 = df1['label_id'].tolist()
    labels2 = df2['label_id'].tolist()
    
    precision = precision_score(labels1, labels2)
    recall = recall_score(labels1, labels2)
    f1 = f1_score(labels1, labels2)
    
    return precision, recall, f1


if __name__ == "__main__":
    CSV_FILE_1 = './data/submission.csv'
    CSV_FILE_2 = './data/private_info/test.csv'
    
    precision, recall, f1 = compare_csv_files(CSV_FILE_1, CSV_FILE_2)
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

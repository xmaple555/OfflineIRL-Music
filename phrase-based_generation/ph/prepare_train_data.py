import pickle
import argparse

import sys
import os

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)
from vocab import Vocab

myvocab = Vocab()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len',
                        type=int,
                        default=2048,
                        help='number of tokens to predict')
    args = parser.parse_args()

    datasets = [
        {
            "name": "theorytab",
            "test_data_num": 100,
        },
        {
            "name": "wikifonia",
            "test_data_num": 20,
        },
        {
            "name": "nottingham",
            "test_data_num": 20,
        },
        {
            "name": "pop909_1",
            "test_data_num": 20,
        },
        {
            "name": "pop909_2",
            "test_data_num": 20,
        },
    ]

    train_dataset = []
    test_dataset = []

    temp_train_seqs = []
    for dataset in datasets:
        with open(f"../ph_bc/data_pkl/{dataset['name']}.pkl", "rb") as f:
            remi_seqs = pickle.load(f)

        print(f"{dataset['name']} has {len(remi_seqs)} sequences")
        count = 0
        train_data_num = len(remi_seqs) - dataset["test_data_num"]
        for remi_seq in remi_seqs:
            remi_seq = myvocab.remove_bc(remi_seq)
            preprocessed_data = myvocab.preprocessREMI(
                remi_seq,
                max_seq_len=args.max_len,
            )

            if count < train_data_num:
                for i in range(len(preprocessed_data["tgt_segments"])):
                    tgt = preprocessed_data["tgt_segments"][i]
                    train_dataset.append({
                        "tgt": tgt,
                    })

                temp_train_seqs.append(remi_seq)
            else:
                for i in range(len(preprocessed_data["tgt_segments"])):
                    tgt = preprocessed_data["tgt_segments"][i]
                    test_dataset.append({
                        "tgt": tgt,
                    })
            count += 1

    for test_data in test_dataset:
        if test_data in train_dataset:
            test_dataset.remove(test_data)
            print(f"Removed duplicate entry in test dataset")

    with open(f"data_pkl/train_{args.max_len}.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open(f"data_pkl/test_{args.max_len}.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

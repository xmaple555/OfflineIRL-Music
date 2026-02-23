import pickle
from tqdm import tqdm
import argparse

import sys
import os

from inference import inference

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from vocab import Vocab
from reward_experiment import *
from model import Generator
from utils import set_seed

myvocab = Vocab()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument('--sample_num', type=int, default=600)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--log_name', type=str)
    parser.add_argument('--max_len',
                        type=int,
                        default=2048,
                        help='number of tokens to predict')
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--use_pkl', action='store_true')
    args = parser.parse_args()

    set_seed(0)
    if args.mode == "experiment1":
        generator = Generator(myvocab.n_tokens).cuda()
        checkpoint = torch.load(args.ckpt)
        try:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        except:
            generator = Generator(829).cuda()
            generator.load_state_dict(checkpoint['generator_state_dict'])

        experiments = [
            RewardExperimentPhBc(f'text_dir/{args.log_name}.txt'),
            RewardExperimentPhBc(
                f'text_dir/{args.log_name}_Non-Repeating-A.txt'),
            RewardExperimentPhBc(
                f'text_dir/{args.log_name}_Repeating-A.txt')
        ]

        with open("../ph_bc/data_pkl/experiment_input1.pkl", "rb") as f:
            input1 = pickle.load(f)

        if args.use_pkl:
            with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl", "rb") as f:
                sampled_remi_seqs = pickle.load(f)
        else:
            sampled_remi_seqs = []

        for i in tqdm(range(args.sample_num)):
            input_data = input1[i % len(input1)]

            if not args.use_pkl:
                while True:
                    sampled_remi_seq = inference(
                        model=generator,
                        myvocab=myvocab,
                        max_len=args.max_len,
                        input_seq=input_data["input_seq"],
                        input_phrase_labels=input_data["phrase_labels"],
                        t=args.t,
                        p=args.p)
                    if sampled_remi_seq is not None:
                        break

                sampled_remi_seqs.append({
                    "name": input_data["name"],
                    "seq": sampled_remi_seq
                })
            else:
                sampled_remi_seq = sampled_remi_seqs[i]["seq"]

            experiments[0].reward_experiment_fun(sampled_remi_seq)

            if input_data["phrase_labels"][0] == input_data["phrase_labels"][1]:
                experiments[2].reward_experiment_fun(sampled_remi_seq)
            else:
                experiments[1].reward_experiment_fun(sampled_remi_seq)

        if not args.use_pkl:
            with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl",
                    "wb") as handle:
                pickle.dump(sampled_remi_seqs,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    elif args.mode == "experiment2":
        generator = Generator(myvocab.n_tokens).cuda()
        checkpoint = torch.load(args.ckpt)
        try:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        except:
            generator = Generator(829).cuda()
            generator.load_state_dict(checkpoint['generator_state_dict'])

        experiment = MelodyRepetitionExperimentPhBc(
            f'text_dir/{args.log_name}_reward_melody_repetition.txt')

        with open("../ph_bc/data_pkl/experiment_input1.pkl", "rb") as f:
            input1 = pickle.load(f)

        if args.use_pkl:
            with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl",
                      "rb") as f:
                sampled_remi_seqs = pickle.load(f)
        else:
            sampled_remi_seqs = []

        for i in tqdm(range(args.sample_num)):
            input_data = input1[i % len(input1)]

            if not args.use_pkl:
                while True:
                    sampled_remi_seq = inference(
                        model=generator,
                        myvocab=myvocab,
                        max_len=args.max_len,
                        input_seq=input_data["input_seq"],
                        input_phrase_labels=input_data["phrase_labels"],
                        t=args.t,
                        p=args.p)
                    if sampled_remi_seq is not None:
                        break

                sampled_remi_seqs.append({
                    "name": input_data["name"],
                    "seq": sampled_remi_seq
                })
            else:
                sampled_remi_seq = sampled_remi_seqs[i]["seq"]

            experiment.reward_experiment_fun(sampled_remi_seq)

        if not args.use_pkl:
            with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl",
                      "wb") as handle:
                pickle.dump(sampled_remi_seqs,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)



    elif args.mode == "experiment3":
        with open("data_pkl/experiment_input1.pkl", "rb") as f:
            input1 = pickle.load(f)

        with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl",
                    "rb") as f:
            sampled_remi_seqs = pickle.load(f)
  
        experiment = MelodyRepetitionExperimentPhBc(
            f'text_dir/{args.log_name}_melody_repetition.txt')
        

        for i in tqdm(range(len(sampled_remi_seqs))):
            seq = sampled_remi_seqs[i]["seq"]
            experiment.reward_experiment_fun(seq)


        temp_file_name = '/tmp/' + str(uuid.uuid4()) + '.txt'
        temp_experiment = RewardExperimentPhBc(temp_file_name)


        for i in tqdm(range(len(sampled_remi_seqs))):
            sampled_remi_seq = sampled_remi_seqs[i]["seq"]
            temp_experiment.reward_experiment_fun(sampled_remi_seq)

        rewards = np.array([
            temp_experiment.rewards["melody"],
            temp_experiment.rewards["chord_progression"]
        ])
        rewards = rewards.transpose()

        rewards = (rewards - rewards.mean(axis=0)) / rewards.std(axis=0)
        rewards = rewards.mean(axis=1)

        rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())


        top_num = 1
        experiment = MelodyRepetitionExperimentPhBc(
            f'text_dir/{args.log_name}_melody_repetition_top-{top_num}.txt')
        

        for i in tqdm(range(len(input1))):
            seqs_2 =[]
            rewards_2 = []
            j = 0 + i
            while j <= len(sampled_remi_seqs) - 1:
                seqs_2.append(sampled_remi_seqs[j]["seq"])
                rewards_2.append(rewards[j])
                j += len(input1)

            top_idx = np.argsort(rewards_2)[-top_num:][::-1]
            for k in top_idx:
                experiment.reward_experiment_fun(seqs_2[k])

        os.remove(temp_file_name)


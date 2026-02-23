import wandb
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import debugpy

import sys
import os

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(grandparent_dir, 'src')
sys.path.append(src_dir)
import music_data
from vocab import Vocab
from utils import *
from model import ChordModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--warmup_step',
                        type=int,
                        default=200,
                        help='upper epoch limit')
    parser.add_argument('--lr_min',
                        type=float,
                        default=1e-5,
                        help='minimum learning rate during annealing')
    parser.add_argument('--clip',
                        type=float,
                        default=0.5,
                        help='gradient clipping')
    parser.add_argument('--max_step',
                        type=int,
                        default=500000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    if args.debug:
        debugpy.listen(("localhost", 5678))

    set_seed(args.seed)

    myvocab = Vocab()

    assert args.mode in ["ph_bc", "no_ph_bc", "ph_bc_only_pop909"]

    if args.wandb:
        wandb.login()
        wandb.init(project="chord_model_train")

    train_dataset = music_data.ChordDataset(
        pkl_path=f"{args.mode}/data_pkl/chord_train.pkl",)

    test_dataset = music_data.ChordDataset(
        pkl_path=f"{args.mode}/data_pkl/chord_test.pkl",)

    sampled_dataset = music_data.ChordDataset(
        pkl_path=f"{args.mode}/data_pkl/chord_sampled.pkl",)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=2,
                             shuffle=False,
                             num_workers=4)

    sampled_loader = DataLoader(dataset=sampled_dataset,
                                batch_size=2,
                                shuffle=False,
                                num_workers=4)

    model = ChordModel(myvocab.n_tokens).cuda()
    model.train()

    n_parameters = network_paras(model)
    print('n_parameters: {:,}'.format(n_parameters))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.max_step,
                                                           eta_min=args.lr_min)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    train_step = 0
    start_epoch = 0

    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_step = checkpoint["train_step"]
        start_epoch = checkpoint["epoch"] + 1

    for epoch in tqdm(range(start_epoch, args.epoch_num)):
        total_loss = 0
        total_acc = 0
        model.train()
        steps = 0
        for batch_idx, data in tqdm(enumerate(train_loader),
                                    leave=False,
                                    total=len(train_loader)):

            data = {key: value.cuda() for key, value in data.items()}

            optimizer.zero_grad()

            data["tgt_msk"] = data["tgt_msk"].bool()

            tgt_input_msk = data["tgt_msk"][:, :-1]
            tgt_output_msk = data["tgt_msk"][:, 1:]

            if torch.sum(~tgt_output_msk) == 0:
                continue

            data["tgt"] = data["tgt"]

            fullsong_input = data["tgt"][:, :-1]
            fullsong_output = data["tgt"][:, 1:]

            output = model(tgt=fullsong_input,)

            loss = loss_fn(output.view(-1, myvocab.n_tokens),
                           fullsong_output.reshape(-1))

            predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)

            correct = predict.eq(fullsong_output.reshape(-1))

            correct = torch.sum(correct *
                                (~tgt_output_msk).reshape(-1).float()).item()

            correct = correct / \
                torch.sum((~tgt_output_msk).reshape(-1).float()).item()

            total_acc += correct

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                scheduler.step()

            total_loss += loss.item()
            train_step += 1
            steps += 1

        total_train_acc = total_acc / steps
        total_train_loss = total_loss / steps

        model.eval()
        total_loss = 0
        total_acc = 0
        steps = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = {key: value.cuda() for key, value in data.items()}

                data["tgt_msk"] = data["tgt_msk"].bool()

                tgt_input_msk = data["tgt_msk"][:, :-1]
                tgt_output_msk = data["tgt_msk"][:, 1:]

                if torch.sum(~tgt_output_msk) == 0:
                    continue

                fullsong_input = data["tgt"][:, :-1]
                fullsong_output = data["tgt"][:, 1:]

                output = model(tgt=fullsong_input,)

                loss = loss_fn(output.view(-1, myvocab.n_tokens),
                               fullsong_output.reshape(-1))

                predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)

                correct = predict.eq(fullsong_output.reshape(-1))

                correct = torch.sum(
                    correct * (~tgt_output_msk).reshape(-1).float()).item()

                correct = correct / \
                    torch.sum((~tgt_output_msk).reshape(-1).float()).item()

                total_acc += correct

                total_loss += loss.item()
                steps += 1

        total_test_acc = total_acc / steps
        total_test_loss = total_loss / steps

        total_loss = 0
        total_acc = 0
        steps = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(sampled_loader):
                data = {key: value.cuda() for key, value in data.items()}

                data["tgt_msk"] = data["tgt_msk"].bool()

                tgt_input_msk = data["tgt_msk"][:, :-1]
                tgt_output_msk = data["tgt_msk"][:, 1:]

                if torch.sum(~tgt_output_msk) == 0:
                    continue

                fullsong_input = data["tgt"][:, :-1]
                fullsong_output = data["tgt"][:, 1:]

                output = model(tgt=fullsong_input,)

                loss = loss_fn(output.view(-1, myvocab.n_tokens),
                               fullsong_output.reshape(-1))

                predict = output.view(-1, myvocab.n_tokens).argmax(dim=-1)

                correct = predict.eq(fullsong_output.reshape(-1))

                correct = torch.sum(
                    correct * (~tgt_output_msk).reshape(-1).float()).item()

                correct = correct / \
                    torch.sum((~tgt_output_msk).reshape(-1).float()).item()

                total_acc += correct

                total_loss += loss.item()
                steps += 1

        total_sampled_acc = total_acc / steps
        total_sampled_loss = total_loss / steps

        if args.wandb:
            wandb.log({
                "train_loss": total_train_loss,
                "train_acc": total_train_acc,
                "test_loss": total_test_loss,
                "test_acc": total_test_acc,
                "sampled_loss": total_sampled_loss,
                "sampled_acc": total_sampled_acc,
                "loss_diff": total_test_loss - total_sampled_loss,
            })

        if epoch % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                "train_step": train_step,
            }
            torch.save(checkpoint,
                       f'{args.mode}/ckpt/chord_model/chord_model_{epoch}.pt')

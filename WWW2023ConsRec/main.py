import sys
import os
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import random
import torch.optim as optim
import numpy as np
from metrics import evaluate as evaluate_metrics
from metrics_after import evaluate as evaluate_metrics_after
from model import ConsRec
from datetime import datetime
import argparse
import time
from dataloader import GroupDataset
# from tensorboardX import SummaryWriter
import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def get_local_time():
    return datetime.now().strftime('%b-%d-%Y-%H-%M-%S')


def training(train_loader, epoch, type_m="group"):
    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    for batch_id, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(running_device), pi_ni[:, 1].to(running_device)

        if type_m == 'user':
            pos_prediction = train_model(None, user_input, pos_items_input)
            neg_prediction = train_model(None, user_input, neg_items_input)
        else:
            pos_prediction = train_model(user_input, None, pos_items_input)
            neg_prediction = train_model(user_input, None, neg_items_input)

        optimizer.zero_grad()
        if args.loss_type == "BPR":
            loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        else:
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)

        losses.append(loss)
        loss.backward()
        optimizer.step()

    logging.info(
        f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMERa2011]", default="Mafengwo")
    parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

    parser.add_argument("--layers", type=int, help="# HyperConv & OverlapConv layers", default=3)
    parser.add_argument("--emb_dim", type=int, help="User/Item/Group embedding dimensions", default=32)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--topK", type=list, default=[1, 5, 10])

    parser.add_argument("--epoch", type=int, default=100, help="# running epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--loss_type", type=str, default="BPR")

    args = parser.parse_args()
    set_seed(args.seed)

    logfilename = f'{args.dataset}-{get_local_time()}.log'
    os.makedirs('log', exist_ok=True)
    logfilepath = os.path.join('log', logfilename)

    file_handler = logging.FileHandler(logfilepath, mode='a', encoding='utf8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)

    logging.info('= ' * 20)
    logging.info('## Starting Time: %s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(args)

    # writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # writer = SummaryWriter(writer_dir)

    running_device = torch.device(args.device)

    # Load dataset
    user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
    dataset = GroupDataset(user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset, seed=args.seed)
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    logging.info(f" #Users {num_users}, #Items {num_items}, #Groups {num_groups}\n")

    user_hg, item_hg, full_hg = dataset.user_hyper_graph.to(running_device), dataset.item_hyper_graph.to(
        running_device), dataset.full_hg.to(running_device)
    overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)
    light_gcn_graph = dataset.light_gcn_graph.to(running_device)

    # Prepare model
    train_model = ConsRec(num_users, num_items, num_groups, args, user_hg, item_hg,
                          full_hg, overlap_graph, running_device, light_gcn_graph, dataset.num_group_net_items)
    train_model.to(running_device)

    for epoch_id in range(args.epoch):
        train_model.train()
        group_loss = training(
            dataset.get_group_dataloader(args.batch_size, epoch=epoch_id),
            epoch_id,
            "group",
        )
        # writer.add_scalar("Group Loss", group_loss, epoch_id)
        user_loss = training(
            dataset.get_user_dataloader(args.batch_size, epoch=epoch_id),
            epoch_id,
            "user",
        )

        group_hits, group_ndcgs = evaluate_metrics(
            train_model,
            dataset.group_test_ratings,
            dataset.group_test_negatives,
            running_device,
            args.topK,
            'group',
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch_id}] [group evaluate] [metrics]",
            log_fn=logging.info,
        )

        logging.info(f"[Epoch {epoch_id}] Group [metrics], Hit@{args.topK}: {group_hits}, NDCG@{args.topK}: {group_ndcgs}")

        group_hits_after, group_ndcgs_after = evaluate_metrics_after(
            train_model,
            dataset.group_test_ratings,
            dataset.group_test_negatives,
            running_device,
            args.topK,
            'group',
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch_id}] [group evaluate] [metrics_after]",
            log_fn=logging.info,
        )

        logging.info(f"[Epoch {epoch_id}] Group [metrics_after], Hit@{args.topK}: {group_hits_after}, NDCG@{args.topK}: {group_ndcgs_after}")
        # writer.add_scalars(f'Group/Hit@{args.topK}', {str(args.topK[i]): hits[i] for i in range(len(args.topK))}, epoch_id)
        # writer.add_scalars(f'Group/NDCG@{args.topK}', {str(args.topK[i]): ndcgs[i] for i in range(len(args.topK))}, epoch_id)

        user_hits, user_ndcgs = evaluate_metrics(
            train_model,
            dataset.user_test_ratings,
            dataset.user_test_negatives,
            running_device,
            args.topK,
            'user',
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch_id}] [user evaluate] [metrics]",
            log_fn=logging.info,
        )

        logging.info(f"[Epoch {epoch_id}] User [metrics], Hit@{args.topK}: {user_hits}, NDCG@{args.topK}: {user_ndcgs}")

        user_hits_after, user_ndcgs_after = evaluate_metrics_after(
            train_model,
            dataset.user_test_ratings,
            dataset.user_test_negatives,
            running_device,
            args.topK,
            'user',
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch_id}] [user evaluate] [metrics_after]",
            log_fn=logging.info,
        )

        logging.info(f"[Epoch {epoch_id}] User [metrics_after], Hit@{args.topK}: {user_hits_after}, NDCG@{args.topK}: {user_ndcgs_after}")
        # writer.add_scalars(f'User/Hit@{args.topK}', {str(args.topK[i]): hrs[i] for i in range(len(args.topK))}, epoch_id)
        # writer.add_scalars(f'User/NDCG@{args.topK}', {str(args.topK[i]): ndcgs[i] for i in range(len(args.topK))}, epoch_id)

    logging.info('## Finishing Time: %s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info('= ' * 20)
    logging.info("Done!")

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
from model import AlignGroup
from datetime import datetime
from utils import get_local_time
import argparse
import time
from dataloader import GroupDataset
# from tensorboardX import SummaryWriter
import logging
from sklearn import manifold

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


def training(train_loader, epoch, type_m="group", group_member_dict=None):
    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    for batch_id, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(running_device), pi_ni[:, 1].to(running_device)

        optimizer.zero_grad()
        if type_m == 'user':
            loss, _ = train_model(None, user_input, pos_items_input, neg_items_input, None, 'train')
        else:
            members = [torch.LongTensor(group_member_dict[group_id]).to(running_device) for group_id in list(u.cpu().numpy())]
            loss, _ = train_model(user_input, None, pos_items_input, neg_items_input, members, 'train')

        losses.append(loss)
        loss.backward()
        optimizer.step()

    string = (f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    logging.info(string)
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011]", default="CAMRa2011") # CAMRa2011, Mafengwo
    parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

    parser.add_argument("--layers", type=int, help="# HyperConv & OverlapConv layers", default=3) # 3 is the best
    parser.add_argument("--emb_dim", type=int, help="User/Item/Group embedding dimensions", default=32)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--topK", type=list, default=[1, 5, 10])

    parser.add_argument("--epoch", type=int, default=100, help="# running epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=float, default=512)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--predictor", type=str, default="MLP")
    parser.add_argument("--loss_type", type=str, default="BPR")
    parser.add_argument("--temp", type=list, default=[0.8])
    parser.add_argument("--cl_weight", type=list, default=[0.1])



    args = parser.parse_args()
    set_seed(args.seed)
    logfilename = '{}-{}.log'.format(args.dataset, get_local_time())
    project_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilepath = os.path.join(log_dir, logfilename)

    file_handler = logging.FileHandler(logfilepath, mode='a', encoding='utf8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    ## t-sne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    logging.info('= ' * 20)
    msg = ('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(msg)
    logging.info(args)

    running_device = torch.device(args.device)

    # Load dataset
    user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
    dataset = GroupDataset(user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset, seed=args.seed)
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    logging.info(" #Users {}, #Items {}, #Groups {}\n".format(num_users, num_items, num_groups))

    user_hg, item_hg, full_hg = dataset.user_hyper_graph.to(running_device), dataset.item_hyper_graph.to(
        running_device), dataset.full_hg.to(running_device)
    overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)

    group_member_dict = dataset.group_member_dict

    # Prepare model
    logging.info('██Dataset: \t' + args.dataset)

    for idx in range(len(args.cl_weight) * len(args.temp)):
        cl_info = args.cl_weight[idx % (len(args.cl_weight))]
        temp = args.temp[idx // (len(args.cl_weight))]
        logging.info(f"Idx = {idx+1} / {len(args.cl_weight) * len(args.temp)}, cl_weight = {cl_info}, temp = {temp}: ")
        # Prepare model
        train_model = AlignGroup(num_users, num_items, num_groups, args, user_hg, item_hg,
                               full_hg, overlap_graph, running_device, cl_info, temp)
        train_model.to(running_device)

        for epoch_id in range(args.epoch):
            train_model.train()
            g_loader = dataset.get_group_dataloader(args.batch_size, epoch=epoch_id)

            group_loss = training(g_loader, epoch_id, "group", group_member_dict)

            user_loss = training(
                dataset.get_user_dataloader(args.batch_size, epoch=epoch_id),
                epoch_id,
                "user",
                group_member_dict,
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
            logging.info(
                "[Epoch {}] Group [metrics], Hit@{}: {}, NDCG@{}: {}".format(
                    epoch_id, args.topK, group_hits, args.topK, group_ndcgs
                )
            )

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
            logging.info(
                "[Epoch {}] Group [metrics_after], Hit@{}: {}, NDCG@{}: {}".format(
                    epoch_id, args.topK, group_hits_after, args.topK, group_ndcgs_after
                )
            )

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
            logging.info(
                "[Epoch {}] User [metrics], Hit@{}: {}, NDCG@{}: {}".format(
                    epoch_id, args.topK, user_hits, args.topK, user_ndcgs
                )
            )

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
            logging.info(
                "[Epoch {}] User [metrics_after], Hit@{}: {}, NDCG@{}: {}".format(
                    epoch_id, args.topK, user_hits_after, args.topK, user_ndcgs_after
                )
            )

            # tsne
            # g_rep, u_rep, i_rep = train_model.group_embedding.weight.clone(), train_model.user_embedding.weight.clone(), train_model.item_embedding.weight.clone()
            # g_rep = g_rep.cpu().data.numpy()
            # u_rep = u_rep.cpu().data.numpy()
            # i_rep = i_rep.cpu().data.numpy()
            #
            # g_rep = tsne.fit_transform(g_rep)
            # u_rep = tsne.fit_transform(u_rep)
            # i_rep = tsne.fit_transform(i_rep)
            # np.savetxt('./embd/wo/g/' + str(epoch_id) + '.csv', g_rep, delimiter=',')
            # np.savetxt('./embd/wo/u/' + str(epoch_id) + '.csv', u_rep, delimiter=',')
            # np.savetxt('./embd/wo/i/' + str(epoch_id) + '.csv', i_rep, delimiter=',')
        msg = ('## Finishing Time: {}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logging.info(msg)
        logging.info('= ' * 20)
        logging.info("Done!")

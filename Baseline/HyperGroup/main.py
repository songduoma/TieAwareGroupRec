import torch
import random
import torch.optim as optim
import dataloader
import model
import metrics
import argparse
from datetime import datetime
import numpy as np
import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def get_local_time():
    return datetime.now().strftime('%b-%d-%Y-%H-%M-%S')


def init_logger(dataset):
    os.makedirs("log", exist_ok=True)
    logfilename = f"{dataset}-{get_local_time()}.log"
    logfilepath = os.path.join("log", logfilename)

    file_handler = logging.FileHandler(logfilepath, mode='a', encoding='utf8')
    file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)


def training(train_loader, epoch, type_m='group'):
    lr = args.lr

    optimizer = optim.Adam(rec_model.parameters(), lr)

    losses = []

    for _, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(device)
        pos_item_input, neg_item_input = pi_ni[:,0].to(device), pi_ni[:,1].to(device)

        if type_m == "user":
            pos_prediction = rec_model(None, user_input, pos_item_input)
            neg_prediction = rec_model(None, user_input, neg_item_input)
        else:
            pos_prediction = rec_model(user_input, None, pos_item_input)
            neg_prediction = rec_model(user_input, None, neg_item_input)

        rec_model.zero_grad()
        loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        losses.append(loss)
        loss.backward()
        optimizer.step()
    logging.info(f"Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:0")

# lr - 1e-3
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epoch", type=int, default=100)

parser.add_argument("--dataset", type=str, default="Mafengwo")
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--num_negatives", type=int, default=2)
parser.add_argument("--topK", type=list, default=[1, 5, 10])

args = parser.parse_args()
set_seed(args.seed)
init_logger(args.dataset)

logging.info('= ' * 20)
logging.info('## Starting Time: %s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logging.info(args)

device = torch.device(args.device)

dataset = dataloader.GroupDataset(num_negatives=args.num_negatives, dataset=args.dataset)
num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
logging.info(f"# Users {num_users}, # Items {num_items}, # Groups {num_groups}")

member_feat, member_mask, hypergraph = dataset.membership.to(device), dataset.member_mask.to(device), \
                                       dataset.hyper_graph.to(device)

rec_model = model.HyperGroup(num_users, num_items, num_groups, hypergraph, member_feat, member_mask,
                             dataset.group_member_dict, args.emb_dim, args.k)
rec_model.to(device)

for epoch_id in range(args.epoch):
    rec_model.train()
    training(dataset.get_group_dataloader(args.batch_size), epoch_id, "group")
    training(dataset.get_user_dataloader(args.batch_size), epoch_id, "user")

    metric_evaluators = [
        ("metrics", metrics.evaluate),
        ("metrics_first", metrics.evaluate_first),
        ("metrics_last", metrics.evaluate_last),
        ("metrics_tie_aware", metrics.evaluate_tie_aware),
    ]
    for metric_name, evaluator in metric_evaluators:
        hits, ndcgs = evaluator(
            rec_model,
            dataset.group_test_ratings,
            dataset.group_test_negatives,
            device,
            args.topK,
            'group',
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch_id}] [group evaluate] [{metric_name}]",
            log_fn=logging.info,
        )
        logging.info(f"[Epoch {epoch_id}] Group [{metric_name}], Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

        hits, ndcgs = evaluator(
            rec_model,
            dataset.user_test_ratings,
            dataset.user_test_negatives,
            device,
            args.topK,
            'user',
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch_id}] [user evaluate] [{metric_name}]",
            log_fn=logging.info,
        )
        logging.info(f"[Epoch {epoch_id}] User [{metric_name}], Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
    logging.info("")


logging.info('## Finishing Time: %s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logging.info('= ' * 20)
logging.info("Done!")

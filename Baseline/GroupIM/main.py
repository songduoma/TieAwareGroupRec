import argparse
import time
import os
import torch
from datetime import datetime
import logging

import model
import metrics
import dataloader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Mafengwo")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--lambda_mi", type=float, default=0.2)
parser.add_argument("--drop_ratio", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--pretrain_epoch", default=5, type=int)

parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--topK", type=list, default=[1, 5, 10])

args = parser.parse_args()
init_logger(args.dataset)
logging.info('= ' * 20)
logging.info('## Starting Time: %s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logging.info(args)
logging.info("")

dataset = dataloader.GroupDataset(dataset=args.dataset)
device = torch.device(args.device)

rec_model = model.GroupIM(dataset.num_items, args.emb_dim, drop_ratio=args.drop_ratio, lambda_mi=args.lambda_mi)
rec_model.to(device)

optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr, weight_decay=args.wd)
metric_evaluators = [
    ("metrics", metrics.group_model_leave_one_test, metrics.user_model_leave_one_test),
    ("metrics_first", metrics.group_model_leave_one_test_first, metrics.user_model_leave_one_test_first),
    ("metrics_last", metrics.group_model_leave_one_test_last, metrics.user_model_leave_one_test_last),
    ("metrics_tie_aware", metrics.group_model_leave_one_test_tie_aware, metrics.user_model_leave_one_test_tie_aware),
]

# Pretrain User-Item
logging.info("Pre-training model on user-item interactions...")
optimizer_ui = torch.optim.Adam(rec_model.parameters(), lr=0.01, weight_decay=args.wd)
for epoch in range(args.pretrain_epoch):
    rec_model.train()

    ui_data = dataset.user_pretrain_dataloader(args.batch_size)
    train_ui_loss = 0.0
    start_time = time.time()
    for user_items in ui_data:
        # user_items (batch_size, n_items)
        user_items = user_items[0].to(device)
        # user_logits (batch_size, n_items), user_embeds (batch_size, emb_dim)
        user_logits, _ = rec_model.user_preference_encoder.pretrain_forward(user_items)
        user_loss = rec_model.user_loss(user_logits, user_items)

        optimizer_ui.zero_grad()
        user_loss.backward()
        optimizer_ui.step()
        train_ui_loss += user_loss.item()
    elapsed = time.time() - start_time
    logging.info(f"[Epoch {epoch+1}] UI, time {elapsed:.2f}s, loss {train_ui_loss/len(ui_data):.4f}")
    for metric_name, _, user_evaluator in metric_evaluators:
        hits, ndcgs = user_evaluator(
            rec_model,
            dataset,
            dataset.user_test_ratings,
            dataset.user_test_negatives,
            device,
            k_list=args.topK,
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch + 1}] [user evaluate] [{metric_name}]",
            log_fn=logging.info,
        )
        logging.info(f"[Epoch {epoch+1}] User [{metric_name}], Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
logging.info("Pre-train finish!\n")

logging.info("Pre-training model on user-group mutual information...")
rec_model.group_predictor.weight.data = rec_model.user_preference_encoder.user_predictor.weight.data
# Pretrain Mutual Information
for epoch in range(args.pretrain_epoch):
    rec_model.train()
    mi_epoch_loss = 0.0
    mi_epoch_start = time.time()
    gi_data = dataset.group_dataloader(args.batch_size)

    for (group_users, group_masks, user_items, corrupt_user_items, _) in gi_data:
        group_users = group_users.to(device)
        group_masks = group_masks.to(device)
        user_items = user_items.to(device)
        corrupt_user_items = corrupt_user_items.to(device)
        _, group_embeds, _ = rec_model(group_masks, user_items)
        obs_user_embed = rec_model.user_preference_encoder(user_items).detach()
        corrupt_user_embed = rec_model.user_preference_encoder(corrupt_user_items).detach()

        score_obs = rec_model.discriminator(group_embeds, obs_user_embed)
        score_corrupt = rec_model.discriminator(group_embeds, corrupt_user_embed)
        # print(score_obs)

        mi_loss = rec_model.discriminator.mi_loss(score_obs, group_masks, score_corrupt, device=device)
        optimizer.zero_grad()
        mi_loss.backward()
        optimizer.step()
        mi_epoch_loss += mi_loss.item()
    elapsed = time.time() - mi_epoch_start
    logging.info(f"[Epoch {epoch + 1}] MI, time {elapsed:.2f}s, loss {mi_epoch_loss / len(gi_data):.4f}")
logging.info("Mutual Information pre-train finish!\n")

logging.info("Training model on group-item interactions...")
for epoch in range(args.epoch):
    epoch_start_time = time.time()
    rec_model.train()
    train_epoch_loss = 0.0

    gi_data = dataset.group_dataloader(args.batch_size)
    for batch_data in gi_data:
        batch_data = [x.to(device) for x in batch_data]
        (group_users, group_masks, user_items, corrupt_user_items, group_items) = batch_data
        group_logits, group_embeds, scores_ug = rec_model(group_masks, user_items)
        group_loss = rec_model.loss(group_logits, group_embeds, scores_ug, group_masks, group_items, user_items,
                                    corrupt_user_items, device)
        optimizer.zero_grad()
        group_loss.backward()
        optimizer.step()
        train_epoch_loss += group_loss.item()
    elapsed = time.time() - epoch_start_time
    logging.info(f"[Epoch {epoch+1}] GI, time {elapsed:.2f}s group-item loss: {train_epoch_loss/len(gi_data):.5f}")
    for metric_name, group_evaluator, user_evaluator in metric_evaluators:
        hits, ndcgs = group_evaluator(
            rec_model,
            dataset,
            dataset.group_test_ratings,
            dataset.group_test_negatives,
            device,
            k_list=args.topK,
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch + 1}] [group evaluate] [{metric_name}]",
            log_fn=logging.info,
        )
        logging.info(f"[Epoch {epoch + 1}] Group [{metric_name}], Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
        hits, ndcgs = user_evaluator(
            rec_model,
            dataset,
            dataset.user_test_ratings,
            dataset.user_test_negatives,
            device,
            k_list=args.topK,
            print_pred_score_stats=True,
            pred_score_stats_prefix=f"[Epoch {epoch + 1}] [user evaluate] [{metric_name}]",
            log_fn=logging.info,
        )
        logging.info(f"[Epoch {epoch + 1}] User [{metric_name}], Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

logging.info("")
logging.info('## Finishing Time: %s', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logging.info('= ' * 20)
logging.info("Done!")

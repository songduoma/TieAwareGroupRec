import torch
import numpy as np
import math


def _log_pred_score_stats(pred_score, prefix="", log_fn=None):
    max_pred_score = float(np.max(pred_score))

    row_max = np.max(pred_score, axis=1, keepdims=True)
    row_max_count = np.sum(pred_score == row_max, axis=1)
    avg_top_tie_count = float(np.mean(row_max_count))
    avg_top_tie_pct = float(np.mean(row_max_count / pred_score.shape[1]) * 100.0)
    top_tie_sample_pct = float(np.mean(row_max_count > 1) * 100.0)

    prefix = f"{prefix} " if prefix else ""
    if log_fn is None:
        log_fn = print
    log_fn(
        f"{prefix}pred_score max: {max_pred_score:.6f}, "
        f"num of top-score tie (avg per sample): {avg_top_tie_count:.4f}"
    )
    log_fn(
        f"{prefix}top-score tie ratio (avg per sample): {avg_top_tie_pct:.4f}%, "
        f"samples with tied top (>1 item): {top_tie_sample_pct:.4f}%"
    )


def get_hit_k(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return round(hit, 5)


def get_ndcg_k(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)


def evaluate(
    model,
    test_ratings,
    test_negatives,
    device,
    k_list,
    type_m='group',
    group_member_dict={},
    group_item_dict={},
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    """Evaluate the performance (HitRatio, NDCG) of top-K recommendation"""
    model.eval()
    hits, ndcgs = [], []
    user_test, item_test = [], []
    members_u = [torch.LongTensor(group_member_dict[group_id]).to(device) for group_id in
              list(group_member_dict.keys())]

    members_i = [torch.LongTensor(group_item_dict[group_id]).to(device) for group_id in
             list(group_item_dict.keys())]

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        items = [rating[1]]
        items.extend(test_negatives[idx])

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

    users_var = torch.LongTensor(user_test).to(device)
    items_var = torch.LongTensor(item_test).to(device)

    bsz = len(test_ratings)
    item_len = len(test_negatives[0]) + 1

    users_var = users_var.view(-1)
    items_var = items_var.view(-1)

    if type_m == 'group':
        _, predictions = model(None, users_var, items_var, items_var, members_u, members_i, "eval", 'group')
    elif type_m == 'user':
        _, predictions = model(users_var, None, items_var, items_var, None, None, "eval", 'user')

    predictions = torch.reshape(predictions, (bsz, item_len))

    pred_score = predictions.data.cpu().numpy()
    if print_pred_score_stats:
        _log_pred_score_stats(pred_score, pred_score_stats_prefix, log_fn)
    # print(pred_score[:10, ])
    pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        hits.append(get_hit_k(pred_rank, k))
        ndcgs.append(get_ndcg_k(pred_rank, k))

    return hits, ndcgs

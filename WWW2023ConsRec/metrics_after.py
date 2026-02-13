import torch
import numpy as np


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


def _get_tie_stats(pred_score):
    """
    Compute tie-aware rank interval for the positive item (index 0).
    For each row:
      lowest possible rank = (#items with strictly higher score) + 1
      highest possible rank = (#items with higher or equal score)
    """
    pos_score = pred_score[:, [0]]
    num_higher = np.sum(pred_score > pos_score, axis=1)
    num_equal = np.sum(pred_score == pos_score, axis=1)
    return num_higher, num_equal


def get_hit_k(pred_score, k):
    """
    Unbiased Hit@K under uniform random tie-breaking.
    """
    num_higher, num_equal = _get_tie_stats(pred_score)
    lowest_rank = num_higher + 1
    highest_rank = num_higher + num_equal

    hit_prob = np.where(
        k < lowest_rank,
        0.0,
        np.where(k >= highest_rank, 1.0, (k - num_higher) / num_equal),
    )
    return round(float(np.mean(hit_prob)), 5)


def get_ndcg_k(pred_score, k):
    """
    Unbiased NDCG@K under uniform random tie-breaking.
    """
    num_higher, num_equal = _get_tie_stats(pred_score)
    lowest_rank = num_higher + 1
    highest_rank = num_higher + num_equal

    discounts = 1.0 / np.log2(np.arange(1, k + 1) + 1)
    prefix_discount = np.concatenate(([0.0], np.cumsum(discounts)))

    start = np.maximum(lowest_rank, 1)
    end = np.minimum(highest_rank, k)

    in_top_k = start <= end
    expected_dcg = np.zeros_like(num_higher, dtype=np.float64)
    expected_dcg[in_top_k] = (
        prefix_discount[end[in_top_k]] - prefix_discount[start[in_top_k] - 1]
    ) / num_equal[in_top_k]

    return np.round(np.mean(expected_dcg), decimals=5)


def evaluate(
    model,
    test_ratings,
    test_negatives,
    device,
    k_list,
    type_m='group',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    """Evaluate the performance (HitRatio, NDCG) of top-K recommendation"""
    model.eval()
    hits, ndcgs = [], []
    user_test, item_test = [], []

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
        predictions = model(users_var, None, items_var)
    elif type_m == 'user':
        predictions = model(None, users_var, items_var)

    predictions = torch.reshape(predictions, (bsz, item_len))

    pred_score = predictions.data.cpu().numpy()
    if print_pred_score_stats:
        _log_pred_score_stats(pred_score, pred_score_stats_prefix, log_fn)
    for k in k_list:
        hits.append(get_hit_k(pred_score, k))
        ndcgs.append(get_ndcg_k(pred_score, k))

    return hits, ndcgs

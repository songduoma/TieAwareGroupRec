import math
import torch
import torch.nn as nn
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


def _get_tie_stats(pred_score, gt_idx=0):
    pos_score = pred_score[:, [gt_idx]]
    num_higher = np.sum(pred_score > pos_score, axis=1)
    num_equal = np.sum(pred_score == pos_score, axis=1)
    return num_higher, num_equal


def get_hit_k(pred_rank, k, gt_idx=0):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == gt_idx)
    hit = hit / pred_rank_k.shape[0]
    return round(hit, 5)


def get_ndcg_k(pred_rank, k, gt_idx=0):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == gt_idx:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    return np.round(np.mean(ndcgs), decimals=5)


def get_hit_k_tie_aware(pred_score, k, gt_idx=0):
    num_higher, num_equal = _get_tie_stats(pred_score, gt_idx=gt_idx)
    lowest_rank = num_higher + 1
    highest_rank = num_higher + num_equal
    hit_prob = np.where(
        k < lowest_rank,
        0.0,
        np.where(k >= highest_rank, 1.0, (k - num_higher) / num_equal),
    )
    return round(float(np.mean(hit_prob)), 5)


def get_ndcg_k_tie_aware(pred_score, k, gt_idx=0):
    num_higher, num_equal = _get_tie_stats(pred_score, gt_idx=gt_idx)
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


def _build_pred_score(model: nn.Module, test_ratings, test_negatives, device, type_m, gt_position="first"):
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))
    gt_idx = 0 if gt_position == "first" else pred_score.shape[1] - 1

    for idx in range(len(test_ratings)):
        test_rating = test_ratings[idx]
        if gt_position == "first":
            test_items = [test_rating[1]] + test_negatives[idx]
        else:
            test_items = test_negatives[idx] + [test_rating[1]]
        test_user = test_rating[0]

        test_users = np.full(len(test_items), test_user)
        users_var = torch.from_numpy(test_users).long().to(device)
        items_var = torch.LongTensor(test_items).to(device)

        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        else:
            predictions = model(None, users_var, items_var)

        pred_score[idx, :] = predictions.data.cpu().numpy().reshape(-1)

    return pred_score, gt_idx


def _evaluate_from_pred_score(
    pred_score,
    k_list,
    gt_idx,
    stable_sort=False,
    tie_aware=False,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    if print_pred_score_stats:
        _log_pred_score_stats(pred_score, pred_score_stats_prefix, log_fn)

    hits, ndcgs = [], []
    if tie_aware:
        for k in k_list:
            hits.append(get_hit_k_tie_aware(pred_score, k, gt_idx=gt_idx))
            ndcgs.append(get_ndcg_k_tie_aware(pred_score, k, gt_idx=gt_idx))
        return hits, ndcgs

    if stable_sort:
        pred_rank = np.argsort(pred_score * -1, axis=1, kind="stable")
    else:
        pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        hits.append(get_hit_k(pred_rank, k, gt_idx=gt_idx))
        ndcgs.append(get_ndcg_k(pred_rank, k, gt_idx=gt_idx))
    return hits, ndcgs


def evaluate(
    model: nn.Module,
    test_ratings,
    test_negatives,
    device,
    k_list,
    type_m='group',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    model.eval()
    pred_score, gt_idx = _build_pred_score(
        model,
        test_ratings,
        test_negatives,
        device,
        type_m,
        gt_position="first",
    )
    return _evaluate_from_pred_score(
        pred_score,
        k_list,
        gt_idx,
        stable_sort=False,
        tie_aware=False,
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def evaluate_first(
    model: nn.Module,
    test_ratings,
    test_negatives,
    device,
    k_list,
    type_m='group',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    model.eval()
    pred_score, gt_idx = _build_pred_score(
        model,
        test_ratings,
        test_negatives,
        device,
        type_m,
        gt_position="first",
    )
    return _evaluate_from_pred_score(
        pred_score,
        k_list,
        gt_idx,
        stable_sort=True,
        tie_aware=False,
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def evaluate_last(
    model: nn.Module,
    test_ratings,
    test_negatives,
    device,
    k_list,
    type_m='group',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    model.eval()
    pred_score, gt_idx = _build_pred_score(
        model,
        test_ratings,
        test_negatives,
        device,
        type_m,
        gt_position="last",
    )
    return _evaluate_from_pred_score(
        pred_score,
        k_list,
        gt_idx,
        stable_sort=True,
        tie_aware=False,
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def evaluate_tie_aware(
    model: nn.Module,
    test_ratings,
    test_negatives,
    device,
    k_list,
    type_m='group',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    model.eval()
    pred_score, gt_idx = _build_pred_score(
        model,
        test_ratings,
        test_negatives,
        device,
        type_m,
        gt_position="first",
    )
    return _evaluate_from_pred_score(
        pred_score,
        k_list,
        gt_idx,
        stable_sort=False,
        tie_aware=True,
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )

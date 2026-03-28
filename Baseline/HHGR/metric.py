import numpy as np
import math
import torch


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
    hit = hit / pred_rank.shape[0]
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


def _build_pred_score(
    model,
    test_ratings,
    test_negatives,
    device,
    user_embeds=None,
    group_embeds=None,
    type_m='group',
    gt_position="first",
):
    user_test, item_test = [], []
    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        if gt_position == "first":
            items = [rating[1]]
            items.extend(test_negatives[idx])
        else:
            items = test_negatives[idx] + [rating[1]]
        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

    users_var = torch.LongTensor(user_test).to(device)
    items_var = torch.LongTensor(item_test).to(device)
    bsz = len(test_ratings)
    item_len = len(test_negatives[0]) + 1

    users_var = users_var.view(-1)
    items_var = items_var.view(-1)

    if type_m == 'group':
        predictions = model.group_forward(users_var, items_var, user_embeds, group_embeds)
    else:
        predictions = model.user_forward(users_var, items_var)

    predictions = torch.reshape(predictions, (bsz, item_len))
    pred_score = predictions.data.cpu().numpy()
    gt_idx = 0 if gt_position == "first" else pred_score.shape[1] - 1
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
    model,
    test_ratings,
    test_negatives,
    device,
    k_list,
    user_embeds=None,
    group_embeds=None,
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
        user_embeds=user_embeds,
        group_embeds=group_embeds,
        type_m=type_m,
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
    model,
    test_ratings,
    test_negatives,
    device,
    k_list,
    user_embeds=None,
    group_embeds=None,
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
        user_embeds=user_embeds,
        group_embeds=group_embeds,
        type_m=type_m,
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
    model,
    test_ratings,
    test_negatives,
    device,
    k_list,
    user_embeds=None,
    group_embeds=None,
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
        user_embeds=user_embeds,
        group_embeds=group_embeds,
        type_m=type_m,
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
    model,
    test_ratings,
    test_negatives,
    device,
    k_list,
    user_embeds=None,
    group_embeds=None,
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
        user_embeds=user_embeds,
        group_embeds=group_embeds,
        type_m=type_m,
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

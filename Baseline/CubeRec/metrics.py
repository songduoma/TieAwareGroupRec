import numpy as np
import math
import model
import torch


def _log_pred_score_stats(pred_score, prefix="", log_fn=None):
    min_pred_score = float(np.min(pred_score))
    row_min = np.min(pred_score, axis=1, keepdims=True)
    row_min_count = np.sum(pred_score == row_min, axis=1)
    avg_top_tie_count = float(np.mean(row_min_count))
    avg_top_tie_pct = float(np.mean(row_min_count / pred_score.shape[1]) * 100.0)
    top_tie_sample_pct = float(np.mean(row_min_count > 1) * 100.0)

    prefix = f"{prefix} " if prefix else ""
    if log_fn is None:
        log_fn = print
    log_fn(
        f"{prefix}pred_score min: {min_pred_score:.6f}, "
        f"num of top-score tie (avg per sample): {avg_top_tie_count:.4f}"
    )
    log_fn(
        f"{prefix}top-score tie ratio (avg per sample): {avg_top_tie_pct:.4f}%, "
        f"samples with tied top (>1 item): {top_tie_sample_pct:.4f}%"
    )


def _get_tie_stats(pred_score, gt_idx=0):
    pos_score = pred_score[:, [gt_idx]]
    num_lower = np.sum(pred_score < pos_score, axis=1)
    num_equal = np.sum(pred_score == pos_score, axis=1)
    return num_lower, num_equal


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
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)


def get_hit_k_tie_aware(pred_score, k, gt_idx=0):
    num_lower, num_equal = _get_tie_stats(pred_score, gt_idx=gt_idx)
    lowest_rank = num_lower + 1
    highest_rank = num_lower + num_equal
    hit_prob = np.where(
        k < lowest_rank,
        0.0,
        np.where(k >= highest_rank, 1.0, (k - num_lower) / num_equal),
    )
    return round(float(np.mean(hit_prob)), 5)


def get_ndcg_k_tie_aware(pred_score, k, gt_idx=0):
    num_lower, num_equal = _get_tie_stats(pred_score, gt_idx=gt_idx)
    lowest_rank = num_lower + 1
    highest_rank = num_lower + num_equal

    discounts = 1.0 / np.log2(np.arange(1, k + 1) + 1)
    prefix_discount = np.concatenate(([0.0], np.cumsum(discounts)))

    start = np.maximum(lowest_rank, 1)
    end = np.minimum(highest_rank, k)
    in_top_k = start <= end
    expected_dcg = np.zeros_like(num_lower, dtype=np.float64)
    expected_dcg[in_top_k] = (
        prefix_discount[end[in_top_k]] - prefix_discount[start[in_top_k] - 1]
    ) / num_equal[in_top_k]

    return np.round(np.mean(expected_dcg), decimals=5)


def _build_pred_score(rec_model: model.CubeRec, test_ratings, test_negatives, device, mode='user', gt_position="first"):
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))
    gt_idx = 0 if gt_position == "first" else pred_score.shape[1] - 1

    if mode == 'user':
        users, items = rec_model.compute()
    else:
        users, items, all_centers, all_offsets = rec_model.compute_all()

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        test_user = rating[0]
        if gt_position == "first":
            test_items = [rating[1]] + test_negatives[idx]
        else:
            test_items = test_negatives[idx] + [rating[1]]

        if mode == 'user':
            test_user_emb = users[test_user].detach().cpu().numpy()
            test_items_emb = items[test_items].detach().cpu().numpy()
            score = np.sqrt(np.sum(np.asarray(test_user_emb - test_items_emb) ** 2, axis=1))
        else:
            test_centers = all_centers[[test_user]].detach().cpu().numpy()
            test_offsets = all_offsets[[test_user]].detach().cpu().numpy()
            test_centers = np.repeat(test_centers, len(test_items), axis=0)
            test_offsets = np.repeat(test_offsets, len(test_items), axis=0)
            score = rec_model.gi_scores(
                torch.FloatTensor(test_centers).to(device),
                torch.FloatTensor(test_offsets).to(device),
                torch.LongTensor(test_items).to(device),
                items,
            )
            score = score.detach().cpu().numpy().reshape(-1)
        pred_score[idx, :] = score
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

    test_hits, test_ndcgs = [], []
    if tie_aware:
        for k in k_list:
            test_hits.append(get_hit_k_tie_aware(pred_score, k, gt_idx=gt_idx))
            test_ndcgs.append(get_ndcg_k_tie_aware(pred_score, k, gt_idx=gt_idx))
        return test_hits, test_ndcgs

    if stable_sort:
        pred_rank = np.argsort(pred_score, axis=1, kind="stable")
    else:
        pred_rank = np.argsort(pred_score, axis=1)
    for k in k_list:
        test_hits.append(get_hit_k(pred_rank, k, gt_idx=gt_idx))
        test_ndcgs.append(get_ndcg_k(pred_rank, k, gt_idx=gt_idx))
    return test_hits, test_ndcgs


def model_leave_one_test(
    rec_model: model.CubeRec,
    test_ratings,
    test_negatives,
    device,
    k_list,
    mode='user',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    rec_model.eval()
    pred_score, gt_idx = _build_pred_score(
        rec_model,
        test_ratings,
        test_negatives,
        device,
        mode=mode,
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


def model_leave_one_test_first(
    rec_model: model.CubeRec,
    test_ratings,
    test_negatives,
    device,
    k_list,
    mode='user',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    rec_model.eval()
    pred_score, gt_idx = _build_pred_score(
        rec_model,
        test_ratings,
        test_negatives,
        device,
        mode=mode,
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


def model_leave_one_test_last(
    rec_model: model.CubeRec,
    test_ratings,
    test_negatives,
    device,
    k_list,
    mode='user',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    rec_model.eval()
    pred_score, gt_idx = _build_pred_score(
        rec_model,
        test_ratings,
        test_negatives,
        device,
        mode=mode,
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


def model_leave_one_test_tie_aware(
    rec_model: model.CubeRec,
    test_ratings,
    test_negatives,
    device,
    k_list,
    mode='user',
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    rec_model.eval()
    pred_score, gt_idx = _build_pred_score(
        rec_model,
        test_ratings,
        test_negatives,
        device,
        mode=mode,
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

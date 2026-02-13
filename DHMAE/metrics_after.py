import torch
import numpy as np


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
    return np.round(np.mean(hit_prob), decimals=5)


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
    ratings,
    negatives,
    device,
    k_list,
    type_m="group",
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
):
    k_list = list(k_list)
    num_samples = len(ratings)
    hit_sums = np.zeros(len(k_list), dtype=np.float64)
    ndcg_sums = np.zeros(len(k_list), dtype=np.float64)
    max_pred_score = float("-inf")
    top_tie_count_sum = 0.0
    top_tie_ratio_sum = 0.0
    num_top_tie_samples = 0

    max_k = max(k_list)
    discounts = 1.0 / np.log2(np.arange(1, max_k + 1) + 1)
    prefix_discount = np.concatenate(([0.0], np.cumsum(discounts)))

    for idx in range(num_samples):
        rating = ratings[idx]
        items = [rating[1]]
        items.extend(negatives[idx])

        users_var = torch.LongTensor(np.full(len(items), rating[0])).to(device)
        items_var = torch.LongTensor(np.array(items)).to(device)

        predictions = model(users_var, items_var, type_m).squeeze()
        pred_score = predictions.data.cpu().numpy().reshape(1, -1)
        row_max = float(np.max(pred_score))
        row_max_count = int(np.sum(pred_score == row_max))
        row_candidate_count = int(pred_score.shape[1])

        if row_max > max_pred_score:
            max_pred_score = row_max

        top_tie_count_sum += row_max_count
        top_tie_ratio_sum += row_max_count / row_candidate_count
        if row_max_count > 1:
            num_top_tie_samples += 1

        num_higher, num_equal = _get_tie_stats(pred_score)
        num_higher = int(num_higher[0])
        num_equal = int(num_equal[0])

        lowest_rank = num_higher + 1
        highest_rank = num_higher + num_equal

        for i, k in enumerate(k_list):
            if k < lowest_rank:
                hit_prob = 0.0
            elif k >= highest_rank:
                hit_prob = 1.0
            else:
                hit_prob = (k - num_higher) / num_equal
            hit_sums[i] += hit_prob

            start = max(lowest_rank, 1)
            end = min(highest_rank, k)
            if start <= end:
                ndcg_sums[i] += (prefix_discount[end] - prefix_discount[start - 1]) / num_equal

    hits_k = np.round(hit_sums / num_samples, decimals=5).tolist()
    ndcgs_k = np.round(ndcg_sums / num_samples, decimals=5).tolist()

    if print_pred_score_stats and num_samples > 0:
        prefix = f"{pred_score_stats_prefix} " if pred_score_stats_prefix else ""
        avg_top_tie_count = top_tie_count_sum / num_samples
        avg_top_tie_pct = (top_tie_ratio_sum / num_samples) * 100.0
        top_tie_sample_pct = (num_top_tie_samples / num_samples) * 100.0
        print(
            f"{prefix}pred_score max: {max_pred_score:.6f}, "
            f"num of top-score tie (avg per sample): {avg_top_tie_count:.4f}"
        )
        print(
            f"{prefix}top-score tie ratio (avg per sample): {avg_top_tie_pct:.4f}%, "
            f"samples with tied top (>1 item): {top_tie_sample_pct:.4f}%"
        )

    return hits_k, ndcgs_k

import math
import numpy as np
import model
import dataloader


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
                ndcgs[user] = math.log(2) / math.log(j+2)
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


def _variant_to_settings(variant):
    if variant == "metrics_first":
        return "first", True, False
    if variant == "metrics_last":
        return "last", True, False
    if variant == "metrics_tie_aware":
        return "first", False, True
    return "first", False, False


def _build_test_item(rating, negatives, gt_position="first"):
    if gt_position == "first":
        return [rating[1]] + negatives
    return negatives + [rating[1]]


def _user_model_leave_one_test_variant(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    variant="metrics",
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    rec_model.eval()
    gt_position, stable_sort, tie_aware = _variant_to_settings(variant)
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))

    test_users = [rating[0] for rating in test_ratings]
    user_feat = dataset.user_pretrain_dataloader(batch_size=256, test_mode=True).to(device)

    test_user_feat = user_feat[test_users]
    test_logits, _ = rec_model.user_preference_encoder.pretrain_forward(test_user_feat)
    test_logits = test_logits.detach().cpu().numpy()

    for idx in range(len(test_negatives)):
        test_item = _build_test_item(test_ratings[idx], test_negatives[idx], gt_position=gt_position)
        pred_score[idx, :] = test_logits[idx, test_item]

    gt_idx = 0 if gt_position == "first" else pred_score.shape[1] - 1
    return _evaluate_from_pred_score(
        pred_score,
        k_list,
        gt_idx,
        stable_sort=stable_sort,
        tie_aware=tie_aware,
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def _group_model_leave_one_test_variant(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    variant="metrics",
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    rec_model.eval()
    gt_position, stable_sort, tie_aware = _variant_to_settings(variant)
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))

    test_groups = [rating[0] for rating in test_ratings]
    _, all_group_mask, all_user_items, _ = dataset.group_dataloader(batch_size=256, test_mode=True)
    all_group_mask = all_group_mask.to(device)
    all_user_items = all_user_items.to(device)

    test_group_logits, _, _ = rec_model(all_group_mask[test_groups], all_user_items[test_groups])
    test_group_logits = test_group_logits.detach().cpu().numpy()

    for idx in range(len(test_negatives)):
        test_item = _build_test_item(test_ratings[idx], test_negatives[idx], gt_position=gt_position)
        pred_score[idx, :] = test_group_logits[idx, test_item]

    gt_idx = 0 if gt_position == "first" else pred_score.shape[1] - 1
    return _evaluate_from_pred_score(
        pred_score,
        k_list,
        gt_idx,
        stable_sort=stable_sort,
        tie_aware=tie_aware,
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def user_model_leave_one_test(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _user_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def user_model_leave_one_test_first(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _user_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics_first",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def user_model_leave_one_test_last(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _user_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics_last",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def user_model_leave_one_test_tie_aware(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _user_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics_tie_aware",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def group_model_leave_one_test(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _group_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def group_model_leave_one_test_first(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _group_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics_first",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def group_model_leave_one_test_last(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _group_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics_last",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )


def group_model_leave_one_test_tie_aware(
    rec_model: model.GroupIM,
    dataset: dataloader.GroupDataset,
    test_ratings,
    test_negatives,
    device,
    k_list=None,
    print_pred_score_stats=False,
    pred_score_stats_prefix="",
    log_fn=None,
):
    return _group_model_leave_one_test_variant(
        rec_model,
        dataset,
        test_ratings,
        test_negatives,
        device,
        k_list=k_list,
        variant="metrics_tie_aware",
        print_pred_score_stats=print_pred_score_stats,
        pred_score_stats_prefix=pred_score_stats_prefix,
        log_fn=log_fn,
    )

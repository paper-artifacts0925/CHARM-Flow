"""Source-transport objectives for set-level EM-v0 and cell-level EM-v1."""

import torch
import torch.nn.functional as F


def _rbf_mmd_per_source(prediction, target):
    """Compute biased RBF-MMD for [B,K,S,G] predictions and [B,S,G] targets."""
    target = target.unsqueeze(1).expand(-1, prediction.shape[1], -1, -1)
    bsz, num_sources, set_size, gene_dim = prediction.shape
    pred = prediction.reshape(bsz * num_sources, set_size, gene_dim).float()
    truth = target.reshape(bsz * num_sources, set_size, gene_dim).float()

    with torch.no_grad():
        joined = torch.cat([pred, truth], dim=1)
        distances = torch.cdist(joined, joined).square()
        positive = distances[distances > 0]
        bandwidth = positive.median() if positive.numel() else distances.new_tensor(1.0)
        bandwidth = bandwidth.clamp_min(1e-6)

    k_xx = torch.exp(-torch.cdist(pred, pred).square() / (2.0 * bandwidth))
    k_yy = torch.exp(-torch.cdist(truth, truth).square() / (2.0 * bandwidth))
    k_xy = torch.exp(-torch.cdist(pred, truth).square() / (2.0 * bandwidth))
    mmd = k_xx.mean(dim=(-2, -1)) + k_yy.mean(dim=(-2, -1))
    mmd = mmd - 2.0 * k_xy.mean(dim=(-2, -1))
    return mmd.reshape(bsz, num_sources).clamp_min(0)


def _source_diversity(source_tokens):
    """Penalize correlated source representations only in latent space."""
    if source_tokens.shape[-2] <= 1:
        return source_tokens.new_zeros(())
    normalized = F.normalize(source_tokens.float(), dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    eye = torch.eye(
        similarity.shape[-1],
        device=similarity.device,
        dtype=torch.bool,
    )
    off_diagonal = similarity.masked_select(~eye)
    return off_diagonal.square().mean()


def _transport_path_separation(prediction, source_start, margin=0.0):
    """Penalize source-path cosine similarity above the configured margin."""
    if source_start is None or prediction.shape[1] <= 1:
        return prediction.new_zeros(())
    paths = (prediction.float() - source_start.float()).flatten(start_dim=2)
    paths = F.normalize(paths, dim=-1)
    similarity = paths @ paths.transpose(-1, -2)
    eye = torch.eye(
        similarity.shape[-1],
        device=similarity.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    off_diagonal = similarity.masked_select(~eye)
    if float(margin) > 0:
        return F.relu(off_diagonal - float(margin)).square().mean()
    return off_diagonal.square().mean()


def _q_summary(responsibilities, source_prior, source_mask=None):
    eps = 1e-8
    entropy = -(
        responsibilities * responsibilities.clamp_min(eps).log()
    ).sum(dim=-1)
    top2 = torch.topk(
        responsibilities,
        k=min(2, responsibilities.shape[-1]),
        dim=-1,
    ).values
    gap = top2[..., 0]
    if top2.shape[-1] > 1:
        gap = gap - top2[..., 1]

    prior = source_prior.to(
        device=responsibilities.device,
        dtype=responsibilities.dtype,
    )
    if source_mask is None:
        source_mask = torch.ones_like(prior, dtype=torch.bool)
    else:
        source_mask = source_mask.to(device=prior.device, dtype=torch.bool)
    prior = torch.where(source_mask, prior.clamp_min(eps), torch.zeros_like(prior))
    prior = prior / prior.sum(dim=-1, keepdim=True)
    uniform = source_mask.to(prior.dtype)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True)
    while prior.ndim < responsibilities.ndim:
        prior = prior.unsqueeze(-2)
        uniform = uniform.unsqueeze(-2)
    kl_to_prior = (
        responsibilities
        * (responsibilities.clamp_min(eps).log() - prior.clamp_min(eps).log())
    ).sum(dim=-1)
    kl_to_uniform = (
        responsibilities
        * (responsibilities.clamp_min(eps).log() - uniform.clamp_min(eps).log())
    ).sum(dim=-1)
    return {
        "entropy_mean": entropy.mean(),
        "entropy_std": entropy.float().std(unbiased=False),
        "effective_num_sources": entropy.mean().exp(),
        "max_responsibility_mean": responsibilities.max(dim=-1).values.mean(),
        "top1_top2_gap": gap.mean(),
        "kl_q_to_prior": kl_to_prior.mean(),
        "kl_to_uniform": kl_to_uniform.mean(),
    }


def _candidate_rank_stability(candidate_loss):
    flat = candidate_loss.detach().reshape(-1, candidate_loss.shape[-1]).float()
    if flat.shape[0] <= 1 or flat.shape[-1] <= 1:
        return flat.new_tensor(1.0)
    order = flat.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    values = torch.arange(flat.shape[-1], device=flat.device, dtype=torch.float32)
    ranks.scatter_(dim=-1, index=order, src=values.expand_as(ranks))
    left = ranks[:-1] - ranks[:-1].mean(dim=-1, keepdim=True)
    right = ranks[1:] - ranks[1:].mean(dim=-1, keepdim=True)
    numerator = (left * right).sum(dim=-1)
    denominator = left.square().sum(dim=-1).sqrt() * right.square().sum(dim=-1).sqrt()
    return (numerator / denominator.clamp_min(1e-8)).mean()


def _responsibility_diagnostics(
    responsibilities,
    candidate_loss,
    source_prior,
    pre_support=None,
    support_mask=None,
    transition_lambda=1.0,
    source_mask=None,
):
    post = _q_summary(responsibilities, source_prior, source_mask)
    pre_support = responsibilities if pre_support is None else pre_support
    pre = _q_summary(pre_support, source_prior, source_mask)
    reduce_dims = tuple(range(responsibilities.ndim - 1))
    mass = responsibilities.mean(dim=reduce_dims)
    source_std = candidate_loss.float().std(dim=-1, unbiased=False)
    source_range = candidate_loss.max(dim=-1).values - candidate_loss.min(dim=-1).values
    smallest = torch.topk(
        candidate_loss,
        k=min(2, candidate_loss.shape[-1]),
        dim=-1,
        largest=False,
    ).values
    loss_gap = smallest[..., 0].new_zeros(smallest[..., 0].shape)
    if smallest.shape[-1] > 1:
        loss_gap = smallest[..., 1] - smallest[..., 0]

    if support_mask is None:
        support_mask = responsibilities > 0
    membership = support_mask.float().mean(dim=reduce_dims)
    flat_mask = support_mask.reshape(-1, support_mask.shape[-1])
    if flat_mask.shape[0] > 1:
        intersection = (flat_mask[:-1] & flat_mask[1:]).sum(dim=-1).float()
        union = (flat_mask[:-1] | flat_mask[1:]).sum(dim=-1).float()
        jaccard = (intersection / union.clamp_min(1.0)).mean()
    else:
        jaccard = responsibilities.new_tensor(1.0)
    top1 = responsibilities.argmax(dim=-1)
    top1_frequency = F.one_hot(
        top1,
        num_classes=responsibilities.shape[-1],
    ).float().mean(dim=reduce_dims)

    diagnostics = {
        **post,
        "candidate_loss_mean": candidate_loss.mean(),
        "candidate_loss_std_across_sources": source_std.mean(),
        "candidate_loss_range_across_sources": source_range.mean(),
        "candidate_top1_top2_loss_gap": loss_gap.mean(),
        "candidate_source_rank_stability": _candidate_rank_stability(candidate_loss),
        "responsibility_mass_per_source": mass,
        "responsibility_mass_min": mass.min(),
        "responsibility_mass_max": mass.max(),
        "pre_support": pre,
        "post_support": post,
        "current_support_k": support_mask.sum(dim=-1).max().float(),
        "topk_membership_frequency_per_source": membership,
        "topk_set_jaccard_across_cells": jaccard,
        "top1_source_frequency": top1_frequency,
        "top1_assignments": top1.detach(),
        "support_transition_lambda": responsibilities.new_tensor(float(transition_lambda)),
    }
    return diagnostics


def source_transport_loss(
    prediction,
    target,
    source_prior,
    source_tokens,
    temperature=0.1,
    lambda_balance=0.1,
    lambda_mmd=0.1,
    lambda_diversity=0.01,
    use_em=True,
    em_version="v0_set_level",
    prior_scale=1.0,
    responsibility_topk=None,
    lambda_path_sep=0.0,
    path_sep_margin=0.0,
    source_start=None,
    source_centroid=None,
    prediction_type="x0",
    support_policy=None,
    source_mask=None,
    router_logits=None,
    router_alpha=1.0,
    lambda_router=0.5,
    lambda_usage=0.05,
    em_topk=None,
    em_candidate_score="diffusion_mse",
    delta_score_weight=0.0,
    delta_center_genes=True,
):
    """Compute either legacy set-level EM-v0 or cell-level online EM-v1."""
    from src.models.source_em import (
        apply_support_policy,
        compute_candidate_loss,
        compute_delta_signature_loss,
        compute_em_m_step_loss,
        compute_soft_em_responsibility,
        source_responsibilities,
        uniform_responsibilities_like,
    )

    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError("prediction must be [B,K,S,G] and target [B,S,G]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes are incompatible")

    pre_support = None
    support_mask = None
    transition_lambda = 1.0
    router_loss = prediction.new_zeros(())
    usage_loss = prediction.new_zeros(())
    delta_signature_loss = prediction.new_zeros(
        prediction.shape[0], prediction.shape[2], prediction.shape[1],
        dtype=torch.float32,
    )
    estep_candidate_score = None
    if em_version == "v0_set_level":
        diffusion_loss = compute_candidate_loss(
            prediction,
            target,
            prediction_type=prediction_type,
            reduce_gene_dim=True,
            reduce_cell_dim=True,
        )
        estep_candidate_score = diffusion_loss
        responsibilities = source_responsibilities(
            estep_candidate_score.detach(),
            source_prior,
            temperature=temperature,
        )
        if not use_em:
            responsibilities = uniform_responsibilities_like(responsibilities)
        em_loss = compute_em_m_step_loss(diffusion_loss, responsibilities)
        live_q = source_responsibilities(
            diffusion_loss,
            source_prior,
            temperature=temperature,
        )
        q_mean = live_q.mean(dim=0).clamp_min(1e-8)
        uniform = torch.full_like(q_mean, 1.0 / q_mean.numel())
        balance_loss = (q_mean * (q_mean.log() - uniform.log())).sum()
        path_separation_loss = prediction.new_zeros(())
    elif em_version == "v1_cell_level_online_gem":
        diffusion_loss = compute_candidate_loss(
            prediction,
            target,
            prediction_type=prediction_type,
            reduce_gene_dim=True,
            reduce_cell_dim=False,
        )
        candidate_score_name = str(em_candidate_score)
        if candidate_score_name == "diffusion_mse":
            estep_candidate_score = diffusion_loss
        elif candidate_score_name in ("delta_signature", "diffusion_mse_delta"):
            if prediction_type != "x0":
                raise ValueError("delta-signature E-step requires prediction_type=x0")
            if source_centroid is None:
                raise ValueError("delta-signature E-step requires source_centroid")
            delta_signature_loss = compute_delta_signature_loss(
                prediction,
                target,
                source_centroid,
                source_mask=source_mask,
                center_genes=bool(delta_center_genes),
            )
            delta_score = float(delta_score_weight) * delta_signature_loss
            if candidate_score_name == "delta_signature":
                estep_candidate_score = delta_score
            else:
                estep_candidate_score = diffusion_loss.detach() + delta_score
        else:
            raise ValueError(f"unsupported em_candidate_score={candidate_score_name}")
        estep_candidate_score = estep_candidate_score.detach()

        if support_policy is None:
            responsibilities = compute_soft_em_responsibility(
                estep_candidate_score,
                source_prior,
                temperature=temperature,
                prior_scale=prior_scale,
                topk=em_topk if em_topk is not None else responsibility_topk,
                detach_candidate_loss=True,
                source_mask=source_mask,
            )
            pre_support = compute_soft_em_responsibility(
                estep_candidate_score,
                source_prior,
                temperature=temperature,
                prior_scale=prior_scale,
                topk=None,
                detach_candidate_loss=True,
                source_mask=source_mask,
            )
            support_mask = responsibilities > 0
        else:
            if source_mask is not None:
                raise ValueError("support schedules with masked sources are not supported")
            support_result = apply_support_policy(
                estep_candidate_score,
                source_prior,
                temperature=temperature,
                prior_scale=prior_scale,
                **support_policy,
            )
            pre_support = support_result["pre_support"]
            responsibilities = support_result["post_support"]
            support_mask = support_result["support_mask"]
            transition_lambda = float(support_result["transition_lambda"])
        assert responsibilities.ndim == 3
        assert responsibilities.shape[-1] == prediction.shape[1]
        if not use_em:
            responsibilities = uniform_responsibilities_like(
                responsibilities, source_mask=source_mask
            )
            support_mask = responsibilities > 0

        # ---- Router fusion (new) ----
        if router_logits is not None and support_policy is None:
            from src.models.source_router import (
                compute_router_loss,
                compute_usage_loss,
                gumbel_softmax_topk,
            )

            # cell-level q → set-level q for router matching
            set_q = responsibilities.mean(dim=1)  # (B, K)

            em_topk = int(em_topk) if em_topk is not None else None
            topk_val = em_topk if em_topk is not None else responsibilities.shape[-1]

            # Router produces its own top-k distribution
            router_q = gumbel_softmax_topk(
                router_logits,
                k=topk_val,
                temperature=1.0,
                hard=False,
                mask=source_mask,
            )

            # Fuse: weighted blend of router and EM
            alpha = max(0.0, min(1.0, float(router_alpha)))
            fused_set_q = alpha * router_q + (1.0 - alpha) * set_q
            fused_set_q = fused_set_q / fused_set_q.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # Broadcast fused set-level q back to cell-level
            responsibilities = fused_set_q.detach().unsqueeze(1).expand_as(responsibilities)
            support_mask = responsibilities > 0

            router_loss = compute_router_loss(router_logits, fused_set_q, mask=source_mask)
            usage_loss = compute_usage_loss(fused_set_q, responsibilities.shape[-1])

        em_loss = compute_em_m_step_loss(diffusion_loss, responsibilities)
        balance_loss = compute_em_m_step_loss(
            diffusion_loss,
            responsibilities,
            source_balanced=True,
        )
        path_separation_loss = _transport_path_separation(
            prediction,
            source_start,
            margin=path_sep_margin,
        )
    else:
        raise ValueError(f"unsupported em_version={em_version}")

    mmd_per_source = _rbf_mmd_per_source(prediction, target)
    mmd_responsibilities = responsibilities
    if mmd_responsibilities.ndim == 3:
        mmd_responsibilities = mmd_responsibilities.mean(dim=1)
    mmd_loss = (mmd_responsibilities * mmd_per_source).sum(dim=-1).mean()
    diversity_loss = _source_diversity(source_tokens)

    if em_version == "v0_set_level":
        total = (
            em_loss
            + float(lambda_balance) * balance_loss
            + float(lambda_mmd) * mmd_loss
            + float(lambda_diversity) * diversity_loss
        )
    else:
        total = (
            em_loss
            + float(lambda_balance) * balance_loss
            + float(lambda_mmd) * mmd_loss
            + float(lambda_path_sep) * path_separation_loss
            + float(lambda_router) * router_loss
            + float(lambda_usage) * usage_loss
        )

    diagnostics = _responsibility_diagnostics(
        responsibilities,
        diffusion_loss,
        source_prior,
        pre_support=pre_support,
        support_mask=support_mask,
        transition_lambda=transition_lambda,
        source_mask=source_mask,
    )
    return {
        "loss": total,
        "em_loss": em_loss,
        "balance_loss": balance_loss,
        "mmd_loss": mmd_loss,
        "diversity_loss": diversity_loss,
        "path_separation_loss": path_separation_loss,
        "router_loss": router_loss,
        "usage_loss": usage_loss,
        "diffusion_loss_per_source": diffusion_loss,
        "estep_candidate_score": estep_candidate_score,
        "delta_signature_loss": delta_signature_loss,
        "responsibilities": responsibilities,
        "pre_support_responsibilities": pre_support,
        "diagnostics": diagnostics,
    }


__all__ = ["source_transport_loss"]

"""Analytic detached-loss Soft-EM helpers for latent control sources."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def robust_normalize_candidate_energy(
    energy,
    source_mask=None,
    clip=5.0,
    eps=1e-4,
):
    """Normalize detached candidate energies within each cell.

    Candidate ranking matters for the E-step, while absolute denoising scales
    vary with timestep and example difficulty. Mean centering and mean
    absolute deviation scaling preserve the ranking without letting one energy
    component dominate solely because of its units.
    """
    if energy.ndim != 3:
        raise ValueError("energy must have shape [B,S,K]")
    values = energy.detach().float()
    if source_mask is None:
        mask = torch.ones(
            values.shape[0], values.shape[-1], device=values.device, dtype=torch.bool
        )
    else:
        mask = source_mask.to(device=values.device, dtype=torch.bool)
    if mask.shape != (values.shape[0], values.shape[-1]):
        raise ValueError("source_mask must have shape [B,K]")
    if not mask.any(dim=-1).all():
        raise ValueError("every sample must have at least one valid source")

    expanded_mask = mask.unsqueeze(1).expand_as(values)
    count = expanded_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    center = (values * expanded_mask).sum(dim=-1, keepdim=True) / count
    deviation = (values - center).abs()
    scale = (deviation * expanded_mask).sum(dim=-1, keepdim=True) / count
    normalized = (values - center) / scale.clamp_min(float(eps))
    if float(clip) > 0.0:
        normalized = normalized.clamp(min=-float(clip), max=float(clip))
    return torch.where(expanded_mask, normalized, torch.zeros_like(normalized))


def compute_candidate_loss(
    model_output,
    target,
    prediction_type="x0",
    reduce_gene_dim=True,
    reduce_cell_dim=False,
):
    """Return source-candidate diffusion MSE without reducing cells for EM-v1."""
    if prediction_type not in ("x0", "epsilon"):
        raise ValueError(f"unsupported prediction_type={prediction_type}")
    if model_output.ndim != 4 or target.ndim != 3:
        raise ValueError("model_output must be [B,K,S,G] and target [B,S,G]")
    if model_output.shape[0] != target.shape[0] or model_output.shape[2:] != target.shape[1:]:
        raise ValueError("model_output and target shapes are incompatible")

    loss = (model_output - target.unsqueeze(1)).square()
    if reduce_gene_dim:
        loss = loss.mean(dim=-1)
    if reduce_cell_dim:
        cell_dim = -1 if reduce_gene_dim else -2
        loss = loss.mean(dim=cell_dim)
    if reduce_gene_dim and not reduce_cell_dim:
        loss = loss.permute(0, 2, 1).contiguous()
    return loss


@torch.no_grad()
def compute_control_reliability_weights(control_sets, eps=1e-6, max_weight=20.0):
    """Estimate reliable genes from between-state signal and within-state noise."""
    if control_sets.ndim != 4:
        raise ValueError("control_sets must have shape [B,K,T,G]")
    values = control_sets.detach().float()
    centroids = values.mean(dim=2)
    between_state = centroids.var(dim=1, unbiased=False)
    within_state = values.var(dim=2, unbiased=False).mean(dim=1)
    reliability = between_state / within_state.clamp_min(float(eps))
    reliability = reliability.clamp(max=float(max_weight)).add(float(eps))
    return reliability / reliability.mean(dim=-1, keepdim=True).clamp_min(float(eps))


@torch.no_grad()
def compute_delta_signature_loss(
    prediction,
    target,
    source_centroid,
    source_mask=None,
    center_genes=True,
    source_sets=None,
    reliable_gene_weighting=False,
    reliability_max_weight=20.0,
    projection_basis=None,
    eps=1e-8,
):
    """Score every treated cell against every deterministic source cluster.

    Both observed and predicted changes are measured from the same cluster
    centroid. This yields a cell-to-cluster score [B,S,K] without constructing
    control/treated cell pairs or averaging cells into pseudobulk.
    """
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError("prediction must be [B,K,S,G] and target [B,S,G]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes are incompatible")

    batch_size, num_sources = prediction.shape[:2]
    centroid = source_centroid.to(device=prediction.device, dtype=torch.float32)
    if centroid.ndim == 2:
        centroid = centroid.unsqueeze(0).expand(batch_size, -1, -1)
    if centroid.ndim != 3 or centroid.shape[:2] != (batch_size, num_sources):
        raise ValueError("source_centroid must have shape [K,G] or [B,K,G]")
    if centroid.shape[-1] != prediction.shape[-1]:
        raise ValueError("source_centroid and prediction gene dimensions disagree")

    centroid = centroid.unsqueeze(2)
    observed_delta = target.detach().float().unsqueeze(1) - centroid
    predicted_delta = prediction.detach().float() - centroid
    if center_genes:
        observed_delta = observed_delta - observed_delta.mean(dim=-1, keepdim=True)
        predicted_delta = predicted_delta - predicted_delta.mean(dim=-1, keepdim=True)
    if reliable_gene_weighting:
        if source_sets is None:
            raise ValueError("source_sets are required for reliable gene weighting")
        reliability = compute_control_reliability_weights(
            source_sets,
            eps=eps,
            max_weight=reliability_max_weight,
        ).sqrt()[:, None, None, :]
        observed_delta = observed_delta * reliability
        predicted_delta = predicted_delta * reliability
    if projection_basis is not None:
        basis = projection_basis.detach().to(
            device=prediction.device,
            dtype=observed_delta.dtype,
        )
        if basis.ndim != 2 or basis.shape[-1] != prediction.shape[-1]:
            raise ValueError("projection_basis must have shape [D,G]")
        observed_delta = observed_delta @ basis.transpose(0, 1)
        predicted_delta = predicted_delta @ basis.transpose(0, 1)

    cosine = F.cosine_similarity(
        predicted_delta,
        observed_delta,
        dim=-1,
        eps=eps,
    )
    loss = (1.0 - cosine).permute(0, 2, 1).contiguous()
    if source_mask is not None:
        mask = source_mask.to(device=prediction.device, dtype=torch.bool)
        if mask.shape != (batch_size, num_sources):
            raise ValueError("source_mask must have shape [B,K]")
        loss = torch.where(mask.unsqueeze(1), loss, torch.zeros_like(loss))
    return loss


@torch.no_grad()
def compute_delta_magnitude_loss(
    prediction,
    target,
    source_centroid,
    source_mask=None,
    center_genes=True,
    eps=1e-8,
):
    """Compare predicted and observed perturbation-delta RMS magnitudes."""
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError("prediction must be [B,K,S,G] and target [B,S,G]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes are incompatible")

    batch_size, num_sources = prediction.shape[:2]
    centroid = source_centroid.to(device=prediction.device, dtype=torch.float32)
    if centroid.ndim == 2:
        centroid = centroid.unsqueeze(0).expand(batch_size, -1, -1)
    if centroid.ndim != 3 or centroid.shape[:2] != (batch_size, num_sources):
        raise ValueError("source_centroid must have shape [K,G] or [B,K,G]")
    if centroid.shape[-1] != prediction.shape[-1]:
        raise ValueError("source_centroid and prediction gene dimensions disagree")

    centroid = centroid.unsqueeze(2)
    observed_delta = target.detach().float().unsqueeze(1) - centroid
    predicted_delta = prediction.detach().float() - centroid
    if center_genes:
        observed_delta = observed_delta - observed_delta.mean(dim=-1, keepdim=True)
        predicted_delta = predicted_delta - predicted_delta.mean(dim=-1, keepdim=True)
    observed_rms = observed_delta.square().mean(dim=-1).add(float(eps)).sqrt()
    predicted_rms = predicted_delta.square().mean(dim=-1).add(float(eps)).sqrt()
    loss = F.smooth_l1_loss(
        torch.log1p(predicted_rms),
        torch.log1p(observed_rms),
        reduction="none",
    ).permute(0, 2, 1).contiguous()
    if source_mask is not None:
        mask = source_mask.to(device=prediction.device, dtype=torch.bool)
        if mask.shape != (batch_size, num_sources):
            raise ValueError("source_mask must have shape [B,K]")
        loss = torch.where(mask.unsqueeze(1), loss, torch.zeros_like(loss))
    return loss


def compute_cell_cluster_mixture(
    prediction,
    target,
    source_centroid,
    source_prior,
    temperature,
    delta_score_weight=0.0,
    delta_magnitude_score_weight=0.0,
    source_mask=None,
    center_genes=True,
    source_sets=None,
    reliable_gene_weighting=False,
    reliability_max_weight=20.0,
    pca_projection_basis=None,
    pca_score_weight=0.0,
    normalize_matching_energy=False,
    matching_energy_clip=5.0,
):
    """Build detached cell-to-cluster responsibilities and the M-step MSE.

    Delta distance is used only to infer q. The optimized objective contains
    the original diffusion reconstruction error and no additive delta term.
    """
    diffusion_loss = compute_candidate_loss(
        prediction,
        target,
        reduce_gene_dim=True,
        reduce_cell_dim=False,
    )
    delta_loss = compute_delta_signature_loss(
        prediction,
        target,
        source_centroid,
        source_mask=source_mask,
        center_genes=center_genes,
        source_sets=source_sets,
        reliable_gene_weighting=reliable_gene_weighting,
        reliability_max_weight=reliability_max_weight,
    )
    delta_magnitude_loss = compute_delta_magnitude_loss(
        prediction,
        target,
        source_centroid,
        source_mask=source_mask,
        center_genes=center_genes,
    )
    pca_delta_loss = delta_loss.new_zeros(delta_loss.shape)
    if pca_projection_basis is not None and float(pca_score_weight) != 0.0:
        pca_delta_loss = compute_delta_signature_loss(
            prediction,
            target,
            source_centroid,
            source_mask=source_mask,
            center_genes=center_genes,
            source_sets=source_sets,
            projection_basis=pca_projection_basis,
        )
    diffusion_energy = diffusion_loss.detach()
    delta_energy = delta_loss
    delta_magnitude_energy = delta_magnitude_loss
    pca_delta_energy = pca_delta_loss
    if bool(normalize_matching_energy):
        normalization_kwargs = {
            "source_mask": source_mask,
            "clip": matching_energy_clip,
        }
        diffusion_energy = robust_normalize_candidate_energy(
            diffusion_energy, **normalization_kwargs
        )
        delta_energy = robust_normalize_candidate_energy(
            delta_energy, **normalization_kwargs
        )
        delta_magnitude_energy = robust_normalize_candidate_energy(
            delta_magnitude_energy, **normalization_kwargs
        )
        if pca_projection_basis is not None and float(pca_score_weight) != 0.0:
            pca_delta_energy = robust_normalize_candidate_energy(
                pca_delta_energy, **normalization_kwargs
            )
    score_loss = (
        diffusion_energy
        + float(delta_score_weight) * delta_energy
        + float(delta_magnitude_score_weight) * delta_magnitude_energy
        + float(pca_score_weight) * pca_delta_energy
    )
    responsibilities = compute_soft_em_responsibility(
        score_loss,
        source_prior,
        temperature=temperature,
        detach_candidate_loss=True,
        source_mask=source_mask,
    )
    mixture_mse_per_cell = (responsibilities * diffusion_loss).sum(dim=-1)
    return {
        "responsibilities": responsibilities,
        "diffusion_loss": diffusion_loss,
        "delta_loss": delta_loss,
        "delta_magnitude_loss": delta_magnitude_loss,
        "pca_delta_loss": pca_delta_loss,
        "diffusion_energy": diffusion_energy,
        "delta_energy": delta_energy,
        "delta_magnitude_energy": delta_magnitude_energy,
        "pca_delta_energy": pca_delta_energy,
        "matching_energy": score_loss.detach(),
        "mixture_mse_per_cell": mixture_mse_per_cell,
        "mixture_mse_per_sample": mixture_mse_per_cell.mean(dim=-1),
    }


def expand_shared_source_randomness(timestep, noise, num_sources):
    """Expand one timestep and noise draw per set across all source candidates."""
    if timestep.ndim != 1 or noise.shape[0] != timestep.shape[0]:
        raise ValueError("timestep and noise must share a batch dimension")
    num_sources = int(num_sources)
    expanded_timestep = timestep.unsqueeze(1).expand(-1, num_sources).reshape(-1)
    expanded_noise = noise.unsqueeze(1).expand(
        -1,
        num_sources,
        *noise.shape[1:],
    ).reshape(timestep.shape[0] * num_sources, *noise.shape[1:])
    return expanded_timestep, expanded_noise




def uniform_responsibilities_like(responsibilities, source_mask=None):
    """Return a normalized uniform distribution over valid sources."""
    if source_mask is None:
        return torch.full_like(responsibilities, 1.0 / responsibilities.shape[-1])
    mask = source_mask.to(device=responsibilities.device, dtype=torch.bool)
    if not mask.any(dim=-1).all():
        raise ValueError("every sample must have at least one valid source")
    while mask.ndim < responsibilities.ndim:
        mask = mask.unsqueeze(-2)
    uniform = mask.to(responsibilities.dtype)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True)
    return uniform.expand_as(responsibilities)

def compute_soft_em_responsibility(
    candidate_loss,
    source_prior,
    temperature,
    prior_scale=1.0,
    topk=None,
    eps=1e-8,
    detach_candidate_loss=True,
    source_mask=None,
):
    """Infer soft responsibilities from detached candidate losses and fixed priors."""
    if candidate_loss.ndim not in (2, 3):
        raise ValueError("candidate_loss must be [B,K] or [B,S,K]")
    if source_prior.shape[-1] != candidate_loss.shape[-1]:
        raise ValueError("source_prior and candidate_loss disagree on num_sources")
    temperature = max(float(temperature), 1e-6)
    prior = source_prior.to(
        device=candidate_loss.device,
        dtype=candidate_loss.dtype,
    )
    if source_mask is None:
        source_mask = torch.ones_like(prior, dtype=torch.bool)
    else:
        source_mask = source_mask.to(device=candidate_loss.device, dtype=torch.bool)
    if source_mask.shape != prior.shape:
        raise ValueError("source_mask and source_prior must have identical shapes")
    if not source_mask.any(dim=-1).all():
        raise ValueError("every sample must have at least one valid source")
    prior = torch.where(source_mask, prior.clamp_min(eps), torch.zeros_like(prior))
    prior = prior / prior.sum(dim=-1, keepdim=True)
    while prior.ndim < candidate_loss.ndim:
        prior = prior.unsqueeze(-2)
        source_mask = source_mask.unsqueeze(-2)
    score_loss = candidate_loss.detach() if detach_candidate_loss else candidate_loss
    scores = -score_loss / temperature + float(prior_scale) * prior.clamp_min(eps).log()
    scores = scores.masked_fill(~source_mask, -torch.inf)
    q = torch.softmax(scores, dim=-1)
    q = torch.where(source_mask, q, torch.zeros_like(q))

    if topk is not None:
        topk = int(topk)
        if topk < 1 or topk > q.shape[-1]:
            raise ValueError("topk must be in [1, num_sources]")
        top_values, top_indices = torch.topk(q, k=topk, dim=-1)
        sparse = torch.zeros_like(q)
        sparse.scatter_(dim=-1, index=top_indices, src=top_values)
        q = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
    return q / q.sum(dim=-1, keepdim=True).clamp_min(eps)


def resolve_support_schedule(global_step, schedule, transition_steps=0):
    """Resolve the active support stage and optional transition metadata."""
    if not schedule:
        return {
            "mode": "full_softmax",
            "topk": None,
            "previous_mode": None,
            "previous_topk": None,
            "transition_lambda": 1.0,
            "stage_index": 0,
        }

    step = int(global_step)
    stages = [dict(stage) for stage in schedule]
    stage_index = len(stages) - 1
    for index, stage in enumerate(stages):
        start = int(stage["start_step"])
        end = int(stage["end_step"])
        if start <= step < end:
            stage_index = index
            break
        if step < start:
            stage_index = max(0, index - 1)
            break

    stage = stages[stage_index]
    result = {
        "mode": str(stage["mode"]),
        "topk": stage.get("topk"),
        "previous_mode": None,
        "previous_topk": None,
        "transition_lambda": 1.0,
        "stage_index": stage_index,
    }
    transition_steps = int(transition_steps or 0)
    if stage_index > 0 and transition_steps > 0:
        elapsed = max(0, step - int(stage["start_step"]))
        if elapsed < transition_steps:
            previous = stages[stage_index - 1]
            result["previous_mode"] = str(previous["mode"])
            result["previous_topk"] = previous.get("topk")
            result["transition_lambda"] = min(
                1.0,
                float(elapsed) / float(transition_steps),
            )
    return result


def _topk_mask(scores, topk, ranking_scores=None):
    topk = int(topk)
    if topk < 1 or topk > scores.shape[-1]:
        raise ValueError("topk must be in [1, num_sources]")
    if ranking_scores is None:
        ranking_scores = scores
    indices = torch.topk(ranking_scores, k=topk, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    return mask


def _masked_softmax(scores, mask, eps):
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    q = torch.softmax(masked_scores, dim=-1)
    q = torch.where(mask, q, torch.zeros_like(q))
    return q / q.sum(dim=-1, keepdim=True).clamp_min(eps)


def _shuffled_ranking_scores(scores, permutation=None):
    flat = scores.reshape(-1, scores.shape[-1])
    if flat.shape[0] <= 1:
        return scores, torch.arange(flat.shape[0], device=scores.device)
    if permutation is None:
        permutation = torch.randperm(flat.shape[0], device=scores.device)
        if torch.equal(
            permutation,
            torch.arange(flat.shape[0], device=scores.device),
        ):
            permutation = permutation.roll(1)
    return flat.index_select(0, permutation).reshape_as(scores), permutation


def apply_support_policy(
    candidate_loss,
    source_prior,
    temperature,
    mode="full_softmax",
    topk=None,
    prior_scale=0.0,
    previous_mode=None,
    previous_topk=None,
    transition_lambda=1.0,
    eps=1e-8,
):
    """Return pre-support and scheduled post-support responsibilities."""
    pre_support = compute_soft_em_responsibility(
        candidate_loss,
        source_prior,
        temperature=temperature,
        prior_scale=prior_scale,
        topk=None,
        eps=eps,
        detach_candidate_loss=True,
    )
    score_loss = candidate_loss.detach()
    prior = source_prior.to(device=score_loss.device, dtype=score_loss.dtype)
    prior = prior.clamp_min(eps)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    scores = (
        -score_loss / max(float(temperature), 1e-6)
        + float(prior_scale) * prior.log()
    )

    random_ranking = None
    shuffled_ranking = None
    shuffle_permutation = None
    modes = {str(mode), str(previous_mode)}
    if "random_topk" in modes:
        random_ranking = torch.rand_like(scores)
    if "shuffled_loss_topk" in modes:
        shuffled_ranking, shuffle_permutation = _shuffled_ranking_scores(scores)

    def build(stage_mode, stage_topk):
        stage_mode = str(stage_mode)
        if stage_mode == "uniform_all_sources":
            q = torch.full_like(pre_support, 1.0 / pre_support.shape[-1])
            mask = torch.ones_like(pre_support, dtype=torch.bool)
        elif stage_mode == "full_softmax":
            q = pre_support
            mask = torch.ones_like(pre_support, dtype=torch.bool)
        elif stage_mode in ("soft_topk", "random_topk", "shuffled_loss_topk"):
            ranking = scores
            if stage_mode == "random_topk":
                ranking = random_ranking
            elif stage_mode == "shuffled_loss_topk":
                ranking = shuffled_ranking
            mask = _topk_mask(scores, stage_topk, ranking_scores=ranking)
            q = _masked_softmax(scores, mask, eps)
        else:
            raise ValueError(f"unsupported support mode={stage_mode}")
        return q, mask

    current_q, current_mask = build(mode, topk)
    lam = min(max(float(transition_lambda), 0.0), 1.0)
    if previous_mode is not None and lam < 1.0:
        previous_q, previous_mask = build(previous_mode, previous_topk)
        post_support = (1.0 - lam) * previous_q + lam * current_q
        post_support = post_support / post_support.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
        support_mask = post_support > 0
    else:
        post_support = current_q
        support_mask = current_mask

    return {
        "pre_support": pre_support,
        "post_support": post_support,
        "support_mask": support_mask,
        "transition_lambda": post_support.new_tensor(lam),
        "shuffle_permutation": shuffle_permutation,
    }


def compute_em_m_step_loss(
    candidate_loss,
    responsibility,
    source_balanced=False,
    eps=1e-8,
):
    """Compute the detached-responsibility generalized M-step objective."""
    if candidate_loss.shape != responsibility.shape:
        raise ValueError("candidate_loss and responsibility shapes must match")
    if responsibility.requires_grad:
        raise ValueError("responsibility must be detached")
    weighted = responsibility * candidate_loss
    if not source_balanced:
        return weighted.sum(dim=-1).mean()
    reduce_dims = tuple(range(candidate_loss.ndim - 1))
    weighted_per_source = weighted.sum(dim=reduce_dims)
    mass_per_source = responsibility.sum(dim=reduce_dims).clamp_min(eps)
    return (weighted_per_source / mass_per_source).mean()


def source_responsibilities(loss_per_source, source_prior, temperature=0.1):
    """Backward-compatible EM-v0 responsibility wrapper."""
    return compute_soft_em_responsibility(
        loss_per_source,
        source_prior,
        temperature=temperature,
        prior_scale=1.0,
        detach_candidate_loss=False,
    )


__all__ = [
    "apply_support_policy",
    "compute_candidate_loss",
    "compute_delta_magnitude_loss",
    "compute_delta_signature_loss",
    "compute_soft_em_responsibility",
    "compute_em_m_step_loss",
    "expand_shared_source_randomness",
    "resolve_support_schedule",
    "source_responsibilities",
]

"""Module `models/covariate_encoding.py`."""
import ast
import math

import torch
import torch.nn as nn
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)


TAHOE_FACTORIZED_DRUG_ENCODINGS = frozenset(
    {
        "tahoe_factorized_categorical",
        "tahoe_factorized_logdose",
        "tahoe_factorized_logdose_interaction",
    }
)


def _parse_tahoe_drug_dose(category):
    """Parse a Tahoe ``drugname_drugconc`` category into drug, dose, and unit."""
    try:
        value = ast.literal_eval(str(category))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid Tahoe drug-dose category: {category!r}") from exc

    if (
        not isinstance(value, (list, tuple))
        or len(value) != 1
        or not isinstance(value[0], (list, tuple))
        or len(value[0]) < 3
    ):
        raise ValueError(
            "Tahoe drug-dose categories must look like "
            "\"[('drug', dose, 'uM')]\"; got "
            f"{category!r}"
        )

    drug, raw_dose, raw_unit = value[0][:3]
    drug = str(drug).strip()
    unit = str(raw_unit).strip()
    try:
        dose_um = float(raw_dose)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Tahoe dose in category: {category!r}") from exc

    if not drug or not math.isfinite(dose_um) or dose_um < 0.0:
        raise ValueError(f"Invalid Tahoe drug or dose in category: {category!r}")
    if unit.lower() not in {"um", "μm", "µm"}:
        raise ValueError(
            f"Tahoe factorized encoders expect concentration in uM; got {unit!r}"
        )
    return drug, dose_um, unit


# ============================================================
# CovEncoder: Covariate Encoder
# Purpose:
# 1) Build perturbation / cell-type / batch encoding branches from config
# 2) Concatenate branch representations and project to a unified output dim
# ============================================================
class CovEncoder(nn.Module):
    """Covencoder implementation used by the PerturbDiff pipeline."""
    # ---------------------------
    # Initialization entry: assemble all encoding branches
    # ---------------------------
    def __init__(self, cov_cfg):
        """Special method `__init__`."""
        super().__init__()
        self.cov_cfg = cov_cfg
        hidden_dim = 0
        # Main perturbation branch (with optional Tahoe / Replogle sub-branches)
        hidden_dim += self._init_generic_perturbation_encoder(cov_cfg)
        # Cell-type branch
        hidden_dim += self._init_celltype_encoder(cov_cfg)
        # Batch branch
        hidden_dim += self._init_batch_encoder(cov_cfg)
        # Gather layer: linear projection after concatenation
        self._init_gather_layers(hidden_dim)

    # ---------------------------
    # Main perturbation branch
    # ---------------------------
    def _init_generic_perturbation_encoder(self, cov_cfg):
        """
        Generic perturbation encoder:
        - onehot/non: default categorical perturbation pathway.
        - esm2: optional PBMC-style pretrained perturbation embeddings.
        """
        hidden_dim = 0
        if cov_cfg.pert_encoding == "onehot" or cov_cfg.pert_encoding == "non":
            # Base onehot perturbation embedding (+1 reserves index 0 for no-perturbation)
            if cov_cfg.drug_encoding == "onehot" and cov_cfg.replogle_gene_encoding == "onehot":
                self.pert_encoder = nn.Embedding(
                    num_embeddings=cov_cfg.num_pert + 1,  # RNA-seq has no perturbation
                    embedding_dim=cov_cfg.hidden_dim,
                )
                hidden_dim += cov_cfg.hidden_dim
            # Optional Tahoe drug branch / Replogle gene branch
            hidden_dim += self._init_tahoe_drug_encoder(cov_cfg)
            hidden_dim += self._init_replogle_gene_encoder(cov_cfg)
        elif cov_cfg.pert_encoding == "esm2":
            # esm2 path only supports the PBMC-like configuration
            assert cov_cfg.drug_encoding == "onehot" and cov_cfg.replogle_gene_encoding == "onehot", (
                "pert_encoding=esm2 is only supported for the PBMC-like path "
                "(drug_encoding=onehot and replogle_gene_encoding=onehot)."
            )
            assert cov_cfg.pert_embedding_path is not None
            # Load pretrained embeddings and wrap as frozen embedding
            with open(cov_cfg.pert_embedding_path, "rb") as f:
                emb = pickle.load(f)
                emb = torch.tensor(list(emb.values())).float()
            emb = torch.cat([torch.zeros(1, emb.size(1)), emb], dim=0)
            emb = nn.Embedding.from_pretrained(emb, freeze=True)
            # index 79 corresponds to the control perturbation in the PBMC dataset, 
            # which is not included in the pretrained embeddings. We initialize 
            # its embedding as the mean of all other perturbation embeddings.
            emb.weight[79] += emb.weight.mean()
            self.pert_encoder = nn.Sequential(emb, nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim))
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Tahoe drug branch
    # ---------------------------
    def _init_tahoe_drug_encoder(self, cov_cfg):
        """
        Tahoe drug branch:
        - drug_encoding=chemberta_cls builds (drug_encoder + dose_encoder).
        - tahoe_factorized_* modes share a learned drug identity across doses and
          represent dose categorically or on a continuous log10 scale.
        - leaves generic pert encoder unchanged for other branches.
        """
        hidden_dim = 0
        if cov_cfg.drug_encoding == "onehot":
            pass
        elif cov_cfg.drug_encoding == "chemberta_cls":
            # Load drug embeddings (i.e., ChemBERTa CLS representations)
            with open(cov_cfg.drug_embedding_path, "rb") as fin:
                drug_embeddings = pickle.load(fin)

            # Collect and discretize doses, then build dose -> index mapping
            dose_idx = []
            for drugname_drugconc, idx in cov_cfg.pert_dict.items():
                drug, dose, _ = eval(drugname_drugconc)[0]
                dose_idx.append(dose)
            dose_idx = sorted(list(set(dose_idx)))
            dose_idx = {v: i for i, v in enumerate(dose_idx)}

            # Parse drug embedding and dose index for each perturbation entry
            all_drug_embed, all_drug_dose = {}, {}
            drug_embed_dim = None
            for drugname_drugconc, idx in cov_cfg.pert_dict.items():
                drug, dose, _ = eval(drugname_drugconc)[0]
                all_drug_dose[idx] = dose_idx[dose]
                drug = drug.strip()
                if drug == "DMSO_TF":
                    assert dose == 0.0
                    control_idx = idx
                    continue
                all_drug_embed[idx] = drug_embeddings[drug]
                drug_embed_dim = len(drug_embeddings[drug])

            # Dose encoder
            self.dose_encoder = nn.Embedding(
                num_embeddings=len(dose_idx),
                embedding_dim=cov_cfg.hidden_dim,
            )
            idx = np.zeros(len(cov_cfg.pert_dict))
            for k, v in all_drug_dose.items():
                idx[k] = v
            # Register as buffer so it moves with the model but is not trainable
            self.register_buffer("dose_indices", torch.tensor(idx, dtype=torch.int))
            hidden_dim += cov_cfg.hidden_dim

            # Drug embedding table (control entry is filled with mean embedding)
            assert sorted(all_drug_embed.keys()) == [x for x in range(len(cov_cfg.pert_dict)) if x != control_idx]
            emb = np.zeros((len(cov_cfg.pert_dict), drug_embed_dim))
            for k, v in all_drug_embed.items():
                emb[k] = v
            emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=True)
            emb.weight[control_idx] = emb.weight.mean()
            self.drug_encoder = nn.Sequential(
                emb,
                nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim),
            )
            hidden_dim += cov_cfg.hidden_dim
        elif cov_cfg.drug_encoding in TAHOE_FACTORIZED_DRUG_ENCODINGS:
            hidden_dim += self._init_tahoe_factorized_encoder(cov_cfg)
        else:
            raise NotImplementedError
        return hidden_dim

    def _init_tahoe_factorized_encoder(self, cov_cfg):
        """Build learned drug identity and explicit dose branches for Tahoe."""
        items = sorted(cov_cfg.pert_dict.items(), key=lambda item: int(item[1]))
        indices = [int(idx) for _, idx in items]
        expected_indices = list(range(len(items)))
        if indices != expected_indices:
            raise ValueError(
                "Tahoe perturbation indices must be contiguous and start at zero; "
                f"got {indices[:10]}"
            )
        if int(cov_cfg.num_pert) != len(items):
            raise ValueError(
                f"num_pert={cov_cfg.num_pert} does not match pert_dict size={len(items)}"
            )

        parsed = [_parse_tahoe_drug_dose(category) for category, _ in items]
        control_mask = [drug == "DMSO_TF" for drug, _, _ in parsed]
        if sum(control_mask) != 1:
            raise ValueError(
                "Tahoe factorized encoders require exactly one DMSO_TF control category"
            )
        for (drug, dose_um, _), is_control in zip(parsed, control_mask):
            if is_control and dose_um != 0.0:
                raise ValueError("DMSO_TF must have concentration 0.0 uM")
            if not is_control and dose_um <= 0.0:
                raise ValueError(
                    f"Treated Tahoe category {drug!r} must have positive concentration"
                )

        active_drugs = sorted({drug for drug, _, _ in parsed if drug != "DMSO_TF"})
        self.tahoe_drug_to_idx = {
            drug: idx + 1 for idx, drug in enumerate(active_drugs)
        }
        drug_indices = [
            0 if is_control else self.tahoe_drug_to_idx[drug]
            for (drug, _, _), is_control in zip(parsed, control_mask)
        ]
        self.register_buffer(
            "tahoe_drug_indices", torch.tensor(drug_indices, dtype=torch.long)
        )
        self.register_buffer(
            "tahoe_treatment_mask",
            torch.tensor([not value for value in control_mask], dtype=torch.float32),
        )
        self.drug_encoder = nn.Embedding(
            num_embeddings=len(active_drugs) + 1,
            embedding_dim=cov_cfg.hidden_dim,
            padding_idx=0,
        )

        mode = cov_cfg.drug_encoding
        if mode == "tahoe_factorized_categorical":
            active_doses = sorted(
                {
                    dose_um
                    for (_, dose_um, _), is_control in zip(parsed, control_mask)
                    if not is_control
                }
            )
            self.tahoe_dose_to_idx = {
                dose: idx + 1 for idx, dose in enumerate(active_doses)
            }
            dose_indices = [
                0 if is_control else self.tahoe_dose_to_idx[dose_um]
                for (_, dose_um, _), is_control in zip(parsed, control_mask)
            ]
            self.register_buffer(
                "tahoe_dose_indices", torch.tensor(dose_indices, dtype=torch.long)
            )
            self.dose_encoder = nn.Embedding(
                num_embeddings=len(active_doses) + 1,
                embedding_dim=cov_cfg.hidden_dim,
                padding_idx=0,
            )
            fusion_inputs = 2
        else:
            features = []
            for (_, dose_um, _), is_control in zip(parsed, control_mask):
                if is_control:
                    features.append((0.0, 0.0))
                else:
                    log_dose = math.log10(dose_um) - math.log10(0.5)
                    features.append((log_dose, 1.0))
            self.register_buffer(
                "tahoe_logdose_features", torch.tensor(features, dtype=torch.float32)
            )
            self.dose_encoder = nn.Sequential(
                nn.Linear(2, 32),
                nn.SiLU(),
                nn.Linear(32, cov_cfg.hidden_dim),
            )
            fusion_inputs = (
                3 if mode == "tahoe_factorized_logdose_interaction" else 2
            )

        # A bias-free fusion preserves the exact zero control condition while
        # keeping every ablation at the same downstream dimensionality.
        self.tahoe_condition_fusion = nn.Linear(
            fusion_inputs * cov_cfg.hidden_dim,
            cov_cfg.hidden_dim,
            bias=False,
        )
        return cov_cfg.hidden_dim

    def _encode_tahoe_factorized_condition(self, pert_input):
        """Return the fixed-width Tahoe drug-dose condition representation."""
        pert_indices = pert_input.long()
        treatment_mask = self.tahoe_treatment_mask[pert_indices].unsqueeze(-1)
        drug_repr = self.drug_encoder(self.tahoe_drug_indices[pert_indices])

        if self.cov_cfg.drug_encoding == "tahoe_factorized_categorical":
            dose_repr = self.dose_encoder(self.tahoe_dose_indices[pert_indices])
        else:
            dose_repr = self.dose_encoder(self.tahoe_logdose_features[pert_indices])
            dose_repr = dose_repr * treatment_mask

        branches = [drug_repr, dose_repr]
        if self.cov_cfg.drug_encoding == "tahoe_factorized_logdose_interaction":
            branches.append(drug_repr * dose_repr)
        condition = self.tahoe_condition_fusion(torch.cat(branches, dim=-1))
        return condition * treatment_mask

    # ---------------------------
    # Replogle gene branch
    # ---------------------------
    def _init_replogle_gene_encoder(self, cov_cfg):
        """
        Replogle gene branch:
        - replogle_gene_encoding=genept replaces self.pert_encoder
          with a pretrained gene embedding encoder.
        """
        hidden_dim = 0
        if cov_cfg.replogle_gene_encoding == "onehot":
            pass
        elif cov_cfg.replogle_gene_encoding == "genept":
            # Load pretrained gene embeddings
            with open(cov_cfg.replogle_gene_embedding_path, "rb") as fin:
                rep_gene_embeddings = pickle.load(fin)
            if len(rep_gene_embeddings) == 0:
                raise ValueError(
                    f"Empty replogle gene embedding dict: {cov_cfg.replogle_gene_embedding_path}"
                )

            # Build gene embedding table; missing perturbations fall back to zero vectors.
            all_gene_embed = {}
            missing_gene_perts = []
            gene_embed_dim = len(next(iter(rep_gene_embeddings.values())))
            control_idx = None
            for gene_pert, idx in cov_cfg.pert_dict.items():
                gene_pert = str(gene_pert)
                if gene_pert == "non-targeting":
                    control_idx = idx
                    continue
                emb_vec = rep_gene_embeddings.get(gene_pert)
                if emb_vec is None:
                    all_gene_embed[idx] = np.zeros(gene_embed_dim, dtype=np.float32)
                    missing_gene_perts.append(gene_pert)
                else:
                    all_gene_embed[idx] = emb_vec

            if missing_gene_perts:
                logger.warning(
                    "Missing %d replogle perturbation embeddings (examples: %s). Using zero vectors as fallback.",
                    len(missing_gene_perts),
                    ", ".join(sorted(set(missing_gene_perts))[:10]),
                )

            emb = np.zeros((len(cov_cfg.pert_dict), gene_embed_dim))
            for k, v in all_gene_embed.items():
                emb[k] = v
            emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=True)
            if control_idx is not None:
                emb.weight[control_idx] = emb.weight.mean()
            self.pert_encoder = nn.Sequential(
                emb,
                nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim),
            )
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Cell-type branch
    # ---------------------------
    def _init_celltype_encoder(self, cov_cfg):
        """
        Init celltype encoder.

        :param cov_cfg: Covariate encoder configuration.
        :return: Computed output(s) for this function.
        """
        hidden_dim = 0
        if cov_cfg.celltype_encoding == "onehot":
            self.celltype_encoder = nn.Embedding(
                num_embeddings=cov_cfg.num_celltype,
                embedding_dim=cov_cfg.hidden_dim,
            )
            hidden_dim += cov_cfg.hidden_dim
        elif cov_cfg.celltype_encoding == "llm":
            # Use external LLM/pretrained cell-type embeddings
            assert cov_cfg.celltype_embedding_path is not None
            with open(cov_cfg.celltype_embedding_path, "rb") as f:
                celltype_emb_dict = pickle.load(f)
            self.celltype_idx_dict = cov_cfg.cell_type_dict
            # Align indices with cell_type_dict to avoid index mismatch
            emb = {cov_cfg.cell_type_dict[k]: celltype_emb_dict[k] for k in cov_cfg.cell_type_dict}
            emb = {k: emb[k] for k in sorted(emb.keys())}
            emb = torch.tensor(list(emb.values())).float()
            emb = nn.Embedding.from_pretrained(emb, freeze=True)
            self.celltype_encoder = nn.Sequential(
                emb,
                nn.Linear(emb.weight.size(1), cov_cfg.hidden_dim),
            )
            hidden_dim += cov_cfg.hidden_dim
        else:
            raise NotImplementedError
        return hidden_dim

    # ---------------------------
    # Batch branch
    # ---------------------------
    def _init_batch_encoder(self, cov_cfg):
        """Execute `_init_batch_encoder` and return values used by downstream logic."""
        hidden_dim = 0
        if cov_cfg.batch_encoding is None:
            return hidden_dim
        if cov_cfg.batch_encoding == "onehot":
            self.batch_encoder = nn.Embedding(
                num_embeddings=cov_cfg.num_batch,
                embedding_dim=cov_cfg.hidden_dim,
            )
            hidden_dim += cov_cfg.hidden_dim
        return hidden_dim

    # ---------------------------
    # Gather layer
    # ---------------------------
    def _init_gather_layers(self, hidden_dim):
        # Map concatenated multi-branch representation to output dimension
        """Execute `_init_gather_layers` and return values used by downstream logic."""
        self.transform = nn.Linear(hidden_dim, self.cov_cfg.output_dim)

    # ---------------------------
    # Forward pass
    # ---------------------------
    def forward(self, pert_input, celltype_input, batch_input):
        """
        Run the module forward pass.

        :param pert_input: Input `pert_input` value.
        :param celltype_input: Input `celltype_input` value.
        :param batch_input: Input `batch_input` value.
        :return: Model output tensor(s) for the given inputs.
        """
        reprs = []

        # 1) Perturbation-related representation
        if self.cov_cfg.drug_encoding == "chemberta_cls":
            # Tahoe mode: ChemBERTa drug embedding + dose embedding
            reprs.append(self.drug_encoder(pert_input))
            reprs.append(self.dose_encoder(self.dose_indices[pert_input]))
        elif self.cov_cfg.drug_encoding in TAHOE_FACTORIZED_DRUG_ENCODINGS:
            reprs.append(self._encode_tahoe_factorized_condition(pert_input))
        elif self.cov_cfg.replogle_gene_encoding == "genept":
            # Replogle mode: gene embedding
            reprs.append(self.pert_encoder(pert_input))
        elif self.cov_cfg.pert_encoding == "non":
            # "non" mode: force zero index (useful for pretraining on marginal cells)
            reprs.append(self.pert_encoder(torch.zeros_like(pert_input, dtype=pert_input.dtype)))
        else:
            # onehot mode: input index +1 (index 0 is reserved for the control type)
            # or PBMC mode: cytokine embeddings (ESM2)
            reprs.append(self.pert_encoder(pert_input + 1))

        # 2) Cell-type representation
        reprs.append(self.celltype_encoder(celltype_input))

        # 3) Batch representation (optional)
        if hasattr(self, "batch_encoder"):            
            reprs.append(self.batch_encoder(batch_input))
        
        # 4) Concatenate and linearly project to output space
        reprs = torch.cat(reprs, dim=-1)
        return self.transform(reprs)

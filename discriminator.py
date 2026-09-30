"""Use the reference SigLIP2 pair reward network as a trainable pair ranker."""

import torch

from siglip2_pair_sr_reward_optional_fidelity import SigLIP2PairSRReward


ARCH_KEYS = (
    "image_size", "use_pooler", "hidden_layers", "token_pool_size", "attn_dim",
    "num_heads", "num_fusion_layers", "sr_quality_layers", "use_fidelity_branch",
    "use_sr_quality_branch", "head_hidden_dim", "dropout", "normalize_global_feature",
)


def build_discriminator(config: dict, architecture: dict | None = None, reward_path: str | None = None):
    reward = torch.load(reward_path, map_location="cpu", weights_only=False) if reward_path else None
    if architecture is None:
        if reward is not None:
            architecture = {key: reward[key] for key in ARCH_KEYS if key in reward}
        else:
            architecture = dict(config["discriminator"])
    architecture = dict(architecture)
    architecture["hidden_layers"] = tuple(architecture["hidden_layers"])
    if not (architecture.get("use_pooler") or architecture.get("use_fidelity_branch")):
        raise ValueError("Pair discriminator must depend on the LR image")
    model = SigLIP2PairSRReward(
        model_name=config["siglip2"], dtype=torch.float32,
        freeze_backbone=True, backbone_no_grad=False, **architecture,
    )
    if reward is not None:
        model.cross_blocks.load_state_dict(reward["cross_blocks"], strict=True)
        if model.sr_quality_branch is not None:
            model.sr_quality_branch.load_state_dict(reward["sr_quality_branch"], strict=True)
        model.reward_head.load_state_dict(reward["reward_head"], strict=True)
    model.vision_model.eval().requires_grad_(False)
    return model, architecture


def trainable_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith("vision_model.")}


def load_trainable_state(model, state):
    expected = {key for key in model.state_dict() if not key.startswith("vision_model.")}
    if set(state) != expected:
        raise ValueError(f"Discriminator keys differ: missing={expected - set(state)}, extra={set(state) - expected}")
    model.load_state_dict(state, strict=False)

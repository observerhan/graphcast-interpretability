"""Load legacy TopK, L1 KAN, and signed KAN + TopK checkpoints."""
import torch


def load_model(saved, device="cpu"):
    kind = saved.get("model_type", "topk")  # Original notebook 02 checkpoints.
    cfg = saved["config"]
    if kind == "kan_topk":
        from graphcast_interpretability.kan_topk_sae import KANTopKSAE
        training = saved.get("training_config", {})
        if "warmup_epochs" in training and saved.get("epoch", 0) <= training["warmup_epochs"]:
            raise ValueError("Complete sparse training before using a warm-up KAN + TopK checkpoint.")
        model = KANTopKSAE(**cfg)
    elif kind == "kan_sae":
        from graphcast_interpretability.kan_sae import KANSAE
        model = KANSAE(**cfg)
    elif kind == "topk":
        from graphcast_interpretability.model import SAE
        model = SAE(d_in=cfg["d_in"], latent=cfg["latent"], k_active=cfg["k"],
                    k_aux=cfg["k_aux"], dead_window=cfg["dead_window"],
                    unit_norm_decoder=cfg.get("unit_norm_decoder", True))
    else:
        raise ValueError(f"Unknown SAE checkpoint model_type: {kind!r}")
    model.load_state_dict(saved["model_state"], strict=True)
    if kind in {"kan_sae", "kan_topk"} and not bool(model.grid_initialized):
        raise ValueError("The KAN checkpoint has no calibrated spline grids.")
    model.sae_type = kind
    return model.to(device)


def model_type(model):
    from graphcast_interpretability.kan_topk_sae import KANTopKSAE
    from graphcast_interpretability.kan_sae import KANSAE
    if isinstance(model, KANTopKSAE):
        return "kan_topk"
    return "kan_sae" if isinstance(model, KANSAE) else "topk"


def feature_codes(model, x):
    """KAN encode avoids an unused reconstruction; TopK keeps its original gate."""
    return model.encode(x) if model_type(model) in {"kan_sae", "kan_topk"} else model(x)[1]

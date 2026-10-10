"""Stable manuscript codes, independent of performance rank or exclusions."""

MODEL_LABELS: dict[str, tuple[str, str]] = {
    "direct_mlp": ("D-MLP", "Direct dense MLP"),
    "independent_cnn1d": ("D-CNN", "Independent temporal CNN"),
    "onishi_cnn": ("D-ONI", "Onishi-inspired paired CNN"),
    "direct_linear_ridge": ("D-LIN", "Direct linear ridge"),
    "direct_minirocket": ("D-MR", "Direct MiniRocket + ridge"),
    "antisymmetric_mlp": ("S-MLP", "Shared dense MLP"),
    "locally_connected_mlp": ("S-LCM", "Shared locally connected MLP"),
    "shared_cnn1d": ("S-CNN", "Shared temporal CNN"),
    "shared_linear_ridge": ("S-LIN", "Shared linear ridge"),
    "shared_minirocket": ("S-MR", "Shared MiniRocket + ridge"),
}


def model_label(name: str) -> tuple[str, str]:
    """Return a stable code and display name, retaining unknown external names."""
    return MODEL_LABELS.get(name, (name, name))

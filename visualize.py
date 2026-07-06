"""
Visualise intermediate EEGNet features with t-SNE and UMAP.

Works by attaching a forward hook to any named layer of the model and
collecting its output activations over a DataLoader — no changes to your
EEGNet code required.

Public API
----------
    extract_features(model, loader, layer_name, device)
        -> (features, labels)   both numpy arrays

    plot_tsne(features, labels, ...)
    plot_umap(features, labels, ...)
    plot_both(features, labels, ...)   <- recommended: side-by-side comparison

    visualize_layer(model, loader, layer_name, ...)
        <- one-stop convenience wrapper

Usage example
-------------
    from visualize import visualize_layer
    from dataset import build_loaders

    # After training:
    train_dl, val_dl, test_dl = build_loaders(...)
    visualize_layer(
        model, test_dl,
        layer_name="depthwise_conv",   # any named module in your EEGNet
        title="EEGNet — depthwise conv features (test set)",
        save_path="figures/depthwise_tsne_umap.png",
    )

Finding layer names
-------------------
    for name, module in model.named_modules():
        print(name, '->', type(module).__name__)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[visualize] umap-learn not installed. "
          "Install with:  pip install umap-learn\n"
          "UMAP plots will be skipped.")

from config import CLASS_NAMES


# -----------------------------------------------------------------------
# Colour palette
# -----------------------------------------------------------------------
_CLASS_COLORS  = ["#E63946", "#457B9D"]        # red / steel-blue  (binary)
_SUBJECT_CMAP  = "tab20"


# -----------------------------------------------------------------------
# 1. Hook-based feature extractor
# -----------------------------------------------------------------------

class _FeatureHook:
    """Attaches a forward hook to a module and buffers its output."""

    def __init__(self):
        self.features = []

    def hook_fn(self, module, input, output):
        # Flatten spatial dims; keep batch dim.
        # output shape varies by layer — we flatten everything after dim 0.
        if isinstance(output, torch.Tensor):
            self.features.append(output.detach().cpu().flatten(start_dim=1))

    def clear(self):
        self.features = []


def _get_module_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Navigate dotted layer names, e.g. 'block1.conv'."""
    parts = layer_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


@torch.no_grad()
def extract_features(
    model:      nn.Module,
    loader:     DataLoader,
    layer_name: str,
    device:     str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run `loader` through `model` and collect the output of `layer_name`.

    Parameters
    ----------
    model       : trained EEGNet (or any nn.Module)
    loader      : DataLoader — labels must be the second element of each batch
    layer_name  : dotted name of the target layer, e.g. "depthwise_conv"
                  Use print_layers(model) to list all valid names.
    device      : 'cuda' / 'mps' / 'cpu'  (auto-detected if None)

    Returns
    -------
    features : np.ndarray  (n_trials, n_features)   flattened activations
    labels   : np.ndarray  (n_trials,)              integer class labels
    """
    if device is None:
        device = _auto_device()

    model.eval().to(device)

    # Attach hook
    hook    = _FeatureHook()
    target  = _get_module_by_name(model, layer_name)
    handle  = target.register_forward_hook(hook.hook_fn)

    all_labels = []
    try:
        for X, y in loader:
            model(X.to(device))          # forward pass — hook fills itself
            all_labels.append(y.numpy())
    finally:
        handle.remove()                  # always clean up, even if an error occurs

    features = torch.cat(
        [torch.tensor(f) for f in hook.features], dim=0
    ).numpy()                            # (n_trials, n_features)
    labels   = np.concatenate(all_labels)

    print(f"[extract_features] layer='{layer_name}'  "
          f"raw shape: {features.shape}  labels: {labels.shape}")
    return features, labels


def print_layers(model: nn.Module):
    """Print all named modules so you can pick a layer_name."""
    print("\nAvailable layers in model:")
    for name, module in model.named_modules():
        if name:   # skip the root ''
            print(f"  {name:<40} {type(module).__name__}")


# -----------------------------------------------------------------------
# 2. Dimensionality reduction helpers
# -----------------------------------------------------------------------

def _scale(features: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance across the feature dimension."""
    return StandardScaler().fit_transform(features)


def _run_tsne(
    features:    np.ndarray,
    perplexity:  float = 30.0,
    n_iter:      int   = 1000,
    random_state: int  = 42,
) -> np.ndarray:
    print(f"  Running t-SNE  (perplexity={perplexity}, n_iter={n_iter}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",          # PCA init is more stable than random
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def _run_umap(
    features:     np.ndarray,
    n_neighbors:  int   = 15,
    min_dist:     float = 0.1,
    random_state: int   = 42,
) -> np.ndarray:
    if not UMAP_AVAILABLE:
        return None
    print(f"  Running UMAP  (n_neighbors={n_neighbors}, min_dist={min_dist}) ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


# -----------------------------------------------------------------------
# 3. Plotting helpers
# -----------------------------------------------------------------------

def _scatter(
    ax,
    embedding:    np.ndarray,
    labels:       np.ndarray,
    subject_ids:  np.ndarray = None,
    title:        str        = "",
    color_by:     str        = "class",   # "class" | "subject"
):
    """
    Draw one scatter plot on `ax`.

    color_by="class"   — colour each point by its class label (default)
    color_by="subject" — colour each point by subject ID (useful for
                         cross-subject experiments to check subject clustering)
    """
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.tick_params(labelsize=8)

    if color_by == "subject" and subject_ids is not None:
        unique_subs = np.unique(subject_ids)
        cmap        = plt.get_cmap(_SUBJECT_CMAP, len(unique_subs))
        sub_to_idx  = {s: i for i, s in enumerate(unique_subs)}
        colors      = [cmap(sub_to_idx[s]) for s in subject_ids]
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=colors, s=12, alpha=0.6, linewidths=0)
        handles = [
            mpatches.Patch(color=cmap(i), label=str(s))
            for i, s in enumerate(unique_subs)
        ]
        ax.legend(handles=handles, title="Subject", fontsize=7,
                  title_fontsize=8, loc="best", markerscale=1.5,
                  ncol=max(1, len(unique_subs) // 10))
    else:
        # Colour by class
        unique_classes = np.unique(labels)
        for cls in unique_classes:
            mask  = labels == cls
            color = _CLASS_COLORS[cls % len(_CLASS_COLORS)]
            name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=name, s=12, alpha=0.7, linewidths=0)
        ax.legend(fontsize=9, loc="best", markerscale=2)


# -----------------------------------------------------------------------
# 4. Public plot functions
# -----------------------------------------------------------------------

def plot_tsne(
    features:    np.ndarray,
    labels:      np.ndarray,
    subject_ids: np.ndarray = None,
    perplexity:  float      = 30.0,
    n_iter:      int        = 1000,
    title:       str        = "t-SNE",
    color_by:    str        = "class",
    save_path:   str        = None,
    show:        bool       = False,
) -> plt.Figure:
    """
    t-SNE plot of `features` coloured by class (or subject).

    Parameters
    ----------
    features    : (n_trials, n_features)  raw activations from extract_features()
    labels      : (n_trials,)             integer class labels
    subject_ids : (n_trials,)             optional subject ID per trial
                                          (required for color_by='subject')
    perplexity  : t-SNE perplexity — rule of thumb: sqrt(n_trials)
    n_iter      : t-SNE optimisation iterations
    title       : figure title
    color_by    : 'class' or 'subject'
    save_path   : if given, save figure to this path (.png / .pdf)
    show        : call plt.show() (use False on headless servers)

    Returns
    -------
    fig : matplotlib Figure
    """
    scaled    = _scale(features)
    embedding = _run_tsne(scaled, perplexity=perplexity, n_iter=n_iter)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, embedding, labels, subject_ids, title=title, color_by=color_by)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved t-SNE plot → {save_path}")
    if show:
        plt.show()
    return fig


def plot_umap(
    features:    np.ndarray,
    labels:      np.ndarray,
    subject_ids: np.ndarray = None,
    n_neighbors: int        = 15,
    min_dist:    float      = 0.1,
    title:       str        = "UMAP",
    color_by:    str        = "class",
    save_path:   str        = None,
    show:        bool       = False,
) -> plt.Figure | None:
    """
    UMAP plot of `features`. Requires umap-learn:  pip install umap-learn

    Parameters
    ----------
    n_neighbors : UMAP neighbourhood size — larger = more global structure
    min_dist    : minimum distance between points in the embedding
                  smaller = tighter clusters, larger = more spread

    Returns None if umap-learn is not installed.
    """
    if not UMAP_AVAILABLE:
        print("[plot_umap] Skipped — umap-learn not installed.")
        return None

    scaled    = _scale(features)
    embedding = _run_umap(scaled, n_neighbors=n_neighbors, min_dist=min_dist)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, embedding, labels, subject_ids, title=title, color_by=color_by)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved UMAP plot  → {save_path}")
    if show:
        plt.show()
    return fig


def plot_both(
    features:    np.ndarray,
    labels:      np.ndarray,
    subject_ids: np.ndarray = None,
    perplexity:  float      = 30.0,
    n_iter:      int        = 1000,
    n_neighbors: int        = 15,
    min_dist:    float      = 0.1,
    title:       str        = "",
    color_by:    str        = "class",
    save_path:   str        = None,
    show:        bool       = False,
) -> plt.Figure:
    """
    Side-by-side t-SNE | UMAP comparison in one figure.
    Falls back to t-SNE only if umap-learn is not installed.

    Parameters are the union of plot_tsne() and plot_umap() params.
    """
    n_cols = 2 if UMAP_AVAILABLE else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    scaled = _scale(features)

    # t-SNE
    tsne_emb = _run_tsne(scaled, perplexity=perplexity, n_iter=n_iter)
    _scatter(axes[0], tsne_emb, labels, subject_ids,
             title="t-SNE", color_by=color_by)

    # UMAP
    if UMAP_AVAILABLE:
        umap_emb = _run_umap(scaled, n_neighbors=n_neighbors, min_dist=min_dist)
        _scatter(axes[1], umap_emb, labels, subject_ids,
                 title="UMAP", color_by=color_by)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved combined plot → {save_path}")
    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------
# 5. One-stop convenience wrapper
# -----------------------------------------------------------------------

def visualize_layer(
    model:       nn.Module,
    loader:      DataLoader,
    layer_name:  str,
    subject_ids: np.ndarray = None,
    perplexity:  float      = 30.0,
    n_neighbors: int        = 15,
    min_dist:    float      = 0.1,
    color_by:    str        = "class",
    title:       str        = None,
    save_path:   str        = None,
    device:      str        = None,
    show:        bool       = False,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Extract features from `layer_name` and produce a side-by-side
    t-SNE + UMAP plot in one call.

    Parameters
    ----------
    model       : trained EEGNet
    loader      : DataLoader (test or val set recommended)
    layer_name  : name of the layer whose output to visualise
                  e.g. "depthwise_conv", "block2.conv", "fc"
                  Use print_layers(model) to discover valid names.
    subject_ids : optional (n_trials,) array of subject IDs — if provided
                  you can pass color_by='subject' to see subject clustering
    perplexity  : t-SNE perplexity  (good default: 30; try 5–50)
    n_neighbors : UMAP neighbourhood size  (good default: 15; try 5–50)
    min_dist    : UMAP min_dist  (0.0 = tight clusters, 0.5 = spread out)
    color_by    : 'class' or 'subject'
    title       : figure suptitle (auto-generated if None)
    save_path   : path to save the figure, e.g. "figures/layer_vis.png"
    device      : compute device (auto-detected if None)
    show        : call plt.show() after plotting

    Returns
    -------
    fig      : matplotlib Figure
    features : raw extracted features  (n_trials, n_features)
    labels   : class labels            (n_trials,)
    """
    import os
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"\n── Feature visualisation: layer='{layer_name}' ──────────────")
    features, labels = extract_features(model, loader, layer_name, device)

    auto_title = title or f"Layer: {layer_name}  (n={len(labels)})"
    fig = plot_both(
        features, labels,
        subject_ids = subject_ids,
        perplexity  = perplexity,
        n_neighbors = n_neighbors,
        min_dist    = min_dist,
        color_by    = color_by,
        title       = auto_title,
        save_path   = save_path,
        show        = show,
    )
    return fig, features, labels


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------
# Quick smoke-test (no real model needed)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    # Fake features and binary labels
    rng      = np.random.default_rng(42)
    n        = 200
    features = np.vstack([
        rng.normal(loc=[ 1,  1], scale=1.2, size=(n // 2, 32)),
        rng.normal(loc=[-1, -1], scale=1.2, size=(n // 2, 32)),
    ])
    labels   = np.array([0] * (n // 2) + [1] * (n // 2))

    fig = plot_both(
        features, labels,
        title     = "Smoke test — random features",
        save_path = "figures/smoke_test.png",
        show      = False,
    )
    print("Smoke test passed. Figure saved to figures/smoke_test.png")
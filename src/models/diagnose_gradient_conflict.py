import itertools
import random

import numpy as np
import torch
import torch.nn as nn

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.nas.search_space import sample_architecture


# ============================================================
# 1. Load off-target data
# ============================================================

def get_offtarget_loaders(batch_size=32):
    data = np.load(
        "data/processed/offtarget_crispr_splits.npz"
    )

    def make(split, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(
                data[f"X_{split}"],
                dtype=torch.float32,
            ),
            torch.tensor(
                data[f"y_{split}"],
                dtype=torch.float32,
            ),
        )

        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    pos_weight = torch.tensor(
        [
            (data["y_train"] == 0).sum()
            / max((data["y_train"] == 1).sum(), 1)
        ],
        dtype=torch.float32,
    )

    return (
        make("train", True),
        make("val", False),
        make("test", False),
        pos_weight,
    )


# ============================================================
# 2. Get a COMPLETE gradient vector
# ============================================================

def get_gradient_vector(model, loss):
    """
    Calculate the gradient of one task loss with respect to
    ALL model parameters.

    Important:
    If a parameter does not receive a gradient from a task,
    we insert zeros for that parameter.

    This guarantees that OT and OFF gradient vectors have
    exactly the same length and can therefore be compared.
    """

    model.zero_grad()

    loss.backward(
        retain_graph=True
    )

    gradients = []

    for parameter in model.parameters():

        if parameter.grad is None:
            # This parameter was not involved in this task.
            # Use zeros with exactly the same shape.
            gradients.append(
                torch.zeros_like(
                    parameter
                ).flatten()
            )

        else:
            gradients.append(
                parameter.grad.detach().flatten()
            )

    return torch.cat(gradients)


# ============================================================
# 3. Main diagnostic
# ============================================================

def diagnose_gradient_conflict(
    model,
    batch_size=32,
):
    """
    Measure:

    1. OT gradient norm
    2. OFF gradient norm
    3. OFF / OT gradient ratio
    4. Cosine similarity between OT and OFF gradients

    Cosine similarity:

        Positive  -> tasks push in similar directions
        Near zero -> tasks are mostly unrelated
        Negative  -> tasks push in opposing directions
    """

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    ot_train = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "train",
        batch_size,
        shuffle=True,
    )

    off_train, _, _, pos_weight = (
        get_offtarget_loaders(batch_size)
    )

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    mse_loss = nn.MSELoss()

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Running-loss normalization
    #
    # Keep exactly the same normalization used in your
    # multitask training.
    # --------------------------------------------------------

    running_ot = 1.0
    running_off = 1.0

    momentum = 0.9

    # --------------------------------------------------------
    # Store measurements
    # --------------------------------------------------------

    ot_gradient_norms = []
    off_gradient_norms = []
    gradient_ratios = []
    cosine_similarities = []

    # --------------------------------------------------------
    # Pair OT and OFF batches exactly like multitask training
    # --------------------------------------------------------

    off_cycle = itertools.cycle(
        off_train
    )

    max_batches = 20

    model.train()

    for step, (
        (x_seq, x_aux, y_ot),
        (x_pair, y_off),
    ) in enumerate(
        zip(ot_train, off_cycle)
    ):

        if step >= max_batches:
            break

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        pred_ot = model.forward_ontarget(
            x_seq,
            x_aux,
        )

        pred_off = model.forward_offtarget(
            x_pair,
        )

        # ----------------------------------------------------
        # Raw task losses
        # ----------------------------------------------------

        loss_ot = mse_loss(
            pred_ot,
            y_ot,
        )

        loss_off = bce_loss(
            pred_off,
            y_off,
        )

        # ----------------------------------------------------
        # Update running loss normalization
        # ----------------------------------------------------

        running_ot = (
            momentum * running_ot
            + (1 - momentum) * loss_ot.item()
        )

        running_off = (
            momentum * running_off
            + (1 - momentum) * loss_off.item()
        )

        # ----------------------------------------------------
        # Normalized losses
        # ----------------------------------------------------

        norm_loss_ot = (
            loss_ot
            / (running_ot + 1e-8)
        )

        norm_loss_off = (
            loss_off
            / (running_off + 1e-8)
        )

        # ----------------------------------------------------
        # OT gradient
        # ----------------------------------------------------

        ot_gradient = get_gradient_vector(
            model,
            norm_loss_ot,
        )

        # ----------------------------------------------------
        # OFF gradient
        # ----------------------------------------------------

        off_gradient = get_gradient_vector(
            model,
            norm_loss_off,
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if ot_gradient.numel() != off_gradient.numel():

            raise RuntimeError(
                "OT and OFF gradient vectors still have "
                "different sizes: "
                f"{ot_gradient.numel()} vs "
                f"{off_gradient.numel()}"
            )

        # ----------------------------------------------------
        # Gradient norms
        # ----------------------------------------------------

        ot_norm = torch.norm(
            ot_gradient
        ).item()

        off_norm = torch.norm(
            off_gradient
        ).item()

        # ----------------------------------------------------
        # Gradient magnitude ratio
        # ----------------------------------------------------

        ratio = (
            off_norm
            / (ot_norm + 1e-12)
        )

        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        cosine = torch.nn.functional.cosine_similarity(
            ot_gradient.unsqueeze(0),
            off_gradient.unsqueeze(0),
            dim=1,
        ).item()

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        ot_gradient_norms.append(
            ot_norm
        )

        off_gradient_norms.append(
            off_norm
        )

        gradient_ratios.append(
            ratio
        )

        cosine_similarities.append(
            cosine
        )

        print(
            f"Batch {step + 1:02d} | "
            f"OT grad: {ot_norm:.6f} | "
            f"OFF grad: {off_norm:.6f} | "
            f"OFF/OT: {ratio:.2f}x | "
            f"Cosine similarity: {cosine:.4f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 65)
    print("GRADIENT CONFLICT DIAGNOSTIC")
    print("=" * 65)

    print(
        f"Average OT gradient norm:  "
        f"{np.mean(ot_gradient_norms):.6f}"
    )

    print(
        f"Average OFF gradient norm: "
        f"{np.mean(off_gradient_norms):.6f}"
    )

    print(
        f"Average OFF / OT ratio:     "
        f"{np.mean(gradient_ratios):.2f}x"
    )

    print(
        f"Average cosine similarity:  "
        f"{np.mean(cosine_similarities):.4f}"
    )

    print(
        f"Minimum cosine similarity:  "
        f"{np.min(cosine_similarities):.4f}"
    )

    print(
        f"Maximum cosine similarity:  "
        f"{np.max(cosine_similarities):.4f}"
    )

    # ========================================================
    # Interpretation
    # ========================================================

    average_cosine = np.mean(
        cosine_similarities
    )

    negative_fraction = np.mean(
        np.array(cosine_similarities) < 0
    )

    print("\nINTERPRETATION")
    print("-" * 65)

    if average_cosine < -0.2:

        print(
            "STRONG GRADIENT CONFLICT detected."
        )

        print(
            "OT and OFF frequently push the shared "
            "model in opposing directions."
        )

    elif average_cosine < 0:

        print(
            "MODERATE GRADIENT CONFLICT detected."
        )

        print(
            "Some opposition exists between OT and "
            "OFF gradients."
        )

    elif average_cosine < 0.2:

        print(
            "WEAK/NEUTRAL gradient alignment."
        )

        print(
            "The tasks are not strongly cooperating, "
            "but there is no strong conflict."
        )

    else:

        print(
            "POSITIVE gradient alignment."
        )

        print(
            "OT and OFF generally push the shared "
            "model in similar directions."
        )

    print(
        f"\nFraction of batches with negative "
        f"cosine similarity: "
        f"{negative_fraction * 100:.1f}%"
    )

    print("=" * 65)


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Same architecture sampling as previous experiments
    # --------------------------------------------------------

    random.seed(14)

    arch = sample_architecture()

    print(
        "Sampled architecture:",
        [
            layer["block_type"]
            for layer in arch
        ],
    )

    # --------------------------------------------------------
    # Same PyTorch seed
    # --------------------------------------------------------

    torch.manual_seed(42)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MultiTaskNASModel(
        arch
    )

    # --------------------------------------------------------
    # Run diagnostic
    # --------------------------------------------------------

    diagnose_gradient_conflict(
        model,
        batch_size=32,
    )
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torchview import draw_graph

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False

# Make imports work from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from JSSP.src.actor import Actor


def build_dummy_batch_states(
    batch_size=2,
    n_nodes_per_state=4,
    in_dim=3,
    device="cpu",
):
    """
    Build a minimal fake batch_states object with the attributes Actor.forward() expects.
    """
    total_nodes = batch_size * n_nodes_per_state

    x = torch.randn(total_nodes, in_dim, device=device)

    # Build identical small edge sets for each graph in the batch
    edge_index_pc_list = []
    edge_index_mc_list = []
    batch = []

    for b in range(batch_size):
        offset = b * n_nodes_per_state

        # precedence-chain style edges
        pc_src = torch.tensor([0, 1, 2], device=device) + offset
        pc_dst = torch.tensor([1, 2, 3], device=device) + offset
        edge_index_pc_list.append(torch.stack([pc_src, pc_dst], dim=0))

        # machine-conflict style edges
        mc_src = torch.tensor([0, 2], device=device) + offset
        mc_dst = torch.tensor([2, 3], device=device) + offset
        edge_index_mc_list.append(torch.stack([mc_src, mc_dst], dim=0))

        batch.extend([b] * n_nodes_per_state)

    edge_index_pc = torch.cat(edge_index_pc_list, dim=1)
    edge_index_mc = torch.cat(edge_index_mc_list, dim=1)
    batch = torch.tensor(batch, dtype=torch.long, device=device)

    return SimpleNamespace(
        x=x,
        edge_index_pc=edge_index_pc,
        edge_index_mc=edge_index_mc,
        batch=batch,
    )


def build_dummy_feasible_actions(batch_size=2):
    """
    Each state gets a small list of feasible [source_node, target_node] actions.
    These indices are local to each state, matching what Actor.forward expects.
    """
    feasible_actions = [
        [[0, 1], [1, 2], [2, 3]],
        [[0, 2], [1, 3]],
    ]
    return feasible_actions[:batch_size]


def visualize_with_torchview(model, batch_states, feasible_actions, out_dir, filename):
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("dot") is None:
        print("Skipping torchview export: Graphviz 'dot' not found.")
        return

    draw_graph(
        model,
        input_data=(batch_states, feasible_actions),
        expand_nested=True,
        depth=5,
        graph_name=filename,
        save_graph=True,
        directory=str(out_dir),
        filename=filename,
        device="cpu",
    )

    print(f"Saved torchview graph: {out_dir / (filename + '.png')}")


def visualize_with_torchviz(model, batch_states, feasible_actions, out_dir, filename):
    if not TORCHVIZ_AVAILABLE:
        print("Skipping torchviz export: torchviz not installed.")
        return

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        model.train()
        _, log_prob = model(batch_states, feasible_actions)

        # scalar for graph rendering
        loss_like = log_prob.sum()

        dot = make_dot(loss_like)
        dot.format = "png"
        dot.render(str(out_dir / filename), cleanup=True)

        print(f"Saved torchviz graph: {out_dir / (filename + '.png')}")
    except Exception as e:
        print(f"Skipping torchviz export for {filename}: {e}")


def main():
    print("Visualizing Actor architectures...")

    device = "cpu"
    in_dim = 3
    hidden_dim = 128
    embedding_l = 4
    policy_l = 3
    heads = 4
    dropout = 0.6

    out_dir = PROJECT_ROOT / "JSSP" / "model_diagrams"

    batch_states = build_dummy_batch_states(
        batch_size=2,
        n_nodes_per_state=4,
        in_dim=in_dim,
        device=device,
    )
    feasible_actions = build_dummy_feasible_actions(batch_size=2)

    for embedding_type in ["gin", "dghan", "gin+dghan"]:
        print(f"\nVisualizing Actor with embedding_type={embedding_type}")

        model = Actor(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            embedding_l=embedding_l,
            policy_l=policy_l,
            embedding_type=embedding_type,
            heads=heads,
            dropout=dropout,
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        visualize_with_torchview(
            model,
            batch_states,
            feasible_actions,
            out_dir,
            f"actor_{embedding_type}_torchview",
        )

        visualize_with_torchviz(
            model,
            batch_states,
            feasible_actions,
            out_dir,
            f"actor_{embedding_type}_torchviz",
        )


if __name__ == "__main__":
    main()
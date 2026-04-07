import os
import sys
from pathlib import Path

import dgl
import torch
from torchview import draw_graph
from torchviz import make_dot

# -------------------------------------------------------------------
# Make sure Python can import from the project root
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from DDI.src.models import DGLGraphModel


def build_dummy_graph(num_nodes: int = 6):
    """
    Create a small toy graph for visualization.
    """
    src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2])
    dst = torch.tensor([1, 2, 3, 4, 5, 0, 2, 4])

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    return g


def visualize_with_torchview(model, graph, features, out_dir: Path, filename: str, depth: int = 10):
    """
    Creates a structural model diagram with torchview.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_viz = draw_graph(
        model,
        input_data=(graph, features),
        expand_nested=True,
        depth=depth,
        graph_name=f"{filename}_depth{depth}",
        save_graph=True,
        directory=str(out_dir),
        filename=f"{filename}_depth{depth}",
        device=features.device,
    )

    # Optional: save a PNG as well if supported in your environment
    try:
        graph_viz.visual_graph.render(
            filename=f"{filename}_depth{depth}",
            directory=str(out_dir),
            format="png",
            cleanup=True,
        )
    except Exception as e:
        print(f"torchview render warning: {e}")


# def visualize_with_torchviz(model, graph, features, out_dir: Path, filename: str):
#     """
#     Creates an autograd computation graph with torchviz.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)

#     model.train()  # torchviz wants autograd graph
#     features = features.clone().requires_grad_(True)

#     output = model(graph, features)

#     # If output is a matrix, reduce it to a scalar so the graph is easier to render
#     loss_like = output.sum()

#     dot = make_dot(loss_like, params=dict(model.named_parameters()))
#     dot.format = "png"
#     dot.render(str(out_dir / filename), cleanup=True)


def main():
    print("Visualizing model architectures...")
    device = torch.device("cpu")

    # Match these to a realistic configuration for your model
    input_features = 8
    hidden_features = 128
    num_layers = 4
    output_features = 4

    out_dir = PROJECT_ROOT / "transfer_learning" / "model_diagrams"

    print("Visualizing DGLGraphModel with GIN layers...")
    print(f"Using device: {device}")
    print(f"Model config - input_features: {input_features}, hidden_features: {hidden_features}, num_layers: {num_layers}, output_features: {output_features}")

    model_gin = DGLGraphModel(
        input_features=input_features,
        hidden_features=hidden_features,
        output_features=output_features,
        num_layers=num_layers,
        use_gin=True,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model_gin.parameters())}")

    g2 = build_dummy_graph(num_nodes=6).to(device)
    x2 = torch.randn(g2.num_nodes(), input_features, device=device)

    print("Generating diagrams...")

    visualize_with_torchview(
        model_gin, g2, x2, out_dir, "dgl_graph_model_gin_torchview", depth=3,
    )
    #visualize_with_torchviz(model_gin, g2, x2, out_dir, "dgl_graph_model_gin_torchviz")

    print(f"Saved diagrams to: {out_dir}")


if __name__ == "__main__":
    main()

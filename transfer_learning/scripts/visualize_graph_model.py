import re
import sys
from pathlib import Path

import dgl
import torch
from torchview import draw_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from DDI.src.models import DGLGraphModel

# Torchview always appends "depth:i" to HTML node labels (see computation_graph.get_node_label).
_DEPTH_LABEL_HTML_RE = re.compile(r"<BR\s*/?>depth:\s*\d+", re.IGNORECASE)


def strip_torchview_depth_labels(computation_graph) -> None:
    """Remove depth:index lines from torchview HTML labels in the Graphviz source."""
    digraph = computation_graph.visual_graph
    if not getattr(digraph, "body", None):
        return
    digraph.body = [_DEPTH_LABEL_HTML_RE.sub("", line) for line in digraph.body]


def build_dummy_graph(num_nodes: int = 6):
    """
    Create a small toy graph for visualization.
    """
    src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2])
    dst = torch.tensor([1, 2, 3, 4, 5, 0, 2, 4]) # todo: make this more realistic

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    return g


def visualize_with_torchview(
    model,
    graph,
    features,
    out_dir: Path,
    filename: str,
    depth: int = 10,
    graph_dir: str = "TB",
    show_shapes: bool = False,
    show_depth_in_labels: bool = False,
):
    """
    Creates a structural model diagram with torchview.

    graph_dir: torchview / Graphviz layout — 'LR' (left-to-right, horizontal),
    'TB' (top-to-bottom, vertical), 'RL', or 'BT'.

    Tensor shapes are hidden (show_shapes=False) so nodes emphasize module names.

    Torchview always adds depth:index inside HTML labels; set show_depth_in_labels=False
    (default) to strip those before rendering. Use save_graph=False here so rendering
    happens once after stripping (torchview's save_graph would render too early).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_viz = draw_graph(
        model,
        input_data=(graph, features),
        expand_nested=True,
        depth=depth,
        graph_name=f"{filename}_depth{depth}",
        save_graph=False,
        directory=str(out_dir),
        filename=f"{filename}_depth{depth}",
        device=features.device,
        graph_dir=graph_dir,
        show_shapes=show_shapes,
    )

    if not show_depth_in_labels:
        strip_torchview_depth_labels(graph_viz)

    try:
        graph_viz.visual_graph.render(
            filename=f"{filename}_depth{depth}",
            directory=str(out_dir),
            format="png",
            cleanup=True,
        )
    except Exception as e:
        print(f"torchview render warning: {e}")


def main():
    print("Visualizing model architectures...")
    device = torch.device("cpu")

    # Match these to a realistic configuration for your model
    input_features = 8
    hidden_features = 128
    num_layers = 4
    output_features = 4

    show_shapes = False
    depths = [1, 2, 3, 4]
    # Diagram flow: "horizontal" (left-to-right) or "vertical" (top-to-bottom)
    diagram_layout = "horizontal"
    graph_dir = "LR" if diagram_layout == "horizontal" else "TB"

    out_dir = PROJECT_ROOT / "transfer_learning" / "model_diagrams"

    print("Visualizing DGLGraphModel with GIN layers...")
    print(f"Using device: {device}")
    print(
        f"Model config - input_features: {input_features}, hidden_features: {hidden_features}, "
        f"num_layers: {num_layers}, output_features: {output_features}, "
        f"diagram_layout: {diagram_layout} (graph_dir={graph_dir})"
    )

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

    for depth in depths:
        if depth > 1: 
            graph_dir = "TB" # todo: remove later 
        
        visualize_with_torchview(
            model_gin,
            g2,
            x2,
            out_dir,
            "dgl_graph_model_gin_torchview",
            depth=depth,
            graph_dir=graph_dir,
            show_shapes=show_shapes,
        )

    print(f"Saved diagrams to: {out_dir}")


if __name__ == "__main__":
    main()

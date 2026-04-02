import re

import torch


class PretrainHelper:
    @staticmethod
    def load_pretrained_state(path, map_location="cpu", remap=True):
        state = torch.load(path, map_location=map_location)

        if not isinstance(state, dict):
            raise ValueError("Expected a state_dict-like object.")
        
        if not remap:
            return state

        # Remap names
        remapped = {}

        for k, v in state.items():
            new_key = k.replace(".nn.", ".apply_func.")
            remapped[new_key] = v

        return remapped

    @staticmethod
    def infer_gin_architecture(state_dict, remap=True):
        net_name = "nn" if not remap else "apply_func"

        layer_ids = set()

        for key in state_dict.keys():
            m = re.match(r"GIN_layers\.(\d+)\.", key)
            if m:
                layer_ids.add(int(m.group(1)))

        if not layer_ids:
            raise ValueError("No GIN_layers.* keys found in checkpoint.")

        num_layers = max(layer_ids) + 1

        first_linear_weight = state_dict[f"GIN_layers.0.{net_name}.0.weight"]
        hidden_dim, input_dim = first_linear_weight.shape

        has_bn = any(f"GIN_layers.0.{net_name}.1.running_mean" in k for k in state_dict.keys())

        return {
            "num_layers": num_layers,
            "pretrained_input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "has_batchnorm": has_bn,
        }

    @staticmethod
    def load_matching_weights(model, pretrained_state, verbose=True):
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []

        for k, v in pretrained_state.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            else:
                skipped.append((k, tuple(v.shape), None))

        model_state.update(filtered_state)
        model.load_state_dict(model_state)

        if verbose:
            print(f"Loaded {len(filtered_state)} matching tensors.")
            if skipped:
                print(f"Skipped {len(skipped)} tensors:")
                for item in skipped:
                    print(" ", item)

        return model
import numpy as np
import torch


class TDCDDIEvaluator:
    def __init__(self, name):
        self.name = name
        self.eval_metric = 'rocauc'

    def _parse_and_check_input(self, input_dict):
        if 'y_pred_pos' not in input_dict:
            raise RuntimeError('Missing key of y_pred_pos')
        if 'y_pred_neg' not in input_dict:
            raise RuntimeError('Missing key of y_pred_neg')

        y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

        # Convert to numpy arrays immediately for metric calculation
        if torch is not None:
            if isinstance(y_pred_pos, torch.Tensor):
                y_pred_pos = y_pred_pos.detach().cpu().numpy()
            if isinstance(y_pred_neg, torch.Tensor):
                y_pred_neg = y_pred_neg.detach().cpu().numpy()
        
        y_pred_pos = np.asarray(y_pred_pos).reshape(-1)
        y_pred_neg = np.asarray(y_pred_neg).reshape(-1)

        return y_pred_pos, y_pred_neg

    def eval(self, input_dict):
        y_pred_pos, y_pred_neg = self._parse_and_check_input(input_dict)
        return self._eval_metrics(y_pred_pos, y_pred_neg)
    
    def _eval_metrics(self, y_pred_pos, y_pred_neg):
        # Ensure inputs are tensors on GPU
        # (Assuming y_pred_pos and y_pred_neg are already GPU Tensors)
        if not isinstance(y_pred_pos, torch.Tensor):
            y_pred_pos = torch.tensor(y_pred_pos)
        if not isinstance(y_pred_neg, torch.Tensor):
            y_pred_neg = torch.tensor(y_pred_neg)
        
        if y_pred_pos.numel() == 0 or y_pred_neg.numel() == 0:
            return {f'hits@{k}': 0.0 for k in [1, 3, 10, 20, 30, 40, 50]}

        # 1. Sort negative scores (on GPU)
        neg_sorted, _ = torch.sort(y_pred_neg)
        
        # 2. Binary search on GPU (equivalent to np.searchsorted)
        num_neg_smaller = torch.searchsorted(neg_sorted, y_pred_pos)
        
        # 3. Calculate ranks
        ranks = len(y_pred_neg) - num_neg_smaller + 1
        
        results = {}
        for k in [1, 3, 10, 20, 30, 40, 50]:
            # Calculate mean on GPU, move only the final float to CPU
            results[f'hits@{k}'] = torch.mean((ranks <= k).float()).item()

        return results

    @property
    def expected_input_format(self):
        return (
            "==== Expected input format ====\n"
            "{'y_pred_pos': Tensor/Array, 'y_pred_neg': Tensor/Array}\n"
            "y_pred_pos: scores for positive edges\n"
            "y_pred_neg: scores for negative edges"
        )

    @property
    def expected_output_format(self):
        return "{'hits@1', ..., 'hits@50'}"
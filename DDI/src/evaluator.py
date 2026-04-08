import torch


class TDCDDIEvaluator:
    def __init__(self, name):
        self.name = name

    def _parse_and_check_input(self, input_dict):
        if "y_pred_pos" not in input_dict:
            raise RuntimeError("Missing key of y_pred_pos")
        if "y_pred_neg" not in input_dict:
            raise RuntimeError("Missing key of y_pred_neg")

        y_pred_pos, y_pred_neg = input_dict["y_pred_pos"], input_dict["y_pred_neg"]

        # Keep everything in torch; preserve device when possible.
        if not isinstance(y_pred_pos, torch.Tensor):
            y_pred_pos = torch.as_tensor(y_pred_pos)
        if not isinstance(y_pred_neg, torch.Tensor):
            y_pred_neg = torch.as_tensor(y_pred_neg)

        y_pred_pos = y_pred_pos.reshape(-1)
        y_pred_neg = y_pred_neg.reshape(-1)

        return y_pred_pos, y_pred_neg

    def eval(self, input_dict):
        y_pred_pos, y_pred_neg = self._parse_and_check_input(input_dict)
        return self._eval_metrics(y_pred_pos, y_pred_neg)
    
    def _eval_metrics(self, y_pred_pos, y_pred_neg):
        if y_pred_pos.numel() == 0 or y_pred_neg.numel() == 0:
            return {f"hits@{k}": 0.0 for k in [1, 3, 10, 20, 30, 40, 50]}

        # Sort negative scores ascending; rank = 1 + #(negatives strictly greater than pos).
        # Use right=True so ties between pos and a negative are not miscounted (left=True
        # counts strict < only and breaks when pos equals a negative score).
        neg_sorted, _ = torch.sort(y_pred_neg)
        first_neg_gt_pos = torch.searchsorted(neg_sorted, y_pred_pos, right=True)
        ranks = len(y_pred_neg) - first_neg_gt_pos + 1
        
        results = {}
        for k in [1, 3, 10, 20, 30, 40, 50]:
            results[f"hits@{k}"] = torch.mean((ranks <= k).float()).item()

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
import torch

class MultitaskMetrics:
    def __init__(self, tasks, threshold=0.5):
        self.tasks = tasks
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        self.stats = {
            task: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'intersection': 0.0, 'union': 0.0, 'target_sum': 0.0, 'pred_sum': 0.0}
            for task in self.tasks
        }
        
    def update(self, preds_dict, targets_dict):
        for task in self.tasks:
            if task not in preds_dict or task not in targets_dict:
                continue
                
            raw_preds = preds_dict[task]
            
            # If the predictions are already probabilities [0, 1] (e.g. from Mask2Former), don't apply sigmoid again.
            if raw_preds.min() >= 0.0 and raw_preds.max() <= 1.0:
                probs = raw_preds
            else:
                probs = torch.sigmoid(raw_preds)
                
            preds = (probs > self.threshold).float()
            targets = targets_dict[task].float()
            
            # Interpolate preds if necessary
            if preds.shape[-2:] != targets.shape[-2:]:
                preds = torch.nn.functional.interpolate(preds, size=targets.shape[-2:], mode='nearest')
                
            intersection = (preds * targets).sum().item()
            union = preds.sum().item() + targets.sum().item() - intersection
            
            tp = intersection
            fp = preds.sum().item() - tp
            fn = targets.sum().item() - tp
            
            self.stats[task]['tp'] += tp
            self.stats[task]['fp'] += fp
            self.stats[task]['fn'] += fn
            self.stats[task]['intersection'] += intersection
            self.stats[task]['union'] += union
            self.stats[task]['target_sum'] += targets.sum().item()
            self.stats[task]['pred_sum'] += preds.sum().item()
            
    def compute(self):
        results = {}
        for task in self.tasks:
            s = self.stats[task]
            
            iou = s['intersection'] / (s['union'] + 1e-6)
            dice = (2. * s['intersection']) / (s['pred_sum'] + s['target_sum'] + 1e-6)
            precision = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
            recall = s['tp'] / (s['tp'] + s['fn'] + 1e-6)
            
            results[f"{task}_iou"] = iou
            results[f"{task}_dice"] = dice
            results[f"{task}_precision"] = precision
            results[f"{task}_recall"] = recall
            
        return results

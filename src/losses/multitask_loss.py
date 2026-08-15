import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Flatten
        logits = logits.flatten(1)
        targets = targets.flatten(1)
        
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(1)
        dice = (2. * intersection + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        return 1 - dice

class SetCriterion(nn.Module):
    """
    Hungarian Matching Loss for Mask2Former (Semantic Segmentation Configured).
    """
    def __init__(self, config):
        super().__init__()
        self.num_classes = len(config.LOSS.weights) # e.g. 3 (Building, Road, Water)
        # Weight configurations
        self.bce_weight = config.LOSS.bce_weight
        self.dice_weight = config.LOSS.dice_weight
        self.class_weight = config.LOSS.get('class_weight', 2.0)
        
        self.dice_loss = DiceLoss()
        # Class loss: cross entropy with a "no object" class
        # Background class is at index num_classes
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = config.LOSS.get('no_object_weight', 0.1)
        self.register_buffer('empty_weight', empty_weight)
        
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
    @torch.no_grad()
    def match(self, class_logits, mask_logits, targets_dict):
        """
        Bipartite matching between predictions and targets.
        """
        B, N, C = class_logits.shape
        indices = []
        
        out_prob = class_logits.flatten(0, 1).softmax(-1) # [B*N, C]
        out_mask = mask_logits.flatten(0, 1) # [B*N, H/4, W/4]
        task_names = list(targets_dict.keys()) # ['building', 'road', 'water']
        target_shape = targets_dict[task_names[0]].shape[-2:]
        
        out_mask = F.interpolate(out_mask.unsqueeze(1), size=target_shape, mode='bilinear', align_corners=False).squeeze(1) # [B*N, H, W]
        out_mask_flat = out_mask.flatten(1) # [B*N, H*W]
        
        for b in range(B):
            tgt_labels = []
            tgt_masks = []
            
            for task_id, task_name in enumerate(task_names):
                if task_name not in targets_dict:
                    continue
                tgt_mask = targets_dict[task_name][b, 0] # [H, W]
                if tgt_mask.sum() > 0:
                    tgt_labels.append(task_id)
                    tgt_masks.append(tgt_mask.flatten())
                    
            if len(tgt_labels) == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue
                
            tgt_labels = torch.tensor(tgt_labels, dtype=torch.int64, device=class_logits.device)
            tgt_masks = torch.stack(tgt_masks) # [num_targets, H*W]
            
            b_out_prob = out_prob[b * N : (b + 1) * N] # [N, C]
            b_out_mask = out_mask_flat[b * N : (b + 1) * N] # [N, H*W]
            
            # Cost Matrix
            cost_class = -b_out_prob[:, tgt_labels]
            
            b_out_mask_sigmoid = b_out_mask.sigmoid()
            
            with torch.amp.autocast('cuda', enabled=False):
                cost_mask = F.binary_cross_entropy_with_logits(
                    b_out_mask.float().unsqueeze(1).expand(-1, len(tgt_labels), -1),
                    tgt_masks.float().unsqueeze(0).expand(N, -1, -1),
                    reduction='none'
                ).mean(-1)
            
            numerator = 2 * torch.einsum("nc,mc->nm", b_out_mask_sigmoid.float(), tgt_masks.float())
            denominator = b_out_mask_sigmoid.float().sum(-1).unsqueeze(1) + tgt_masks.float().sum(-1).unsqueeze(0)
            cost_dice = 1 - (numerator / (denominator + 1e-6))
            
            C_mat = (self.class_weight * cost_class + 
                     self.bce_weight * cost_mask + 
                     self.dice_weight * cost_dice)
                     
            if torch.isnan(C_mat).any() or torch.isinf(C_mat).any():
                print(f"[WARNING] NaN/Inf detected in Cost Matrix! Sanitizing...")
                C_mat = torch.nan_to_num(C_mat, nan=100.0, posinf=100.0, neginf=-100.0)
                
            C_mat = C_mat.cpu().numpy()
            
            src_ind, tgt_ind = linear_sum_assignment(C_mat)
            indices.append((torch.as_tensor(src_ind, dtype=torch.int64), torch.as_tensor(tgt_ind, dtype=torch.int64)))
            
        return indices
        
    def forward(self, preds_dict, targets_dict):
        """
        preds_dict: 
            'pred_logits': list of [B, N, num_classes+1]
            'pred_masks': list of [B, N, H/4, W/4]
        targets_dict:
            {'building': [B, 1, H, W], ...}
        """
        total_loss = 0
        loss_dict = {}
        
        task_names = list(targets_dict.keys())
        B = preds_dict['pred_logits'][0].shape[0]
        
        for layer_idx, (class_logits, mask_logits) in enumerate(zip(preds_dict['pred_logits'], preds_dict['pred_masks'])):
            
            indices = self.match(class_logits, mask_logits, targets_dict)
            src_idx = self._get_src_permutation_idx(indices)
            
            tgt_labels_list = []
            tgt_masks_list = []
            
            for b in range(B):
                b_tgt_ind = indices[b][1]
                b_valid_labels = []
                b_valid_masks = []
                for task_id, task_name in enumerate(task_names):
                    if task_name not in targets_dict:
                        continue
                    tgt_mask = targets_dict[task_name][b, 0]
                    if tgt_mask.sum() > 0:
                        b_valid_labels.append(task_id)
                        b_valid_masks.append(tgt_mask)
                
                if len(b_valid_labels) > 0:
                    b_valid_labels = torch.tensor(b_valid_labels, dtype=torch.int64, device=class_logits.device)
                    b_valid_masks = torch.stack(b_valid_masks)
                    
                    tgt_labels_list.append(b_valid_labels[b_tgt_ind])
                    tgt_masks_list.append(b_valid_masks[b_tgt_ind])
            
            target_classes = torch.full(class_logits.shape[:2], self.num_classes, dtype=torch.int64, device=class_logits.device)
            if len(tgt_labels_list) > 0:
                target_classes[src_idx] = torch.cat(tgt_labels_list)
            
            loss_class = F.cross_entropy(class_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
            
            if len(tgt_masks_list) > 0:
                matched_mask_logits = mask_logits[src_idx]
                target_masks = torch.cat(tgt_masks_list).float()
                
                matched_mask_logits = F.interpolate(matched_mask_logits.unsqueeze(1).float(), size=target_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                
                loss_bce = F.binary_cross_entropy_with_logits(matched_mask_logits, target_masks)
                loss_dice = self.dice_loss(matched_mask_logits, target_masks).mean()
            else:
                loss_bce = mask_logits.sum() * 0
                loss_dice = mask_logits.sum() * 0
                
            layer_loss = self.class_weight * loss_class + self.bce_weight * loss_bce + self.dice_weight * loss_dice
            total_loss += layer_loss
            
            if layer_idx == len(preds_dict['pred_logits']) - 1:
                loss_dict['class_loss'] = loss_class
                loss_dict['bce_loss'] = loss_bce
                loss_dict['dice_loss'] = loss_dice
                
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

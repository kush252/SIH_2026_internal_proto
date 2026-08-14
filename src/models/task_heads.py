import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase1_encoder_loader import load_phase1_encoder
from .mask2former import TransformerEncoderPixelDecoder, TrueMask2FormerDecoder

class Phase2MultiTaskModel(nn.Module):
    """
    True Mask2Former Architecture:
    Swin-T Encoder -> Pixel Decoder -> Transformer Decoder (Set Prediction)
    """
    def __init__(self, config):
        super().__init__()
        
        # 1. Load Pretrained Encoder
        self.encoder = load_phase1_encoder(config)
        
        # 2. Pixel Decoder (Transformer FPN)
        embed_dim = config.MODEL.encoder.embed_dim
        in_channels_list = [
            embed_dim, 
            embed_dim * 2, 
            embed_dim * 4, 
            embed_dim * 8
        ]
        hidden_dim = config.MODEL.decoder.hidden_dim
        self.pixel_decoder = TransformerEncoderPixelDecoder(in_channels_list, hidden_dim)
        
        # 3. Mask2Former Transformer Decoder
        # Target classes: Building, Road, Water
        num_classes = len(config.LOSS.weights)
        self.transformer_decoder = TrueMask2FormerDecoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=config.MODEL.decoder.num_queries,
            num_layers=config.MODEL.decoder.dec_layers,
            nheads=config.MODEL.decoder.nheads
        )
        
    def forward(self, x):
        """
        x: [B, 3, H, W]
        Returns: 
            Dictionary containing:
            "pred_logits": list of [B, N, num_classes + 1] (one per layer)
            "pred_masks": list of [B, N, H/4, W/4] (one per layer)
        """
        # Get Swin-T multi-scale features
        features = self.encoder(x)
        
        # Format features to B C H W
        formatted_features = []
        for feat in features:
            if feat.dim() == 4:
                if feat.shape[-1] in [96, 192, 384, 768]:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
            formatted_features.append(feat)
            
        # 1. Pixel Decoder
        mask_features, multi_scale_features = self.pixel_decoder(formatted_features)
        
        # 2. Transformer Decoder
        outputs_class, outputs_mask = self.transformer_decoder(multi_scale_features, mask_features)
        
        return {
            "pred_logits": outputs_class,
            "pred_masks": outputs_mask
        }

    @torch.no_grad()
    def semantic_inference(self, x, class_names):
        """
        Converts Mask2Former set predictions into standard semantic segmentation logits.
        class_names: list of class names in order, e.g. ['building', 'road', 'water']
        """
        preds = self.forward(x)
        # Use final layer predictions
        class_logits = preds['pred_logits'][-1] # [B, N, num_classes + 1]
        mask_logits = preds['pred_masks'][-1]   # [B, N, H/4, W/4]
        
        class_probs = class_logits.softmax(-1)  # [B, N, num_classes + 1]
        mask_probs = mask_logits.sigmoid()      # [B, N, H/4, W/4]
        
        # We need a unified semantic tensor: [B, num_classes, H/4, W/4]
        # Multiply class_probs [B, N, C, 1, 1] with mask_probs [B, N, 1, H, W]
        # Then sum over queries (N)
        semantic_logits = torch.einsum("bnc,bnhw->bchw", class_probs, mask_probs)
        
        # Exclude background class (the last channel)
        semantic_logits = semantic_logits[:, :-1]
        
        # Upsample to original resolution
        semantic_logits = F.interpolate(semantic_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Convert to dict format expected by metrics and visualization
        out_dict = {}
        for i, name in enumerate(class_names):
            out_dict[name] = semantic_logits[:, i:i+1] # [B, 1, H, W]
            
        return out_dict

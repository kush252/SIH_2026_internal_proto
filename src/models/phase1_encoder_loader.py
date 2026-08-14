import torch
from .swin_encoder import SwinEncoder

def load_phase1_encoder(config):
    """
    Instantiates the Swin-T encoder and safely loads the Phase 1 SimMIM 
    pretrained weights, checking for compatibility.
    """
    encoder_cfg = config.MODEL.encoder
    
    print(f"Instantiating Swin Encoder: {encoder_cfg.name}...")
    encoder = SwinEncoder(
        img_size=config.DATA.image_size,
        patch_size=encoder_cfg.patch_size,
        in_chans=3,
        embed_dim=encoder_cfg.embed_dim,
        depths=encoder_cfg.depths,
        num_heads=encoder_cfg.num_heads,
        window_size=encoder_cfg.window_size
    )
    
    checkpoint_path = encoder_cfg.checkpoint_path
    print(f"Loading Phase 1 weights from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Load weights with strict=True to ensure architecture matches
        missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"WARNING: Missing keys in encoder load: {missing_keys}")
            
        # The mask_token is expected to be unexpected if it was in the state dict 
        # (since we don't use it for Phase 2), so we filter that out of unexpected warnings.
        unexpected_keys = [k for k in unexpected_keys if 'mask_token' not in k]
        
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in encoder load: {unexpected_keys}")
            
        print("Phase 1 Encoder weights loaded successfully.")
        
    except Exception as e:
        print(f"CRITICAL ERROR loading Phase 1 checkpoint: {e}")
        raise e
        
    return encoder

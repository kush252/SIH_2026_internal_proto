import torch
import argparse
import os

def extract_encoder(best_pt_path, output_path):
    print(f"Loading {best_pt_path}...")
    checkpoint = torch.load(best_pt_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    print("Extracting encoder weights...")
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            # Remove the 'encoder.' prefix
            encoder_state[k.replace('encoder.', '')] = v
            
    print(f"Found {len(encoder_state)} encoder tensors.")
    
    torch.save(encoder_state, output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="best.pt", help="Path to best.pt")
    parser.add_argument("--output", type=str, default="swin_t_simmim_encoder.pt", help="Path to save output")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Could not find {args.input}. Please provide the correct path.")
    else:
        extract_encoder(args.input, args.output)

import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nips_task34")
    parser.add_argument("--model_name", type=str, default="TransformerKT")
    parser.add_argument("--emb_type", type=str, default="encoder_only", 
                      help="Type of transformer architecture")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Model parameters (aligned with TRANSFORMERKT class)
    parser.add_argument("--emb_size", type=int, default=256, 
                       help="Embedding dimension (formerly d_model)")
    parser.add_argument("--d_ff", type=int, default=512, 
                       help="Dimension of feedforward network")
    parser.add_argument("--nhead", type=int, default=8, 
                       help="Number of attention heads (formerly num_attn_heads)")
    parser.add_argument("--num_layers", type=int, default=4, 
                       help="Number of transformer layers (formerly n_blocks)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    params = vars(args)
    main(params)

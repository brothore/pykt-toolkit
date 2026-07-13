import argparse

try:
    from .wandb_train import main
except ImportError:
    from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="akt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--random_rev", type=float, default=0.0, help="Probability of label reversal (poisoning)")

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--use_trained", type=int, default=0)
    parser.add_argument("--use_also", type=int, default=0)
    parser.add_argument("--also_grouping_mode", type=str, default="none")
    parser.add_argument("--also_n_groups", type=int, default=None)
    parser.add_argument("--also_mode", type=str, choices=["optimistic", "descent-ascent"], default="optimistic")
    parser.add_argument("--also_alpha", type=float, default=1.0)
    parser.add_argument("--also_lr", type=float, default=None)
    parser.add_argument("--also_weight_decay", type=float, default=None)
    parser.add_argument("--also_pi_lr", type=float, default=1e-3)
    parser.add_argument("--also_pi_decay", type=float, default=1e-2)
    parser.add_argument("--also_loss_scale", type=float, default=None)
    parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)

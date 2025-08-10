import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nips_task34")
    parser.add_argument("--model_name", type=str, default="abqr")
    parser.add_argument("--emb_type", type=str, default="qid_ni34")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--d_model", type=int, default=256)

    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--step_size", type=float, default=3e-2)
    parser.add_argument("--step_m", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=15.0)
    parser.add_argument("--mm", type=float, default=0.99)

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    print("+"*100)

    main(params)

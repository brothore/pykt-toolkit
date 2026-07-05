import argparse

try:
    from .wandb_train import main
except ImportError:
    from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--use_trained", type=int, default=0)
    parser.add_argument("--use_also", type=int, default=0)
    parser.add_argument("--also_grouping_mode", type=str, default="none")
    parser.add_argument("--also_n_groups", type=int, default=None)
    parser.add_argument("--also_pi_lr", type=float, default=1e-3)
    parser.add_argument("--also_pi_decay", type=float, default=1e-2)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--random_rev", type=float, default=0.0, help="Probability of label reversal (poisoning)")
    
    args = parser.parse_args()

    params = vars(args)
    print("=" * 80)
    print("[EXP_START] DKT baseline experiment")
    print(f"dataset_name={params['dataset_name']}")
    print(f"model_name={params['model_name']}")
    print(f"emb_type={params['emb_type']}")
    print(f"seed={params['seed']}")
    print(f"fold={params['fold']}")
    print(f"dropout={params['dropout']}")
    print(f"emb_size={params['emb_size']}")
    print(f"learning_rate={params['learning_rate']}")
    print(f"batch_size={params['batch_size']}")
    print(f"num_epochs={params['num_epochs']}")
    print(f"use_wandb={params['use_wandb']}")
    print(f"add_uuid={params['add_uuid']}")
    print("=" * 80)
    main(params)

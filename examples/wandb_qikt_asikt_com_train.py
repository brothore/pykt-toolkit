import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #dataset config
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--fold", type=int, default=0)

    # train config
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    
    parser.add_argument("--random_rev", type=float, default=0.0, help="Probability of label reversal (poisoning)")
    parser.add_argument("--random_trunc", type=float, default=0.0, help="Probability of random sequence truncation")
    parser.add_argument("--random_dup", type=float, default=0.0, help="Probability of random sequence duplication")
    parser.add_argument("--random_shuf", type=float, default=0.0, help="Probability of random sequence shuffling")
    parser.add_argument("--copy_ratio", type=float, default=0, help="Ratio of steps to duplicate when random_dup is triggered")
    parser.add_argument("--drop_ratio", type=float, default=0, help="Ratio of steps to drop when discrete truncation is triggered")    #log config & save config

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved_model")

    # model config
    parser.add_argument("--model_name", type=str, default="qikt_asikt_com")
    parser.add_argument("--emb_type", type=str, default="iekt")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--mlp_layer_num", type=int, default=2)

    parser.add_argument("--loss_q_all_lambda", type=float, default=0)
    parser.add_argument("--loss_c_all_lambda", type=float, default=0)
    parser.add_argument("--loss_q_next_lambda", type=float, default=0)
    parser.add_argument("--loss_c_next_lambda", type=float, default=0)
    
    parser.add_argument("--output_q_all_lambda", type=float, default=1)
    parser.add_argument("--output_c_all_lambda", type=float, default=1)
    parser.add_argument("--output_q_next_lambda", type=float, default=0)
    parser.add_argument("--output_c_next_lambda", type=float, default=1)
    parser.add_argument("--num_attn_head", type=float, default=1)
    
    parser.add_argument("--output_mode", type=str, default="an")
    parser.add_argument("--version", type=str, default="an")
    parser.add_argument("--use_trained", type=int, default=0)
    args = parser.parse_args()

    params = vars(args)
    remove_keys = ['output_mode'] + [x for x in params.keys() if "lambda" in x]
    other_config = {}
    for k in remove_keys:
        other_config[k] = params[k]
        del params[k]
    params['other_config'] = other_config
    main(params)

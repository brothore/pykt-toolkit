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
    parser.add_argument("--model_name", type=str, default="qikt_mamba")
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
    parser.add_argument("--use_trained", type=int, default=1)
    parser.add_argument("--history_response_dropout", type=float, default=0.0)
    parser.add_argument("--item_residual", type=int, choices=[0, 1], default=0)
    parser.add_argument("--separate_mamba_branches", type=int, choices=[0, 1], default=0)
    parser.add_argument("--concept_memory_residual", type=int, choices=[0, 1], default=0)
    parser.add_argument("--pairwise_loss_weight", type=float, default=0.0)
    parser.add_argument("--head_fusion_residual", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fusion_gate_scale", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", type=int, choices=[0, 1], default=0)
    parser.add_argument("--anchor_kl_weight", type=float, default=0.0)
    parser.add_argument("--causal_adapter_mode", choices=['off','question','prefix','multiscale','supported','uncertain','disagreement','low_support','state_only','state_multiscale','state_interaction'], default='off')
    parser.add_argument("--causal_adapter_scale", type=float, default=.25)
    parser.add_argument("--causal_adapter_kl", type=float, default=0.)
    parser.add_argument("--retrieval_adapter_mode", choices=['off','content','kc_bias','multi_scale'], default='off')
    parser.add_argument("--retrieval_adapter_scale", type=float, default=.25)
    parser.add_argument("--retrieval_adapter_kl", type=float, default=0.)
    parser.add_argument('--ukt_transfer_mode', choices=['off','mean','w2','w2_legacy'], default='off')
    parser.add_argument('--ukt_transfer_scale', type=float, default=.1)
    parser.add_argument('--ukt_transfer_fusion', choices=['additive','uncertainty'], default='additive')
    parser.add_argument('--ukt_transfer_freeze', type=int, choices=[0,1], default=0)
    parser.add_argument('--ukt_rasch_mode', choices=['off','input_scalar','input_vector','input_additive','query_vector','output_vector'], default='off')
    parser.add_argument('--ukt_rasch_scale', type=float, default=1.)
    parser.add_argument('--ukt_rasch_freeze', type=int, choices=[0,1], default=0)
    parser.add_argument('--ukt_rasch_warmstart', type=int, choices=[0,1], default=0)
    parser.add_argument('--reference_fusion_mode', choices=['off','cic_question','cic_concept','cic_both','time_input','time_decode','time_gate','time_input_decode','time_sparse_input','alibi_logit','alibi_question','alibi_concept','scalar_query','shared_history','shared_scalar','scalar_additive_query','scalar_additive_input'], default='off')
    parser.add_argument('--reference_fusion_scale', type=float, default=.1)
    parser.add_argument('--reference_fusion_bias', choices=['alibi','none'], default='alibi')
    parser.add_argument('--reference_cic_kernel', type=int, default=3)
    parser.add_argument('--reference_cic_context', choices=['past','current'], default='past')
    parser.add_argument('--reference_history_weight', type=float, default=0.)
    args = parser.parse_args()

    params = vars(args)
    remove_keys = ['output_mode', 'history_response_dropout', 'item_residual', 'separate_mamba_branches', 'concept_memory_residual', 'pairwise_loss_weight', 'head_fusion_residual', 'fusion_gate_scale', 'freeze_backbone', 'anchor_kl_weight'] + [x for x in params.keys() if "lambda" in x]
    other_config = {}
    remove_keys += ['causal_adapter_mode', 'causal_adapter_scale', 'causal_adapter_kl']
    remove_keys += ['retrieval_adapter_mode', 'retrieval_adapter_scale', 'retrieval_adapter_kl']
    remove_keys += ['ukt_transfer_mode','ukt_transfer_scale','ukt_transfer_fusion','ukt_transfer_freeze','ukt_rasch_mode','ukt_rasch_scale','ukt_rasch_freeze','ukt_rasch_warmstart']
    remove_keys += ['reference_fusion_mode','reference_fusion_scale','reference_fusion_bias','reference_cic_kernel','reference_cic_context','reference_history_weight']
    for k in remove_keys:
        other_config[k] = params[k]
        del params[k]
    params['other_config'] = other_config
    main(params)

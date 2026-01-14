que_type_models = ["iekt","qdkt","qikt","lpkt", "rkt", "promptkt", "qikt_mamba", "qikt_lpkt", "qikt_dimkt", "qikt_iekt", "qikt_iekt_low_dropout", "qikt_iekt_train", "qikt_iekt_mask", "qikt_iekt_dual", "qikt_iekt_dual_actor", "qikt_iekt_dual_gae", "qikt_iekt_dual_ppo", "qikt_asikt", "qikt_asikt_com", "qikt_iekt_dual_gae_v2", "qikt_iekt_dual_gae_v3"]

qikt_ab_models = ["qikt_ab_a+b+c","qikt_ab_a+b+c+irt","qikt_ab_a+b+irt","qikt_ab_a+c+irt","qikt_ab_a+irt","qikt_ab_b+irt"]

que_type_models += qikt_ab_models

needs_uid_models = ["balance_dkt","balance_akt","long_dkt"]

predict_after_train = 2

random_rev = 1.0
import torch
import numpy as np
import os

# 保持通用的库导入
# ⚠️ 注意: os、torch和numpy是基础库，通常保留在文件顶部以供全局使用。

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        from .dkt import DKT
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "mamba_dkt":
        from .mamba_dkt import MAMBA_DKT
        model = MAMBA_DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "mult_dataset_dkt":
        from .mult_dataset_dkt import MULT_DATASET_DKT
        model = MULT_DATASET_DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "long_dkt":
        from .long_dkt import LONG_DKT
        model = LONG_DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "balance_dkt":
        from .balance_dkt import BALANCE_DKT
        model = BALANCE_DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        from .dkt_plus import DKTPlus
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        from .dkvmn import DKVMN
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        from .deep_irt import DeepIRT
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        from .sakt import SAKT
        model = SAKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        from .saint import SAINT
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        from .dkt_forget import DKTForget
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        from .akt import AKT

        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "multi_dataset_akt":
        from .multi_dataset_akt import MULTI_DATASET_AKT
        model = MULTI_DATASET_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "TransformerKT":
        from .TransformerKT import TRANSFORMERKT
        model = TRANSFORMERKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "qwen":
        from .qwen import QWEN
        model = QWEN(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "llm":
        from .llm import LLM
        model = LLM()
    elif model_name == "balance_akt":
        from .balance_akt import BALANCE_AKT
        model = BALANCE_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "mpllm":
        from .mpllm import MPLLM
        model = MPLLM()
    elif model_name == "Transformer_template":
        from .Transformer_template import TRANSFORMER_TEMPLATE
        model = TRANSFORMER_TEMPLATE(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lefokt_akt":
        from .lefokt_akt import LEFOKT_AKT
        model = LEFOKT_AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "extrakt":
        from .extrakt import extraKT
        model = extraKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "folibikt":
        from .folibikt import folibiKT
        model = folibiKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        from .kqn import KQN
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        from .atkt import ATKT
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "at_dkt":
        from .at_dkt import AT_DKT
        model = AT_DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    # MAMBA_ATAKT被注释掉了，所以我们在这里也注释掉它
    # elif model_name == "mamba_atakt":
    #     from .mamba_atakt import MAMBA_ATAKT
    #     model = MAMBA_ATAKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        from .atkt import ATKT
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        from .gkt import GKT
        from .gkt_utils import get_gkt_graph
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"],
                                  data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config, graph=graph, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        from .lpkt import LPKT
        from .lpkt_utils import generate_qmatrix
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        from .skvmn import SKVMN
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "hawkes":
        from .hawkes import HawkesKT
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions and concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config,emb_type=emb_type)
        model = model.double()
        model.apply(model.init_weights)
        model = model.to(device)
    elif model_name == "hawkes_lstm":
        from .hawkes_lstm import HawkesLSTM
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions and concepts! but the dataset has no both")
            return None
        model = HawkesLSTM(data_config["num_c"], data_config["num_q"], **model_config,emb_type=emb_type)
        model = model.double()
        model.apply(model.init_weights)
        model = model.to(device)
    elif model_name == "hawkes_mamba":
        from .hawkes_mamba import HawkesMamba
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions and concepts! but the dataset has no both")
            return None
        model = HawkesMamba(data_config["num_c"], data_config["num_q"], **model_config,emb_type=emb_type)
        model = model
        model.apply(model.init_weights)
        model = model.to(device)
    elif model_name == "iekt":
        from .iekt import IEKT
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                     max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], device=device).to(device)
    elif model_name == "qdkt":
        from .qdkt import QDKT
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                     max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], device=device).to(device)
    elif model_name == "qikt":
        from .qikt import QIKT
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                     max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], device=device).to(device)
    elif model_name == "qikt_mamba":
        from .qikt_mamba import QIKT_MAMBA
        model = QIKT_MAMBA(num_q=data_config['num_q'], num_c=data_config['num_c'],
                     max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], device=device).to(device)
    elif model_name == "atdkt":
        from .atdkt import ATDKT
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt_time":
        from .bakt_time import BAKTTime
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    # DBAKT被注释掉了
    # elif model_name == "dbakt":
    #     from .dbakt import DBAKT
    #     model = DBAKT(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "simplekt":
        from .simplekt import simpleKT
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rekt":
        from .rekt import ReKT
        model = ReKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type).to(device)
    elif model_name == "stablekt":
        from .stablekt import stableKT
        model = stableKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dimkt":
        from .dimkt import DIMKT
        model = DIMKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sparsekt":
        from .sparsekt import sparseKT
        model = sparseKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "rkt":
        from .rkt import RKT
        model = RKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "cskt":
        from .cskt import CSKT
        model = CSKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "fluckt":
        from .fluckt import FlucKT
        model = FlucKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "ukt":
        from .ukt import UKT
        model = UKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "hcgkt":
        from .hcgkt import HCGKT
        model = HCGKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "abqr":
        from .abqr import ABQR
        model = ABQR(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "robustkt":
        from .robustkt import Robustkt
        model = Robustkt(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dtransformer":
        from .dtransformer import DTransformer
        model = DTransformer(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type,
                             emb_path=data_config["emb_path"]).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    if model_name not in ["llm", "mpllm"]:
        net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
        model.load_state_dict(net)
    return model
import os
import argparse
import json
import copy
import torch
import pandas as pd
# from pykt.config import ERR_PATH,stu_pk,SET_TARGET_STU
from pykt.models import evaluate, evaluate_question, load_model,evaluate_llm_question_async
from pykt.datasets import init_test_datasets
#只进行windows_acc/auc_late的predict
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
import asyncio
def visualize_and_save(data, title, xlabel, ylabel, filename, save_path):
    """通用可视化函数，保存图片到本地"""
    plt.figure(figsize=(10, 6))
    if isinstance(data, dict):
        for label, d in data.items():
            plt.plot(d, label=label)
        plt.legend()
    else:
        plt.plot(data)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    print(f"Saved plot to {full_path}")
    plt.close()

def analyze_lstm_robustness(clean_model, robust_model, data_loader, save_dir):
    """
    分析原始模型和对抗训练过的LSTM模型的鲁棒性。
    
    Args:
        clean_model (AT_DKT): 原始训练好的模型。
        robust_model (AT_DKT): 对抗训练好的模型。
        data_loader (DataLoader): 包含数据的DataLoader。
        save_dir (str): 保存可视化图表的目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("开始参数分析...")
    
    # 门控偏置分析
    clean_forget_bias = clean_model.lstm_layer.bias_ih_l0.chunk(4)[1].data + clean_model.lstm_layer.bias_hh_l0.chunk(4)[1].data
    robust_forget_bias = robust_model.lstm_layer.bias_ih_l0.chunk(4)[1].data + robust_model.lstm_layer.bias_hh_l0.chunk(4)[1].data
    
    plt.figure(figsize=(12, 6))
    sns.histplot(clean_forget_bias.cpu().numpy(), color='blue', label='原始模型', kde=True, stat="density", linewidth=0)
    sns.histplot(robust_forget_bias.cpu().numpy(), color='red', label='对抗训练模型', kde=True, stat="density", linewidth=0)
    plt.title("遗忘门偏置 (Bias) 分布比较")
    plt.xlabel("偏置值")
    plt.ylabel("密度")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'forget_gate_bias_distribution.png'))
    plt.close()
    
    # 门控权重绝对值分布
    clean_W_if = clean_model.lstm_layer.weight_ih_l0.chunk(4)[1].data
    robust_W_if = robust_model.lstm_layer.weight_ih_l0.chunk(4)[1].data
    
    plt.figure(figsize=(12, 6))
    sns.histplot(clean_W_if.abs().cpu().numpy().flatten(), color='blue', label='原始模型', kde=True, stat="density", linewidth=0)
    sns.histplot(robust_W_if.abs().cpu().numpy().flatten(), color='red', label='对抗训练模型', kde=True, stat="density", linewidth=0)
    plt.title("遗忘门权重 (W_if) 绝对值分布比较")
    plt.xlabel("权重绝对值")
    plt.ylabel("密度")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'forget_gate_weight_abs_distribution.png'))
    plt.close()

    print("开始动态分析...")
    clean_model.eval()
    robust_model.eval()

    # 替换LSTM层为可分析版本
    input_size = robust_model.lstm_layer.input_size
    hidden_size = robust_model.lstm_layer.hidden_size
    analyzable_clean_lstm = AnalyzableLSTM(input_size, hidden_size)
    analyzable_clean_lstm.copy_weights_from_nn_lstm(clean_model.lstm_layer)
    clean_model.lstm_layer = analyzable_clean_lstm.to(device)
    
    analyzable_robust_lstm = AnalyzableLSTM(input_size, hidden_size)
    analyzable_robust_lstm.copy_weights_from_nn_lstm(robust_model.lstm_layer)
    robust_model.lstm_layer = analyzable_robust_lstm.to(device)

    # 存储动态分析结果
    clean_state_dists = []
    robust_state_dists = []
    clean_grad_norms = []
    robust_grad_norms = []
    clean_forget_activations = []
    robust_forget_activations = []

    # 仅取一个batch进行分析
    for batch_idx, (q, r, rshft, sm) in enumerate(data_loader):
        q, r, rshft, sm = q.to(device), r.to(device), rshft.to(device), sm.to(device)

        # ----------------- 生成对抗扰动 -----------------
        # 这部分逻辑与你的训练代码一致
        h_clean, features_clean, _, _ = robust_model(q, r)
        y_clean = robust_model.get_output(h_clean, q)
        loss_clean = cal_loss(robust_model, [y_clean], r, rshft, sm)
        
        features_grad = grad(loss_clean, features_clean, retain_graph=True)[0].data
        p_adv = torch.FloatTensor(robust_model.epsilon * _l2_normalize_adv(features_grad)).to(device)

        # ----------------- 轨迹分析 -----------------
        # 干净输入
        _, _, clean_c_clean, _ = clean_model(q, r)
        _, _, robust_c_clean, _ = robust_model(q, r)
        
        # 扰动输入
        _, _, clean_c_adv, _ = clean_model(q, r, perturbation=p_adv)
        _, _, robust_c_adv, _ = robust_model(q, r, perturbation=p_adv)

        # 计算状态距离（欧氏距离）
        for t in range(q.shape[1]):
            clean_dist = torch.norm(clean_c_clean[:, t, :] - clean_c_adv[:, t, :], dim=1).mean().item()
            robust_dist = torch.norm(robust_c_clean[:, t, :] - robust_c_adv[:, t, :], dim=1).mean().item()
            clean_state_dists.append(clean_dist)
            robust_state_dists.append(robust_dist)

        # ----------------- 梯度流分析 -----------------
        clean_model.zero_grad()
        h_clean, _, _, _ = clean_model(q, r)
        y_clean = clean_model.get_output(h_clean, q)
        loss_clean = cal_loss(clean_model, [y_clean], r, rshft, sm)
        loss_clean.backward(retain_graph=True)
        clean_grad_norm_clean = sum([p.grad.norm().item() for p in clean_model.parameters() if p.grad is not None])
        clean_model.zero_grad()
        
        h_adv, _, _, _ = clean_model(q, r, perturbation=p_adv)
        y_adv = clean_model.get_output(h_adv, q)
        loss_adv = cal_loss(clean_model, [y_adv], r, rshft, sm)
        loss_adv.backward()
        clean_grad_norm_adv = sum([p.grad.norm().item() for p in clean_model.parameters() if p.grad is not None])
        clean_grad_norms.append(abs(clean_grad_norm_clean - clean_grad_norm_adv))
        
        robust_model.zero_grad()
        h_clean, _, _, _ = robust_model(q, r)
        y_clean = robust_model.get_output(h_clean, q)
        loss_clean = cal_loss(robust_model, [y_clean], r, rshft, sm)
        loss_clean.backward(retain_graph=True)
        robust_grad_norm_clean = sum([p.grad.norm().item() for p in robust_model.parameters() if p.grad is not None])
        robust_model.zero_grad()
        
        h_adv, _, _, _ = robust_model(q, r, perturbation=p_adv)
        y_adv = robust_model.get_output(h_adv, q)
        loss_adv = cal_loss(robust_model, [y_adv], r, rshft, sm)
        loss_adv.backward()
        robust_grad_norm_adv = sum([p.grad.norm().item() for p in robust_model.parameters() if p.grad is not None])
        robust_grad_norms.append(abs(robust_grad_norm_clean - robust_grad_norm_adv))

        # ----------------- 激活值分析 -----------------
        _, _, _, clean_gates = clean_model(q, r)
        _, _, _, robust_gates = robust_model(q, r)
        clean_forget_activations.extend(clean_gates['forget_gates'].cpu().numpy().flatten())
        robust_forget_activations.extend(robust_gates['forget_gates'].cpu().numpy().flatten())
        
        # 仅取一个batch进行演示
        break

    # 状态距离可视化
    visualize_and_save(
        {'原始模型': clean_state_dists, '对抗训练模型': robust_state_dists},
        "细胞状态距离随时间步变化 (干净 vs 扰动)",
        "时间步",
        "细胞状态欧氏距离",
        'cell_state_distance_over_time.png',
        save_dir
    )
    
    # 梯度范数变化可视化
    visualize_and_save(
        {'原始模型': clean_grad_norms, '对抗训练模型': robust_grad_norms},
        "梯度范数变化幅度比较 (扰动 vs 干净)",
        "Batch",
        "梯度范数变化幅度",
        'gradient_norm_change.png',
        save_dir
    )

    # 遗忘门激活值分布
    plt.figure(figsize=(12, 6))
    sns.histplot(clean_forget_activations, color='blue', label='原始模型', kde=True, stat="density", linewidth=0, bins=50)
    sns.histplot(robust_forget_activations, color='red', label='对抗训练模型', kde=True, stat="density", linewidth=0, bins=50)
    plt.title("遗忘门激活值分布比较")
    plt.xlabel("激活值")
    plt.ylabel("密度")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'forget_gate_activation_distribution.png'))
    plt.close()

    print("分析和可视化完成，所有图片已保存到指定目录。")
    print(f"分析目录: {save_dir}")


def main(params):
    if params['use_wandb'] == 1:
        import wandb
        with open("../configs/wandb.json") as fin:
            wandb_config = json.load(fin)
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir_origin, batch_size, fusion_type = params["save_dir_origin"], params["bz"], params["fusion_type"].split(",")
    save_dir_pert = params["save_dir_pert"]
    # 确保保存目录存在
    os.makedirs(save_dir_origin, exist_ok=True)
    if params["model_name"] == "":
        with open(os.path.join(save_dir_origin, "config.json")) as fin:
            config = json.load(fin)
            model_config = copy.deepcopy(config["model_config"])
            for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
                if remove_item in model_config:
                    del model_config[remove_item]
            trained_params = config["params"]
            fold = trained_params["fold"]
            model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
            if model_name in ["saint", "sakt", "atdkt"]:
                train_config = config["train_config"]
                seq_len = train_config["seq_len"]
                model_config["seq_len"] = seq_len
    elif params["model_name"] in ["llm"]:
        model_name, dataset_name, emb_type = "llm", params["dataset_name"], params.get("emb_type", "qid")
        model_config = {
        }
    elif params["model_name"] in ["mpllm"]:
        model_config = {
        }
    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]
    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir_origin: {save_dir_origin}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    origin_model = load_model("at_dkt", model_config, data_config, emb_type, save_dir_origin)
    pert_model = load_model("at_dkt", model_config, data_config, emb_type, save_dir_pert)
    
    save_test_path = os.path.join(save_dir_origin, f"{model.emb_type}_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))

    save_result_path = ""
    # if model.model_name == "rkt":
    #     testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path,save_result_path)
    # else:
    #     testauc, testacc = evaluate(model, test_loader, model_name, save_test_path,save_result_path)
    # print(f"testauc: {testauc}, testacc: {testacc}")

    testauc,testacc,window_testauc, window_testacc = -1, -1,-1,-1
    save_test_window_path = os.path.join(save_dir_origin, f"{model.emb_type}_test_window_predictions.txt")
    # if model.model_name == "rkt":
    #     window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel)
    # else:
    #     window_testauc, window_testacc = evaluate(model, test_window_loader, model_name)
    # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    dres = {}

    q_testaucs, q_testaccs = -1, -1
    qw_testaucs, qw_testaccs = -1, -1
    # if "test_question_file" in data_config and not test_question_loader is None:
    #     save_test_question_path = os.path.join(save_dir, f"{model.emb_type}_test_question_predictions.txt")
    #     q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
    #     for key in q_testaucs:
    #         dres["oriauc" + key] = q_testaucs[key]
    #     for key in q_testaccs:
    #         dres["oriacc" + key] = q_testaccs[key]

    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir_origin, f"{model.emb_type}_test_question_window_predictions.txt")
        # print(f"stu_id: {SET_TARGET_STU}")
        save_directory = 'lstm_robustness_analysis'
        analyze_lstm_robustness(origin_model, pert_model, test_question_window_loader, save_directory)








        # qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        
        # for key in qw_testaucs:
        #     dres["windowauc" + key] = qw_testaucs[key]
        # for key in qw_testaccs:
        #     dres["windowacc" + key] = qw_testaccs[key]

    # print(dres)
    # if model_name not in ["llm","mpllm"]:
    #     raw_config = json.load(open(os.path.join(save_dir_origin, "config.json")))
    #     dres.update(raw_config['params'])

    # if params['use_wandb'] == 1:
    #     wandb.log(dres)
    # print(f"windowauclate_mean: {dres['windowauclate_mean']}")
    # print(f"windowacclate_mean: {dres['windowacclate_mean']}")
    
    # # 将评估结果保存到 save_dir 目录下的 evaluation_results.json 文件

    # results_path = os.path.join(save_dir_origin, "evaluation_results.json")
    # try:
    #     with open(results_path, "w") as fout:
    #         json.dump(dres, fout, indent=4, ensure_ascii=False)
    #     print(f"评估结果已保存到 {results_path}")
    # except Exception as e:
    #     print(f"保存评估结果时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=4)
    parser.add_argument("--save_dir_origin", type=str, default="saved_model")
    parser.add_argument("--save_dir_pert", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
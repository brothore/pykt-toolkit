import os
import argparse
import json
import subprocess
import sys
import datetime
from pykt.models.cuda_retry import retry_decorator
import torch
# torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy
from wandb_multi_predict import main as predict_main
from pykt.models import train_model,evaluate,init_model
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train,init_dataset4train_local
from pykt.config import predict_after_train
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)
@retry_decorator
def run_prediction(predict_mode, save_dir):
    """
    运行预测脚本并实时显示输出
    Args:
        predict_mode: 1 for wandb_multi_predict.py, 2 for wandb_predict.py
        save_dir: 模型保存目录
    """
    try:
        if predict_mode == 1:
            script_path = "./wandb_multi_predict.py"
            if not os.path.exists(script_path):
                print(f"Warning: {script_path} not found")
                return False
            cmd = [sys.executable, script_path, "--save_dir", save_dir,"--bz","16" ]
            
        elif predict_mode == 2:
            script_path = "/root/autodl-tmp/pykt-toolkit/examples/wandb_predict.py"
            if not os.path.exists(script_path):
                print(f"Warning: {script_path} not found")
                return False
            cmd = [sys.executable, script_path, "--save_dir", save_dir]
            
        else:
            print(f"Invalid predict_after_train value: {predict_mode}")
            return False
        
        print(f"Starting prediction (mode {predict_mode}) at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*50} PREDICTION OUTPUT {'='*50}")

        # 创建子进程并实时捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并标准输出和错误输出
            text=True,
            bufsize=1,  # 行缓冲模式
            universal_newlines=True
        )
        
        # 实时输出处理
        try:
            # 逐行读取输出并打印
            for line in iter(process.stdout.readline, ''):
                # 添加前缀便于识别预测输出
                print(f"[Prediction] {line}", end='') 
                sys.stdout.flush()  # 确保立即输出
            
            # 等待进程完成
            process.wait()
            
        except KeyboardInterrupt:
            print("\nCtrl-C detected! Terminating prediction process...")
            process.terminate()
            return False
        
        # 检查返回码
        returncode = process.returncode
        print(f"\n{'='*50} PREDICTION FINISHED ({returncode}) {'='*50}")
        
        if returncode == 0:
            print("Prediction completed successfully!")
            return True
        else:
            print(f"Prediction failed with return code: {returncode}")
            return False
            
    except Exception as e:
        print(f"Error running prediction: {str(e)}")
        return False
def update_run_history(file_path, run_id, status):
    """
    更新 run_history.txt 中指定 run_id 的 Is_Completed 状态。
    """
    if not os.path.exists(file_path):
        print(f"错误: 历史文件不存在，无法更新状态: {file_path}")
        return
        
    # 读取所有行
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 找到并更新对应的行
    updated_lines = []
    # 跳过表头
    for i, line in enumerate(lines):
        if i == 0:
            updated_lines.append(line)
            continue
            
        parts = line.strip().split('\t')
        if len(parts) >= 4 and parts[0] == run_id:
            # 假设 Is_Completed 是第三列 (索引 2)
            parts[2] = status 
            updated_lines.append('\t'.join(parts) + '\n')
        else:
            updated_lines.append(line)

    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(updated_lines)

def main(params):
    # 记录整个流程的开始时间
    global_start_time = datetime.datetime.now()
    print(f"\n{'='*80}")
    print(f"🚀 **训练主流程开始** 🚀 | 当前时间: {global_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    run_history_path = ""
    run_id = ""
    
    try:
        # **初始化和配置检查**
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚙️ 阶段1: 初始化配置和路径...")
        
        save_dir = params.get('save_dir', 'saved_model')
        os.makedirs(save_dir, exist_ok=True)
        run_history_path = os.path.join(save_dir, 'run_history.txt')
        
        # 记录开始时间
        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # **生成唯一的本次运行ID** (用于后续更新状态，确保唯一性)
        import uuid
        run_id = str(uuid.uuid4())
        
        # 检查文件是否存在，如果不存在则创建表头
        if not os.path.exists(run_history_path):
            header = "Run_ID\tRun_Start_Time\tIs_Completed\tRun_Path\n"
            with open(run_history_path, 'w') as f:
                f.write(header)
                
        # 立即写入运行记录，'Is_Completed' 暂时设为 'Running'
        initial_log_line = f"{run_id}\t{start_time}\tRunning\t{save_dir}\n"
        with open(run_history_path, 'a') as f:
            f.write(initial_log_line)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 初始运行记录已写入: {run_history_path} (ID: {run_id})")

        if "use_trained" not in params:
            params['use_trained'] = 1
        use_trained = params['use_trained']
        if "use_wandb" not in params:
            params['use_wandb'] = 0
            params['use_wandb'] = 0 # 重复行，保留以匹配原代码

        if params['use_wandb']==1:
            import wandb
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📊 初始化 Weights & Biases...")
            wandb.init()

        set_seed(params["seed"])
        model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
            params["fold"], params["emb_type"], params["save_dir"]
        save_path_param = f"saved_params_{model_name}.json"
        
        # 确保保存目录存在
        save_dir = params.get('save_dir', 'saved_model')
        os.makedirs(save_dir, exist_ok=True) # 重复行，保留以匹配原代码
        
        # 确保保存目录存在
        # save_dir = params.get('save_dir', 'saved_model') # 重复行，注释掉
        # os.makedirs(save_dir, exist_ok=True) # 重复行，注释掉
        
        with open(os.path.join(save_dir, save_path_param), 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 参数已保存至: {os.path.join(save_dir, save_path_param)}")    
        
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚙️ 阶段2: 加载配置文件和数据配置...")
        debug_print(text = "load config files.",fuc_name="main")
        
        with open("../configs/kt_config.json") as f:
            config = json.load(f)
            train_config = config["train_config"]
            # ... (模型特定的 batch size 调整逻辑不变) ...
            if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "robustkt", "folibikt", "atkt", "lpkt", "skvmn", "dimkt",  "Transformer_template", "mamba_atakt", "mamba_atakt", "balance_akt", "qwen", "at_dkt", "TransformerKT", "multi_dataset_akt"]:
                train_config["batch_size"] = 64 ## because of OOM
            if model_name in ["simplekt","stablekt", "bakt_time", "sparsekt", "dbakt"]:
                train_config["batch_size"] = 64 ## because of OOM
            if model_name in ["gkt"]:
                train_config["batch_size"] = 16 
            if model_name in ["qdkt","qikt", "qikt_mamba"] and dataset_name in ['algebra2005','bridge2algebra2006', "qikt_mamba"]:
                train_config["batch_size"] = 32 
            if model_name in ["dtransformer"]:
                train_config["batch_size"] = 16 ## because of OOM
            if model_name in ["long_dkt"]:
                train_config["batch_size"] = 1 ## because of OOM
            
            model_config = copy.deepcopy(params)
            for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed","use_trained"]:
                del model_config[key]
            if 'batch_size' in params:
                train_config["batch_size"] = params['batch_size']
            if 'num_epochs' in params:
                train_config["num_epochs"] = params['num_epochs']
            # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]} # 注释掉原代码中的重复行
            
        batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

        with open("../configs/data_config.json") as fin:
            data_config = json.load(fin)
        if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
            train_config["seq_len"] = data_config[dataset_name]['maxlen']
        seq_len = train_config["seq_len"]
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📚 训练配置: epochs={num_epochs}, batch_size={batch_size}, optimizer={optimizer}, seq_len={seq_len}")
        
        # **数据初始化**
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 📦 阶段3: 初始化数据集...")
        debug_print(text="init_dataset",fuc_name="main")
        if model_name not in ["dimkt"]:
            train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)
        else:
            diff_level = params["difficult_levels"]
            train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 数据集加载完成。")
        
        params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])

        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📑 **运行参数概览**: Model={model_name}, Dataset={dataset_name}, Fold={fold}, EmbType={emb_type}")
        # print(f"params: {params}, params_str: {params_str}") # 原代码中的重复行，注释掉
        if params['add_uuid'] == 1 and params["use_wandb"] == 1:
            # import uuid # 重复行，注释掉
            # if not model_name in ['saint','saint++']:
            params_str = params_str+f"_{ str(uuid.uuid4())}"
        ckpt_path = os.path.join(save_dir, params_str)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 💾 模型检查点路径设置: {ckpt_path}")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] **开始训练模型**: {model_name}, EmbType: {emb_type}")
        print(f"模型配置 (model_config): {model_config}")
        print(f"训练配置 (train_config): {train_config}")

        if model_name in ["dimkt"]:
            # del model_config['num_epochs'] # 注释掉原代码中的重复行
            del model_config['weight_decay']

        save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 完整配置已保存至 {os.path.join(ckpt_path, 'config.json')}")
        
        learning_rate = params["learning_rate"]
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","stablekt", "bakt_time","folibikt", "dbakt"]:
            model_config["seq_len"] = seq_len
            
        # **模型和优化器初始化**
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🧠 阶段4: 初始化模型和优化器...")
        debug_print(text = "init_model",fuc_name="main")
        model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        model.cuda()  # 确保模型在GPU上
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 模型初始化完成，模型已移至 {device}。")
        
        # 优化器配置
        if model_name == "hawkes":
            # ... (hawkes 优化器配置不变) ...
            weight_p, bias_p = [], []
            for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
                if 'bias' in name:
                    bias_p.append(p)
                else:
                    weight_p.append(p)
            optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
            opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
        elif model_name == "iekt":
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        elif model_name == "dtransformer":
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 优化器: Adam, lr={learning_rate}, weight_decay=1e-5 (dtransformer special)")
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif model_name == "dimkt":
            opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
        else:
            if optimizer == "sgd":
                opt = SGD(model.parameters(), learning_rate, momentum=0.9)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 优化器: SGD, lr={learning_rate}, momentum=0.9")
            elif optimizer == "adam":
                opt = Adam(model.parameters(), learning_rate)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 优化器: Adam, lr={learning_rate}")
    
        testauc, testacc = -1, -1
        window_testauc, window_testacc = -1, -1
        validauc, validacc = -1, -1
        best_epoch = -1
        save_model = True
        
        # **模型训练**
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚂 阶段5: **开始模型训练** (共 {num_epochs} 轮)...")
        debug_print(text = "train model",fuc_name="main")
        
        # model.cuda()  # 重复行，注释掉

        if model_name == "rkt":
            testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(
                model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, 
                data_config[dataset_name], fold, use_trained=use_trained
            )
        elif model_name == "long_dkt":
            testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(
                model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model,
                data_config[dataset_name], use_trained=use_trained
            )
        else:
            testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(
                model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, 
                use_trained=use_trained
            )
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 模型训练完成！最佳轮次: {best_epoch}")
        
        if save_model:
            best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
            model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
            net = torch.load(model_save_path)
            best_model.load_state_dict(net)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 💾 最佳模型已加载用于评估或保存。路径: {model_save_path}")

        print("\n" + "="*80)
        print("📈 最终训练结果汇总:")
        print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
        print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
        print("="*80 + "\n")
        
        
        if params['use_wandb']==1:
            wandb.log({ 
                        "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📊 结果已同步至 Weights & Biases。")
        
        # **训练后预测**
        if predict_after_train in [1, 2]:
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🔮 阶段6: 开始训练后预测 (predict_after_train={predict_after_train})...")
            print(f"{'='*50}")

            if predict_after_train == 1:
                predict_params = {
                    "bz": 16,  # 来自原cmd的--bz 16
                    "save_dir": ckpt_path,
                    "fusion_type": "late_fusion",  # 原脚本默认
                    "use_wandb": params['use_wandb'],
                    "start_student": 1,  # 原脚本默认
                    "save_reult": 0,  # 原脚本默认
                    "use_saved_result": 1,  # 原脚本默认
                    "mode": "all",  # 原脚本默认
                    "target_file_type": "test_window_sequences"  # 原脚本默认
                }
                
                try:
                    predict_start_time = datetime.datetime.now()
                    print(f"[{predict_start_time.strftime('%H:%M:%S')}] 调用内部预测脚本: wandb_multi_predict.main()")
                    predict_main(predict_params)  # 直接调用wandb_multi_predict的main
                    prediction_success = True
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 预测成功完成！")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "success"})
                except Exception as e:
                    prediction_success = False
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ 预测失败: {str(e)}")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "failed"})
            elif predict_after_train == 2:
                 # 调用 run_prediction 函数
                run_prediction_start_time = datetime.datetime.now()
                print(f"[{run_prediction_start_time.strftime('%H:%M:%S')}] 调用外部预测脚本 (mode 2) via subprocess...")
                prediction_success = run_prediction(predict_after_train, ckpt_path)
                if prediction_success:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 外部预测脚本执行成功。")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "success"})
                else:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ 外部预测脚本执行失败。")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "failed"})
            elif predict_after_train == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] predict_after_train is set to 0, skipping prediction.")
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Invalid predict_after_train value: {predict_after_train}. Valid values are 0, 1, or 2.")
            
        
        # **更新运行历史和结束**
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🏁 阶段7: 流程结束和清理...")
        
        # **更新运行历史为成功**
        update_run_history(run_history_path, run_id, 'True')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ 运行历史状态已更新为成功 (ID: {run_id})")
        
        global_end_time = datetime.datetime.now()
        total_time = global_end_time - global_start_time
        print(f"\n{'='*80}")
        print(f"🎉 **整个训练和预测流程完成** 🎉")
        print(f"开始时间: {global_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结束时间: {global_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_time}")
        print(f"{'='*80}")
        
    except Exception as e:
        # 记录错误信息
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ 训练过程中发生错误: {e}")
        if 'run_id' in locals() and run_id != "":
            # 确保在异常发生时，如果 run_id 已经被定义，就去更新状态
            update_run_history(run_history_path, run_id, 'False')
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ 运行历史状态已更新为失败 (ID: {run_id})")
        
        # 重新抛出异常，让装饰器捕获并决定是否重试
        raise e
    finally:
        # 无论 try 块是成功执行、还是通过 except 块抛出异常（准备重试或退出），finally 都会执行。
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 💡 正在执行最终资源清理...")
        
        # 1. 确保释放 PyTorch 内部缓存 (对 OOM 尤为重要)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

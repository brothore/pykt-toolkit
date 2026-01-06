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
from pykt.datasets import init_dataset4train
from pykt.config import predict_after_train
import datetime
# from auc_processor import save_auc_results_from_file
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
import runpy
def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

# def run_prediction(predict_mode, save_dir):
#     """
#     运行预测脚本并实时显示输出
#     Args:
#         predict_mode: 1 for wandb_multi_predict.py, 2 for wandb_predict.py
#         save_dir: 模型保存目录
#     """
#     try:
#         if predict_mode == 1:
#             script_path = "./wandb_multi_predict.py"
#             if not os.path.exists(script_path):
#                 print(f"Warning: {script_path} not found")
#                 return False
#             cmd = [sys.executable, script_path, "--save_dir", save_dir,"--bz","16" ]
#             print(cmd)
            
#         elif predict_mode == 2:
#             script_path = "/root/autodl-tmp/pykt-toolkit/examples/wandb_predict.py"
#             if not os.path.exists(script_path):
#                 print(f"Warning: {script_path} not found")
#                 return False
#             cmd = [sys.executable, script_path, "--save_dir", save_dir]
            
#         else:
#             print(f"Invalid predict_after_train value: {predict_mode}")
#             return False
        
#         print(f"Starting prediction (mode {predict_mode}) at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         print(f"Command: {' '.join(cmd)}")
#         print(f"{'='*50} PREDICTION OUTPUT {'='*50}")

#         # 创建子进程并实时捕获输出
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,  # 合并标准输出和错误输出
#             text=True,
#             bufsize=1,  # 行缓冲模式
#             universal_newlines=True
#         )
        
#         # 实时输出处理
#         try:
#             # 逐行读取输出并打印
#             for line in iter(process.stdout.readline, ''):
#                 # 添加前缀便于识别预测输出
#                 print(f"[Prediction] {line}", end='') 
#                 sys.stdout.flush()  # 确保立即输出
            
#             # 等待进程完成
#             process.wait()
            
#         except KeyboardInterrupt:
#             print("\nCtrl-C detected! Terminating prediction process...")
#             process.terminate()
#             return False
        
#         # 检查返回码
#         returncode = process.returncode
#         print(f"\n{'='*50} PREDICTION FINISHED ({returncode}) {'='*50}")
        
#         if returncode == 0:
#             print("Prediction completed successfully!")
#             return True
#         else:
#             print(f"Prediction failed with return code: {returncode}")
#             return False
            
#     except Exception as e:
#         print(f"Error running prediction: {str(e)}")
#         return False
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
    try:
        
        aug_probs = {
            'truncate': params.pop("random_trunc", 0),
            'duplicate': params.pop("random_dup", 0),
            'shuffle': params.pop("random_shuf", 0),   # 随机乱序概率
            
            # 补齐参数：如果没有在 params 定义，默认复制序列长度的 15% (这里示例给的是0)
            'copy_ratio': params.pop("copy_ratio", 0), 
            
            # 补齐参数：如果没有在 params 定义，离散截断时默认丢弃 10% (这里示例给的是0)
            'drop_ratio': params.pop("drop_ratio", 0),
            'random_rev': params.pop('random_rev', 0)  
        }
        # **开始记录运行历史**
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
                
        # 立即写入运行记录，'Is_Completed' 暂时设为 'Running' 或占位符
        initial_log_line = f"{run_id}\t{start_time}\tRunning\t{save_dir}\n"
        with open(run_history_path, 'a') as f:
            f.write(initial_log_line)
        print(f"✅ 初始运行记录已写入: {run_history_path} (ID: {run_id})")

        start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if "use_trained" not in params:
            params['use_trained'] = 0
        use_trained = params['use_trained']
        # print(f"\n\n\n\n\nuse_trained!!!!!!!!!!!\n\n\n\n\n: {use_trained}")
        # print(f"\n\n\n\n\nuse_trained!!!!!!!!!!!\n\n\n\n\n: {use_trained}")
        if "use_wandb" not in params:
            params['use_wandb'] = 0

        mode = params.get('mode', "train")

        if params['use_wandb']==1:
            import wandb
            wandb.init()

        set_seed(params["seed"])
        model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
            params["fold"], params["emb_type"], params["save_dir"]
        save_path_param = f"saved_params_{model_name}.json"
        # 确保保存目录存在
        save_dir = params.get('save_dir', 'saved_model')
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, save_path_param), 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"✅ 参数已保存至: {os.path.join(save_dir, save_path_param)}")    
        save_path_param = f"saved_params_{model_name}.json"
        # 确保保存目录存在
        save_dir = params.get('save_dir', 'saved_model')

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, save_path_param), 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"✅ 参数已保存至: {os.path.join(save_dir, save_path_param)}")    
        debug_print(text = "load config files.",fuc_name="main")
        
        with open("../configs/kt_config.json") as f:
            config = json.load(f)
            train_config = config["train_config"]
            
            model_config = copy.deepcopy(params)
            for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed","use_trained"]:
                del model_config[key]
            if 'batch_size' in params:
                train_config["batch_size"] = params['batch_size']
            if 'num_epochs' in params:
                train_config["num_epochs"] = params['num_epochs']
            # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
        batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

        with open("../configs/data_config.json") as fin:
            data_config = json.load(fin)
        if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
            train_config["seq_len"] = data_config[dataset_name]['maxlen']
        seq_len = train_config["seq_len"]

        # print("Start init data")
        print(dataset_name, model_name, data_config, fold, batch_size)
        
        debug_print(text="init_dataset",fuc_name="main")
        if model_name not in ["dimkt"]:
            train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size,aug_probs=aug_probs)
        else:
            diff_level = params["difficult_levels"]
            train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size,aug_probs=aug_probs)

        params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])

        print(f"params: {params}, params_str: {params_str}")
        if params['add_uuid'] == 1 and params["use_wandb"] == 1:
            import uuid
            # if not model_name in ['saint','saint++']:
            params_str = params_str+f"_{ str(uuid.uuid4())}"
        ckpt_path = os.path.join(save_dir, params_str)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
        print(f"model_config: {model_config}")
        print(f"train_config: {train_config}")

        if model_name in ["dimkt"]:
            # del model_config['num_epochs']
            del model_config['weight_decay']

        save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
        learning_rate = params["learning_rate"]
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt","stablekt", "bakt_time","folibikt", "dbakt"]:
            model_config["seq_len"] = seq_len
            
        debug_print(text = "init_model",fuc_name="main")
        # print(f"model_name:{model_name}")
        model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        # print(f"model is {model}")
        if model_name == "hawkes":
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
            print(f"dtransformer weight_decay = 1e-5")
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif model_name == "dimkt":
            opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
        else:
            if optimizer == "sgd":
                opt = SGD(model.parameters(), learning_rate, momentum=0.9)
            elif optimizer == "adam":
                opt = Adam(model.parameters(), learning_rate)
    
        testauc, testacc = -1, -1
        window_testauc, window_testacc = -1, -1
        validauc, validacc = -1, -1
        best_epoch = -1
        save_model = True
        
        debug_print(text = "train model",fuc_name="main")
        
        model.cuda()  # 确保模型在GPU上

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


        if save_model:
            best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
            net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
            best_model.load_state_dict(net)

        print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
        print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
        model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
        print(f"end:{datetime.datetime.now()}")
        
        if params['use_wandb']==1:
            wandb.log({ 
                        "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})
        
        # 训练完成后进行预测
        if predict_after_train in [1, 2]:
            print(f"\n{'='*50}")
            print(f"Training completed. Starting prediction with predict_after_train={predict_after_train}")
            print(f"{'='*50}")

           
            if predict_after_train == 1:
                predict_params = {
                    "bz": batch_size,  # 来自原cmd的--bz 16
                    "save_dir": ckpt_path,
                    "fusion_type": "late_fusion",  # 原脚本默认
                    "use_wandb": params['use_wandb'],
                    "start_student": 1,  # 原脚本默认
                    "save_reult": 0,  # 原脚本默认
                    "seq_len":0,
                    "use_saved_result": 1,  # 原脚本默认
                    "mode": "all",  # 原脚本默认
                    "target_file_type": "test_question_window_sequences"  # 原脚本默认
                }
                try:
                    predict_main(predict_params)  # 直接调用wandb_multi_predict的main
                    prediction_success = True
                    print("Prediction completed successfully!")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "success"})
                except Exception as e:
                    prediction_success = False
                    print(f"Prediction failed: {str(e)}")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "failed"})
            elif predict_after_train == 2:
                predict_params = {
                    "bz": batch_size,  # 来自原cmd的--bz 16
                    "save_dir": ckpt_path,
                    "use_wandb": params['use_wandb'],
                }
                
                # 定义要运行的脚本的路径
                script_path = "/data/pykt-toolkit/examples/wandb_predict_with_unbalance.py"
                
                # 保存当前的 sys.argv，以便后续恢复
                original_argv = sys.argv
                
                try:
                    # 1. 构建新的 sys.argv 列表
                    # sys.argv[0] 必须是脚本的路径
                    new_argv = [script_path]
                    for key, value in predict_params.items():
                        new_argv.append(f"--{key}")
                        new_argv.append(str(value))
                    
                    # 2. 临时替换 sys.argv
                    sys.argv = new_argv
                    
                    print(f"--- 正在调用外部脚本 (runpy): {script_path}")
                    print(f"--- 模拟参数: {' '.join(new_argv[1:])}")
                    print(f"--- 完整运行指令: python {script_path} {' '.join(new_argv[1:])}")


                    # 3. 使用 runpy 运行脚本
                    # run_name="__main__" 会让脚本认为它是主程序
                    runpy.run_path(script_path, run_name="__main__")
                    
                    # --- 运行成功后的原始逻辑 ---
                    prediction_success = True
                    print("Prediction completed successfully!")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "success"})

                    # # --- (你之前添加的 AUC 解析逻辑应放在这里) ---
                    # print("="*30)
                    # print("开始解析和保存详细的AUC结果...")
                    # target_suffix = "window_predictions.txt"
                    # target_file_path = None
                    
                    # try:
                    #     for filename in os.listdir(ckpt_path):
                    #         if filename.endswith(target_suffix):
                    #             target_file_path = os.path.join(ckpt_path, filename)
                    #             print(f"✅ 已找到目标文件: {target_file_path}")
                    #             break
                    # except Exception as e:
                    #     print(f"❌ 在搜索文件时发生意外错误: {e}")
                
                    # if target_file_path and os.path.exists(target_file_path):
                    #     save_auc_results_from_file(target_file_path)
                    # else:
                    #     print(f"❌ 错误：在目录 {ckpt_path} 中未找到结尾为 '{target_suffix}' 的文件")
                    # print("="*30)
                    # # --- AUC 解析逻辑结束 ---

                except Exception as e:
                    # 这个 except 块现在可以捕获来自 runpy 脚本内部的任何 Python 错误
                    prediction_success = False
                    print(f"Prediction failed: {str(e)}")
                    if params['use_wandb']==1:
                        wandb.log({"prediction_status": "failed"})
                
                finally:
                    # 4. 无论成功还是失败，都必须恢复原始的 sys.argv
                    sys.argv = original_argv
                    print("--- 外部脚本执行完毕 ---")
        elif predict_after_train == 0:
            print("predict_after_train is set to 0, skipping prediction.")
        else:
            print(f"Invalid predict_after_train value: {predict_after_train}. Valid values are 0, 1, or 2.")
            
        print(f"Training and prediction process completed at: {datetime.datetime.now()}")
        
        # **添加：更新运行历史为成功**
        update_run_history(run_history_path, run_id, 'True')
        print(f"✅ 运行历史状态已更新为成功 (ID: {run_id})")
    except Exception as e:
        # 记录错误信息
        print(f"训练过程中发生错误: {e}")
        if 'run_id' in locals():
            # 确保在异常发生时，如果 run_id 已经被定义，就去更新状态
            update_run_history(run_history_path, run_id, 'False')
            print(f"⚠️ 运行历史状态已更新为失败 (ID: {run_id})")
        
        # 重新抛出异常，让装饰器捕获并决定是否重试
        raise e
    finally:
        # 无论 try 块是成功执行、还是通过 except 块抛出异常（准备重试或退出），finally 都会执行。
        print("💡 正在执行最终资源清理...")
        
        # 1. 确保释放 PyTorch 内部缓存 (对 OOM 尤为重要)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

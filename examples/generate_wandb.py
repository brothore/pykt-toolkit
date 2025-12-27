import os
import json
import argparse
import yaml  # <--- 引入 yaml 库进行智能解析

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(params):
    src_dir = params["src_dir"]
    project_name = params["project_name"]
    dataset_names = params["dataset_names"]
    model_names = params["model_names"]
    folds = params["folds"]
    save_dir_suffix = params["save_dir_suffix"]
    all_dir = params["all_dir"]
    launch_file = params["launch_file"]
    generate_all = params["generate_all"]
    emb_types = params["emb_types"]
    # 强制将 batch_size 转为 int，防止 wandb 报类型错误
    target_batch_size = int(params["batch_size"])

    emb_types_list = [x for x in emb_types.split(",")]
    
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)

    # 获取 API Key
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        try:
            with open("../configs/wandb.json") as fin:
                wandb_config = json.load(fin)
                wandb_api_key = wandb_config.get("api_key")
        except:
            wandb_api_key = ""

    print(f"Using API KEY: {wandb_api_key[:6]}******")
    
    with open(launch_file, "w") as fallsh:
        pre = f"WANDB_API_KEY={wandb_api_key} wandb sweep "
        
        for dataset_name in dataset_names.split(","):
            for m in model_names.split(","):
                for _type in emb_types_list:
                    for fold in folds.split(","):
                        
                        fpath = os.path.join(src_dir, f"{m}.yaml")
                        if not os.path.exists(fpath):
                            print(f"⚠️  Skipping: {fpath} not found")
                            continue

                        # type_str = _type.replace("linear", "")
                        # fname = f"{dataset_name}_{m}_{type_str}_{fold}.yaml"
                        fname = f"{dataset_name}_{m}_{fold}.yaml"
                        ftarget = os.path.join(all_dir, fname)
                        
                        print(f"🔄 Processing: {fname} | Batch Size -> {target_batch_size}")

                        # ======================================================
                        # 阶段 1: 传统的文本替换 (保留对路径、字符串的兼容性)
                        # ======================================================
                        with open(fpath, "r") as fin:
                            raw_data = fin.read()
                        
                        # 替换数据集
                        raw_data = raw_data.replace("xes", dataset_name)
                        # 替换保存路径
                        raw_data = raw_data.replace("tiaocan", f"tiaocan_{dataset_name}{save_dir_suffix}")
                        
                        # 替换 Embedding (原有逻辑)
                        # if '["qid"]' in raw_data:
                        #     raw_data = raw_data.replace('["qid"]', f"['{_type}']")
                        
                        # 替换 Fold
                        raw_data = raw_data.replace("[0, 1, 2, 3, 4]", str([int(fold)]))

                        # ======================================================
                        # 阶段 2: 智能结构解析 (处理 Batch Size)
                        # ======================================================
                        try:
                            # 将替换过文本的字符串加载为 YAML 对象
                            config = yaml.safe_load(raw_data)
                            
                            # 1. 确保 parameters 节点存在
                            if "parameters" not in config:
                                config["parameters"] = {}
                            
                            # 2. 智能覆盖 batch_size
                            # 无论原来有 "batch_size: values: [32]" 还是 "batch_size: values: [BATCH_SIZE]"
                            # 甚至原来根本没有 batch_size，这里都会强制写入新的值
                            config["parameters"]["batch_size"] = {"values": [target_batch_size]}

                            # ======================================================
                            # 阶段 3: 写回文件
                            # ======================================================
                            with open(ftarget, "w") as fout:
                                # 写入 name 字段 (WandB 推荐)
                                # 只有当 yaml 里没定义 name 时才写入，避免覆盖原有的 logic (可选)
                                # 这里我们按照你原脚本逻辑，强制写在第一行
                                fout.write(f"name: {fname.split('.')[0]}\n")
                                
                                # 将修改后的对象转回 yaml 字符串
                                # sort_keys=False 保持原有顺序，default_flow_style=None 让列表显示更自然
                                yaml.dump(config, fout, sort_keys=False, default_flow_style=None)

                        except yaml.YAMLError as exc:
                            print(f"❌ YAML 解析失败: {exc}")
                            continue

                        if not generate_all:
                            fallsh.write(f"{pre}{ftarget} -p {project_name}\n")
        
        if generate_all:
            files = sorted(os.listdir(all_dir))
            for f in files:
                if f.endswith(".yaml"):
                    fpath = os.path.join(all_dir, f)
                    fallsh.write(f"{pre}{fpath} -p {project_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./seedwandb/")
    parser.add_argument("--project_name", type=str, default="kt_toolkits")
    parser.add_argument("--dataset_names", type=str, default="assist2015")
    parser.add_argument("--model_names", type=str, default="dkt")
    parser.add_argument("--emb_types", type=str, default="qid")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir_suffix", type=str, default="")
    parser.add_argument("--all_dir", type=str, default="all_wandbs")
    parser.add_argument("--launch_file", type=str, default="all_start.sh")
    parser.add_argument("--generate_all", type=str2bool, default="False")

    args = parser.parse_args()
    
    # 自动命名逻辑兜底
    if args.launch_file == "all_start.sh" and args.project_name != "kt_toolkits":
         pass 

    params = vars(args)
    print("Parameters:", params)
    main(params)
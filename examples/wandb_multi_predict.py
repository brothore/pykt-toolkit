import os
import argparse
import json
import copy
import torch
import pandas as pd
# from pykt.config import ERR_PATH, stu_pk
from pykt.models import evaluate, evaluate_question, load_model
from pykt.datasets import init_test_datasets,init_test_datasets_multi_stu
import pykt.config as config_module
import traceback  # 在文件顶部添加导入
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
def parse_dataset_name(save_dir):
    """从save_dir参数中解析数据集名称"""
    # 获取最后一个目录（模型目录）
    model_dir = os.path.basename(os.path.normpath(save_dir))
    # 提取数据集名称的逻辑
    if "nips_task34" in model_dir:
        # 对于nips_task34: 取第二个横线前的部分
        parts = model_dir.split('_')
        if len(parts) >= 4:  # 确保有足够的横线
            # 前两部分组合为数据集名称
            return "_".join(parts[0:2])
        return "nips_task34"  # 回退值
    else:
        # 其他数据集: 取第一个横线前的部分
        return model_dir.split('_')[0]
    
def extract_student_stats(stats_file_path):
    """从学生统计文件中提取统计数据"""
    if not os.path.exists(stats_file_path):
        print(f"警告：统计文件不存在: {stats_file_path}")
        return {
            'record_count': -1,
            'total_questions': -1,
            'avg_questions_per_concept': -1.0,
            'max_questions_per_concept': -1,
            'min_questions_per_concept': -1,
            'questions_range': -1,
            'overall_accuracy': -1.0,
            'accuracy_range': -1.0,
            'accuracy_variance': -1.0,
            'max_accuracy': -1.0,
            'min_accuracy': -1.0
        }
    
    try:
        # 读取统计文件
        stats_df = pd.read_csv(stats_file_path)
        
        # 如果空文件
        if stats_df.empty:
            return {
                'record_count': 0,
                'total_questions': 0,
                'avg_questions_per_concept': 0.0,
                'max_questions_per_concept': 0,
                'min_questions_per_concept': 0,
                'questions_range': 0,
                'overall_accuracy': 0.0,
                'accuracy_range': 0.0,
                'accuracy_variance': 0.0,
                'max_accuracy': 0.0,
                'min_accuracy': 0.0
            }
        
        # 提取第一条记录的统计信息（所有行数据相同）
        row = stats_df.iloc[0]
        
        return {
            'record_count': row.get('record_count', stats_df.shape[0]),  # 使用文件行数作为record_count
            'total_questions': row.get('total_questions', -1),
            'avg_questions_per_concept': row.get('avg_questions_per_concept', -1.0),
            'max_questions_per_concept': row.get('max_questions_per_concept', -1),
            'min_questions_per_concept': row.get('min_questions_per_concept', -1),
            'questions_range': row.get('questions_range', -1),
            'overall_accuracy': row.get('overall_accuracy', -1.0),
            'accuracy_range': row.get('accuracy_range', -1.0),
            'accuracy_variance': row.get('accuracy_variance', -1.0),
            'max_accuracy': row.get('max_accuracy', -1.0),
            'min_accuracy': row.get('min_accuracy', -1.0)
        }
    except Exception as e:
        print(f"读取统计文件时出错: {e}")
        return {}
def predict_interval_data(model, data_config, model_name, fusion_type, save_dir):
    """预测所有预定义的区间数据文件"""
    # 定义所有需要预测的文件键名列表
    interval_file_keys = [
        "total_questions_low_interval_file",
        "total_questions_medium_interval_file",
        "total_questions_high_interval_file",
        "overall_accuracy_low_interval_file",
        "overall_accuracy_medium_interval_file",
        "overall_accuracy_high_interval_file",
        "accuracy_range_low_interval_file",
        "accuracy_range_medium_interval_file",
        "accuracy_range_high_interval_file",
        "accuracy_variance_low_interval_file",
        "accuracy_variance_medium_interval_file",
        "accuracy_variance_high_interval_file",
        "questions_range_low_interval_file",
        "questions_range_medium_interval_file",
        "questions_range_high_interval_file"
    ]
    
    interval_results = {}
    processed_files = []  # 用于记录已处理的文件信息
    
    for config_key in interval_file_keys:
        # 检查配置是否存在
        if config_key not in data_config:
            print(f"警告: 未找到配置键: {config_key}")
            continue
            
        file_path = data_config[config_key]
        if not file_path:
            print(f"警告: {config_key} 的值为空")
            continue
            
        # 从键名中解析出文件类型和区间类型
        parts = config_key.split('_')
        if len(parts) < 4:
            print(f"警告: 无法解析配置键名: {config_key}")
            continue
            
        file_type = '_'.join(parts[:-3])  # 如 "total_questions"
        interval_type = parts[-3]         # 如 "low", "medium", "high"
        
        print(f"\n开始预测 {interval_type} 区间的 {file_type} 数据: {config_key}")
        
        try:
            # 初始化数据集 - 直接使用配置键名作为predict_file_type
            _, _, _, test_question_window_loader = init_test_datasets_multi_stu(
                data_config, model_name, batch_size=256, 
                predict_file_type=config_key
            )
            
            if test_question_window_loader is None:
                print(f"警告: 无法为 {config_key} 创建数据加载器")
                continue
                
            # 预测并保存结果
            save_path = os.path.join(save_dir, f"{model.emb_type}_{file_type}_{interval_type}_predictions.txt")
            testaucs, testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_path)
            
            # 保存结果
            for key in testaucs:
                result_key = f"{file_type}_{interval_type}_auc{key}"
                interval_results[result_key] = testaucs[key]
            for key in testaccs:
                result_key = f"{file_type}_{interval_type}_acc{key}"
                interval_results[result_key] = testaccs[key]
                
            # 记录处理信息
            processed_files.append({
                '文件类型': file_type,
                '区间类型': interval_type,
                'AUC': testaucs.get('late_mean', 'N/A'),
                'ACC': testaccs.get('late_mean', 'N/A'),
                '保存路径': save_path
            })
            
            print(f"完成 {interval_type} 区间的 {file_type} 数据预测")
            print(f"预测结果已保存到: {save_path}")
            
        except Exception as e:
            print(f"预测 {config_key} 时出错: {str(e)}")
            traceback.print_exc() 
            print("\n")  # 添加空行分隔不同错误
            continue
    
    # 打印汇总结果表格
    print("\n" + "="*80)
    print("区间数据预测结果汇总:")
    print("="*80)
    
    # 打印表格头
    print(f"{'文件类型':<25} | {'区间类型':<10} | {'AUC':<10} | {'ACC':<10} | {'保存路径'}")
    print("-"*80)
    
    # 打印每个文件的结果
    for info in processed_files:
        print(f"{info['文件类型']:<25} | {info['区间类型']:<10} | {info['AUC']:<10.4f} | {info['ACC']:<10.4f} | {info['保存路径']}")
    
    # 打印所有结果的键值对
    print("\n详细结果键值对:")
    for key, value in interval_results.items():
        print(f"{key}: {value}")
    
    print("="*80 + "\n")
            
    return interval_results

    
def evaluate_single_student(params, student_id):
    """评估单个学生的函数"""
    print(f"\n开始评估学生 {student_id}")
    
    
    if params['use_wandb'] == 1:
        import wandb
        with open("../configs/wandb.json") as fin:
            wandb_config = json.load(fin)
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json")) as fin:
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

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time","dbakt"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]
    
    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets_multi_stu(data_config, model_name, batch_size,predict_file_type=f"top_{student_id}_student.csv",load_flags=[0,0,0,1])
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"学生 {student_id}: 开始预测模型 {model_name}, embtype: {emb_type}, dataset_name: {dataset_name}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = os.path.join(save_dir, f"{model.emb_type}_test_predictions_student_{student_id}.txt")
    # 在评估前提取学生统计信息
    stats_file_path = os.path.join(data_config["dpath"], f"top_{student_id}_student.csv")
    student_stats = extract_student_stats(stats_file_path)
    print(f"学生 {student_id}: 已加载统计信息")
    
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
    
   
    dres = {
        "student_id": student_id,
    }

    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, f"summary_{model.emb_type}_test_question_window_predictions_student.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc" + key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc" + key] = qw_testaccs[key]

    raw_config = json.load(open(os.path.join(save_dir, "config.json")))
    dres.update(raw_config['params'])
    
    # 添加学生统计信息到评估结果
    dres.update(student_stats)
    print(f"学生 {student_id}: 已添加统计信息到评估结果")

    if params['use_wandb'] == 1:
        wandb.log(dres)

    # 输出关键指标
    if 'windowauclate_mean' in dres:
        print(f"学生 {student_id}: windowauclate_mean: {dres['windowauclate_mean']}")
    if 'windowacclate_mean' in dres:
        print(f"学生 {student_id}: windowacclate_mean: {dres['windowacclate_mean']}")
    
    # 将评估结果保存到单独的JSON文件
    results_path = os.path.join(save_dir, f"evaluation_results_{student_id}.json")
    # try:
    with open(results_path, "w") as fout:
        json.dump(dres, fout, indent=4, ensure_ascii=False)
    print(f"学生 {student_id}: 评估结果已保存到 {results_path}")
    # except Exception as e:
    #     print(f"学生 {student_id}: 保存评估结果时出错: {e}")

    return dres

def main(params):
    dataset_name = parse_dataset_name(params["save_dir"])
    print(f"解析出的数据集名称: {dataset_name}")
    
    """主函数，循环评估所有学生"""
    config_path = os.path.join(os.path.dirname(__file__), '../configs/data_config.json')
    try:
        with open(config_path) as fin:
            data_configs = json.load(fin)
            dataset_config = data_configs.get(dataset_name)
            
            if dataset_config:
                # 获取学生总数
                total_students = dataset_config["students_num_eval"]
                print(f"从配置加载总学生数: {total_students}")
                params['total_students'] = total_students
                # 数据集配置存入params
                params['full_data_config'] = dataset_config  
            else:
                print(f"警告: 配置中未找到数据集 '{dataset_name}'，使用默认学生数")
    except Exception as e:
        print(f"加载数据配置时出错: {e}")

    start_student = params.get('start_student', 1)
    save_dir = params["save_dir"]
    
    # 创建统计文件路径
    stat_file_path = os.path.join(save_dir, "evaluation_statistics.txt")
    
    # 存储所有学生的评估结果
    all_results = []
    
    print(f"开始批量评估，共 {total_students} 个学生，从学生 {start_student} 开始")
    
    for student_id in range(start_student, total_students + 1):
        # 评估单个学生
        result = evaluate_single_student(params, student_id)
        all_results.append(result)
        print(f"完成学生 {student_id} 的评估 ({student_id - start_student + 1}/{total_students - start_student + 1})")
    
    # 将所有结果转换为DataFrame并保存为CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 生成CSV文件路径
        csv_path = os.path.join(save_dir, "batch_evaluation_results.csv")
        
        try:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n所有评估结果已保存到 CSV 文件: {csv_path}")
            
            # 打开统计文件进行写入
            with open(stat_file_path, 'w', encoding='utf-8') as stat_file:
                # 写入CSV文件信息
                stat_file.write(f"所有评估结果已保存到 CSV 文件: {csv_path}\n")
                
                # 写入评估统计摘要
                stat_file.write("\n评估完成统计:\n")
                stat_file.write(f"总计评估学生数: {len(all_results)}\n")
                
                # 计算并写入各统计列的基本统计信息
                stat_columns = [
                    'total_questions', 'avg_questions_per_concept', 'max_questions_per_concept',
                    'min_questions_per_concept', 'questions_range', 'overall_accuracy',
                    'accuracy_range', 'accuracy_variance', 'max_accuracy', 'min_accuracy'
                ]
                
                for col in stat_columns:
                    if col in df.columns:
                        valid_data = df[col][df[col] >= 0]  # 只取有效值
                        if not valid_data.empty:
                            # 控制台输出
                            print(f"\n{col}统计:")
                            print(f"  最小值: {valid_data.min():.2f}")
                            print(f"  最大值: {valid_data.max():.2f}")
                            print(f"  平均值: {valid_data.mean():.2f}")
                            print(f"  标准差: {valid_data.std():.2f}")
                            
                            # 文件输出
                            stat_file.write(f"\n{col}统计:\n")
                            stat_file.write(f"  最小值: {valid_data.min():.2f}\n")
                            stat_file.write(f"  最大值: {valid_data.max():.2f}\n")
                            stat_file.write(f"  平均值: {valid_data.mean():.2f}\n")
                            stat_file.write(f"  标准差: {valid_data.std():.2f}\n")
                
                # 处理windowauclate_mean统计
                if 'windowauclate_mean' in df.columns:
                    valid_windowauc = df['windowauclate_mean'][df['windowauclate_mean'] != -1]
                    if len(valid_windowauc) > 0:
                        # 控制台输出
                        print(f"\nwindowauclate_mean统计:")
                        print(f"  最小值: {valid_windowauc.min():.6f}")
                        print(f"  最大值: {valid_windowauc.max():.6f}")
                        print(f"  平均值: {valid_windowauc.mean():.6f}")
                        print(f"  标准差: {valid_windowauc.std():.6f}")
                        
                        # 文件输出
                        stat_file.write(f"\nwindowauclate_mean统计:\n")
                        stat_file.write(f"  最小值: {valid_windowauc.min():.6f}\n")
                        stat_file.write(f"  最大值: {valid_windowauc.max():.6f}\n")
                        stat_file.write(f"  平均值: {valid_windowauc.mean():.6f}\n")
                        stat_file.write(f"  标准差: {valid_windowauc.std():.6f}\n")
                
                # 处理windowacclate_mean统计
                if 'windowacclate_mean' in df.columns:
                    valid_windowacc = df['windowacclate_mean'][df['windowacclate_mean'] != -1]
                    if len(valid_windowacc) > 0:
                        # 控制台输出
                        print(f"\nwindowacclate_mean统计:")
                        print(f"  最小值: {valid_windowacc.min():.6f}")
                        print(f"  最大值: {valid_windowacc.max():.6f}")
                        print(f"  平均值: {valid_windowacc.mean():.6f}")
                        print(f"  标准差: {valid_windowacc.std():.6f}")
                        
                        # 文件输出
                        stat_file.write(f"\nwindowacclate_mean统计:\n")
                        stat_file.write(f"  最小值: {valid_windowacc.min():.6f}\n")
                        stat_file.write(f"  最大值: {valid_windowacc.max():.6f}\n")
                        stat_file.write(f"  平均值: {valid_windowacc.mean():.6f}\n")
                        stat_file.write(f"  标准差: {valid_windowacc.std():.6f}\n")
            
            print(f"\n评估统计信息已保存到文件: {stat_file_path}")
                    
        except Exception as e:
            print(f"保存结果时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("警告：没有收集到任何评估结果")
    if params['use_wandb'] == 1:
        import wandb
        with open("../configs/wandb.json") as fin:
            wandb_config = json.load(fin)
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json")) as fin:
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

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = os.path.join(save_dir, f"{model.emb_type}_test_predictions.txt")

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
    save_test_window_path = os.path.join(save_dir, f"{model.emb_type}_test_window_predictions.txt")
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
        save_test_question_window_path = os.path.join(save_dir, f"{model.emb_type}_test_question_window_predictions.txt")

        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc" + key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc" + key] = qw_testaccs[key]

    print(dres)
    raw_config = json.load(open(os.path.join(save_dir, "config.json")))
    dres.update(raw_config['params'])

    if params['use_wandb'] == 1:
        wandb.log(dres)
    print(f"windowauclate_mean: {dres['windowauclate_mean']}")
    print(f"windowacclate_mean: {dres['windowacclate_mean']}")
    
    # 将评估结果保存到 save_dir 目录下的 evaluation_results.json 文件
    
    results_path = os.path.join(save_dir, "evaluation_results.json")
    try:
        with open(results_path, "w") as fout:
            json.dump(dres, fout, indent=4, ensure_ascii=False)
        print(f"评估结果已保存到 {results_path}")
    except Exception as e:
        print(f"保存评估结果时出错: {e}")
    
    
    
    # === 新增代码：在所有学生评估前预测区间数据 ===
    # print("\n=== 开始预测区间数据文件 ===")
    
    # # 加载模型配置（与evaluate_single_student中相同）
    # with open(os.path.join(save_dir, "config.json")) as fin:
    #     config = json.load(fin)
    #     model_config = copy.deepcopy(config["model_config"])
    #     for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
    #         if remove_item in model_config:
    #             del model_config[remove_item]
    #     trained_params = config["params"]
    #     fold = trained_params["fold"]
    #     model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
    #     if model_name in ["saint", "sakt", "atdkt"]:
    #         train_config = config["train_config"]
    #         seq_len = train_config["seq_len"]
    #         model_config["seq_len"] = seq_len

    # with open("../configs/data_config.json") as fin:
    #     curconfig = copy.deepcopy(json.load(fin))
    #     data_config = curconfig[dataset_name]
    #     data_config["dataset_name"] = dataset_name
    #     if model_name in ["dkt_forget", "bakt_time","dbakt"]:
    #         data_config["num_rgap"] = config["data_config"]["num_rgap"]
    #         data_config["num_sgap"] = config["data_config"]["num_sgap"]
    #         data_config["num_pcount"] = config["data_config"]["num_pcount"]
    #     elif model_name == "lpkt":
    #         data_config["num_at"] = config["data_config"]["num_at"]
    #         data_config["num_it"] = config["data_config"]["num_it"]
    
    
    # # # 加载数据配置
    # # with open("../configs/data_config.json") as fin:
    # #     data_config = json.load(fin)[dataset_name]
    
    # # 加载模型
    # model = load_model(model_name, model_config, data_config, emb_type, save_dir)
    
    # # 预测区间数据
    # interval_results = predict_interval_data(model, data_config, model_name, params["fusion_type"], save_dir)
    
    # # 保存区间预测结果到单独文件
    # interval_results_path = os.path.join(save_dir, "interval_data_predictions.json")
    # with open(interval_results_path, "w") as f:
    #     json.dump(interval_results, f, indent=4)
    # print(f"\n区间数据预测结果已保存到: {interval_results_path}")
    # print("=== 区间数据文件预测完成 ===\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)

    parser.add_argument("--start_student", type=int, default=1, help="开始评估的学生ID")
    # 添加新参数：统计信息文件目录
    # parser.add_argument("--stats_dir", type=str, default="", required=True, help="学生统计信息文件目录")

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
import json
import numpy as np
import os
from pathlib import Path

# ====================== 你提供的两个函数（保持不变） ======================
def gini_coefficient(wealth):
    if len(wealth) == 0:
        return 0.0
    wealth = np.array(wealth)
    if np.all(wealth == 0):
        return 0.0
    if np.any(wealth < 0):
        raise ValueError("财富值不能为负数")
    sorted_wealth = np.sort(wealth)
    n = len(wealth)
    index = np.arange(1, n + 1)
    numerator = np.sum((2 * index - n - 1) * sorted_wealth)
    denominator = n * np.sum(sorted_wealth)
    return numerator / denominator


def calculate_wealth_metrics(wealth_list, alphas=[1.0, 2.0, 3.0]):
    if not wealth_list or len(wealth_list) == 0:
        avg = 0.0
        G = 0.0
        eawi_dict = {f'eawi_alpha_{a}': 0.0 for a in alphas}
        return avg, G, eawi_dict

    wealth = np.array(wealth_list)
    n = len(wealth)
    W = float(np.sum(wealth))
    average_wealth = W / n
    G = gini_coefficient(wealth_list)

    eawi_dict = {}
    for alpha in alphas:
        eawi = average_wealth * ((1 - G) ** alpha)
        key = f'eawi_alpha_{alpha:.1f}'.replace('.', '')
        eawi_dict[key] = float(eawi)

    return average_wealth, G, eawi_dict
# =====================================================================


def main(dir_path: str):
    dir_path = Path(dir_path).expanduser().resolve()  # 支持 ~ 和相对路径
    if not dir_path.is_dir():
        print(f"错误：路径不是一个文件夹 → {dir_path}")
        return

    # 固定文件名
    jsonl_file = dir_path / "evaluate_results_all_stu.jsonl"

    if not jsonl_file.exists():
        print(f"错误：在目录中未找到文件 → {jsonl_file}")
        print("   请确认文件名是否为：evaluate_results_all_stu.jsonl")
        return

    # 读取所有 window_testauc
    wealth_list = []
    with jsonl_file.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                auc = data.get("window_testauc")
                if auc is not None:
                    wealth_list.append(float(auc))
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析失败，已跳过: {e}")

    if not wealth_list:
        print("警告：没有读取到任何 window_testauc 数据！")
        return

    print(f"成功读取 {len(wealth_list)} 条 window_t ‐testauc 数据（来自 {jsonl_file.name}）")

    # 计算指标
    avg_wealth, gini, eawi_dict = calculate_wealth_metrics(wealth_list, alphas=[1.0, 2.0, 3.0])

    # 构造本次运行的时间戳（可选，方便你区分不同次运行）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "",
        "=" * 60,
        f"【{timestamp}】  window_testauc 不公平指标（文件: {jsonl_file.name}）",
        f"  平均值: {avg_wealth:.6f}",
        f"  Gini系数: {gini:.6f}",
        f"  EAWI_alpha_1.0 : {eawi_dict['eawi_alpha_10']:.6f}",
        f"  EAWI_alpha_2.0 : {eawi_dict['eawi_alpha_20']:.6f}",
        f"  EAWI_alpha_3.0 : {eawi_dict['eawi_alpha_30']:.6f}",
        f"  样本数量: {len(wealth_list)}",
        "=" * 60,
        ""
    ]

    # 追加写入 evaluation_statistics.txt
    out_file = dir_path / "evaluation_statistics.txt"
    with out_file.open('a', encoding='utf-8') as f:
        for ln in lines:
            f.write(ln + "\n")
        print(ln)   # 终端也同步显示

    print(f"\n结果已追加到：{out_file}")
    print("一次计算完成！你可以继续把别的文件夹拖进来运行～")


if __name__ == "__main__":
    import sys
    folder_path = "/data/pykt-toolkit/examples/saved_model/assist2009_0_0.0001_3407_32_200_0_1_saved_model_qikt_mamba_attn_0.5_256_1_2.0_gru_1"
    # if len(sys.argv) > 1:
    #     # 支持直接拖拽文件夹或输入路径
    #     folder_path = sys.argv[1]
    # else:
    #     # 如果没参数，就让用户手动输入
    #     folder_path = input("请粘贴或拖入包含 evaluate_results_all_stu.jsonl 的文件夹路径：").strip().strip('"\'')
    
    main(folder_path)
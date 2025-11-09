import pandas as pd
import numpy as np
import os

# ==================== 配置区 ====================
CSV_FILE = "your_results.csv"          # ←←← 修改成你的文件名
OUTPUT_TEX = "table_merged.tex"        # 输出的 LaTeX 文件名
DATASETS = ["assist2009", "nips_task34", "peiyou"]   # 三个必须同时出现的数据集
# ================================================

def load_and_filter(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. 只保留这三个数据集的行
    df = df[df['Dataset'].isin(DATASETS)].copy()
    
    # 2. 找出在三个数据集上都出现过的 Model
    model_counts = df['Model'].value_counts()
    valid_models = model_counts[model_counts == len(DATASETS)].index
    
    df = df[df['Model'].isin(valid_models)]
    return df

def aggregate_metrics(df):
    """
    对每个 Model，把三个数据集的同一指标聚合成  mean ± std  形式
    """
    # 需要合并的指标列（请根据你实际的列名调整）
    metric_cols = [
        'student_auc_mean', 'student_auc_std',
        'student_auc_range',
        'gini_coefficient',
        'eawi_alpha_10', 'eawi_alpha_20', 'eawi_alpha_30'
    ]
    
    # 检查列是否存在
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 中缺少以下列：{missing}")
    
    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = [
            ('mean', 'mean'),
            ('std',  'std')
        ]
    
    # 按 Model 聚合
    grouped = df.groupby('Model')[metric_cols].agg(**{
        f"{col}_{agg}": (col, agg) for col, agg in 
        [(c, 'mean') for c in metric_cols] + [(c, 'std') for c in metric_cols]
    })
    
    # 重命名列为我们想要的格式
    new_columns = []
    for col in metric_cols:
        new_columns.append(f"{col}_mean")
        new_columns.append(f"{col}_std")
    grouped.columns = new_columns
    
    # 合并成 mean ± std 字符串
    result_rows = []
    for model in grouped.index:
        row = {'Model': model}
        for col in metric_cols:
            mean_val = grouped.loc[model, f"{col}_mean"]
            std_val  = grouped.loc[model, f"{col}_std"]
            # 保留4位小数
            row[col] = f"{mean_val:.4f} \\pm {std_val:.4f}"
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    # 按 Model 字母序排序（可选）
    result_df = result_df.sort_values('Model').reset_index(drop=True)
    return result_df

def df_to_latex_bigtable(df):
    """
    生成一个漂亮的大表格：
    - 每行一个 Model
    - 每列一个指标的 mean ± std
    - 同时在表格最上方展示每个指标在三个数据集上的原始值（次级表头）
    """
    # 指标顺序（可调整）
    metric_order = [
        'student_auc_mean', 'student_auc_std',
        'student_auc_range',
        'gini_coefficient',
        'eawi_alpha_10', 'eawi_alpha_20', 'eawi_alpha_30'
    ]
    
    # 为了让表格更易读，我们把三个数据集作为多级列
    # 先恢复原始数据（带 Dataset 列），方便做 multi-index
    original = load_and_filter(CSV_FILE)
    
    # 设置多重索引
    pivot_dfs = []
    for metric in metric_order:
        piv = original.pivot_table(
            index='Model',
            columns='Dataset',
            values=metric,
            aggfunc='mean'
        )
        # 保证列顺序
        piv = piv[DATASETS]
        # 重命名列为 (metric, dataset)
        piv.columns = pd.MultiIndex.from_product([[metric], DATASETS])
        pivot_dfs.append(piv)
    
    # 合并所有指标
    full_multi = pd.concat(pivot_dfs, axis=1)
    full_multi = full_multi.sort_index(axis=1, level=0)
    
    # 再把我们之前算好的 mean±std 合并进来
    mean_std_df = aggregate_metrics(original)
    mean_std_df = mean_std_df.set_index('Model')
    final_df = pd.concat([full_multi, mean_std_df[metric_order]], axis=1)
    
    # ------------------- 生成 LaTeX -------------------
    def _format_val(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)
    
    # 多级列转成 LaTeX 的 \multicolumn + \cmidrule 形式
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{三个数据集上完整指标对比（均保留4位小数）}\n"
    latex += "\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l|" + "ccc|" * len(metric_order) + "c}\n"
    latex += "\\toprule\n"
    
    # 第一行：指标名
    header1 = "Model "
    header2 = " "
    for metric in metric_order:
        nice_name = metric.replace("_", "\\_")
        header1 += f"& \\multicolumn{{3}}{{c|}}{{{nice_name}}} "
        header2 += "& assist2009 & nips\\_task34 & peiyou "
    header1 += "& \\textbf{{Mean$\\pm$Std}} \\\\\n"
    header2 += "& \\multicolumn{{1}}{{c}}{{\\textbf{{Mean$\\pm$Std}}}} \\\\\n\\midrule\n"
    latex += header1 + header2
    
    # 数据行
    for model in final_df.index:
        row = model.replace("_", "\\_") + " "
        for metric in metric_order:
            vals = [final_df.loc[model, (metric, ds)] for ds in DATASETS]
            formatted = " & ".join(_format_val(v) if not pd.isna(v) else "-" for v in vals)
            row += f"& {formatted} "
        # 最后一列 mean±std
        mean_std_val = final_df.loc[model, metric_order[-1]].split(" \\pm ")[0]  # 随便取一个，这里已经保证所有指标都有
        # 更稳妥的做法：直接从 mean_std_df 取
        actual_mean_std = mean_std_df.loc[model, metric_order[0]]  # 第一个指标的 mean±std 即可代表
        no, actual_mean_std = mean_std_df.loc[model, metric_order].iloc[0]  # 取第一个
        row += f"& \\textbf{{{actual_mean_std}}} "
        row += "\\\\\n"
        latex += row
    
    latex += "\\bottomrule\n\\end{tabular}\n}\n\\label{tab:three_datasets_full}\n\\end{table}"
    
    return latex

# ==================== 主流程 ====================
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"请把 CSV 文件命名为 {CSV_FILE} 放在脚本同目录下")
        exit(1)
    
    print("正在读取并筛选数据...")
    df_filtered = load_and_filter(CSV_FILE)
    print(f"筛选后剩余 {len(df_filtered)} 行，"
          f"同时出现在三个数据集的模型有 {df_filtered['Model'].nunique()} 个")
    
    print("正在生成 LaTeX 表格...")
    latex_code = df_to_latex_bigtable(df_filtered)
    
    with open(OUTPUT_TEX, "w", encoding="utf8") as f:
        f.write(latex_code)
    
    print(f"完成！LaTeX 表格已保存为 {OUTPUT_TEX}")
    print("\n你只需要在 .tex 文件中 \\input{table_merged.tex} 即可使用")
    print("\n表格预览（前几行）：")
    print(latex_code.split("\\\\\n")[:6])
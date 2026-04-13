import os
import pandas as pd
import argparse

def run_diagnostics(primary_data_path, meta_data_dir, task_name="task_3_4"):
    print("="*40)
    print("🕵️‍♂️ 启动独立数据诊断程序...")
    print("="*40)

    answer_metadata_path = os.path.join(meta_data_dir, f"answer_metadata_{task_name}.csv")
    question_metadata_path = os.path.join(meta_data_dir, f"question_metadata_{task_name}.csv")

    # 1. 基础数据加载
    print(f"\n[1/4] 读取源文件...")
    df_primary = pd.read_csv(primary_data_path)
    df_answer = pd.read_csv(answer_metadata_path)
    df_question = pd.read_csv(question_metadata_path)
    
    print(f"  👉 主表 (df_primary) 行数: {len(df_primary)}")
    print(f"  👉 答题表 (df_answer) 行数: {len(df_answer)}")

    # 2. 核心排查：检查用来 merge 的关键列的数据类型是否一致
    print(f"\n[2/4] 检查键值(AnswerId)数据类型...")
    print(f"  👉 df_primary['AnswerId'] 类型: {df_primary['AnswerId'].dtype}")
    print(f"  👉 df_answer['AnswerId'] 类型: {df_answer['AnswerId'].dtype}")
    if df_primary['AnswerId'].dtype != df_answer['AnswerId'].dtype:
        print("  ⚠️ 警告: 两个表的 AnswerId 类型不一致，这可能是 Merge 后变 NaN 的直接原因！")

    # 3. 模拟真实的 Merge 过程
    print(f"\n[3/4] 模拟 Left Merge...")
    df_merge = df_primary.merge(df_answer[['AnswerId', 'DateAnswered']], how='left', on='AnswerId')
    df_merge = df_merge.merge(df_question[["QuestionId", "SubjectId"]], how='left', on='QuestionId')

    # 4. 统计即将被 Dropna 杀掉的数据
    print(f"\n[4/4] 执行 Dropna 前的伤亡统计...")
    
    # 看看具体是哪一列导致了 NaN
    missing_date = df_merge['DateAnswered'].isna().sum()
    missing_subject = df_merge['SubjectId'].isna().sum()
    
    print(f"  ❌ 找不到时间的记录 (DateAnswered 缺失): {missing_date} 条")
    print(f"  ❌ 找不到知识点的记录 (SubjectId 缺失): {missing_subject} 条")

    subset_cols = ["UserId", "SubjectId", "IsCorrect", "DateAnswered", "QuestionId"]
    df_to_drop = df_merge[df_merge[subset_cols].isna().any(axis=1)]
    
    print(f"\n🚨 结论: 如果执行你的 dropna，将有 {len(df_to_drop)} 条数据被删除！")
    
    if len(df_to_drop) > 0:
        print("\n👀 抽样看前 5 条被删掉的数据（看看 NaN 到底在哪）：")
        print(df_to_drop[subset_cols].head())
        
    print("\n" + "="*40 + " 诊断结束 " + "="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径写你报错的那个数据集路径
    parser.add_argument("-p", "--primary", type=str, default="../data/nips_task34/train_task_3_4.csv")
    parser.add_argument("-m", "--meta", type=str, default="/root/autodl-tmp/pykt-toolkit/data/nips_task34/metadata/")
    args = parser.parse_args()

    run_diagnostics(args.primary, args.meta)
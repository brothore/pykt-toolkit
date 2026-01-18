import os

# 这是报错日志中生成的中间文件路径
file_path = '/root/pykt-toolkit/data/nips_task34/data.txt'

print(f"正在检查文件: {file_path}")

if not os.path.exists(file_path):
    print("错误: 文件不存在！可能是上一步预处理完全失败了。")
else:
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        
    print(f"文件总行数: {len(lines)}")
    
    # 检查前 3 个学生的记录 (每个学生占 6 行，检查前 18 行)
    # 报错的行通常是索引 3, 9, 15...
    for i in range(min(20, len(lines))):
        line = lines[i].strip()
        
        # 打印每一行内容的预览
        print(f"[Line {i}] 内容: {line[:50]}..." + (f" (剩余 {len(line)-50} 字符)" if len(line)>50 else ""))
        
        # 专门检查报错的 Response 行 (索引模 6 余 3)
        if i % 6 == 3:
            print(f"  >>> 正在诊断第 {i} 行 (Responses)...")
            if "NA" in line:
                print("  >>> 跳过: 内容为 NA")
                continue
                
            values = line.split(',')
            has_error = False
            for idx, val in enumerate(values):
                try:
                    # 原始代码使用的是 int(val)，我们测试一下
                    int_val = int(val)
                    if int_val not in [0, 1]:
                        print(f"    [异常数值] 位置 {idx}: 值 '{val}' 不是 0 或 1")
                except ValueError:
                    print(f"    [格式错误] 位置 {idx}: 无法将 '{val}' 转换为整数。")
                    # 顺便测试一下是不是浮点数
                    try:
                        float_val = float(val)
                        print(f"    [提示] 这是一个浮点数 '{val}'。需要修改代码为 int(float(val))。")
                    except:
                        pass
                    has_error = True
                    break # 发现一个错误就停止当前行检查
            
            if not has_error:
                print("  >>> ✅ 这一行格式通过检查 (全是整数 0 或 1)")
            else:
                print("  >>> ❌ 这一行会导致报错！")
        print("-" * 30)
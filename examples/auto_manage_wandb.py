import sys
import os
import warnings
import subprocess

warnings.filterwarnings("ignore")

try:
    import wandb
except ImportError:
    print("❌ 错误: 找不到 wandb 库")
    sys.exit(1)

# 配置
api_key = "b2fd3c192e86f37e55450d3c8894511ff8bce88d"
curr_project = "kt_toolkits_nips_task34_qikt_mamba_v1"
base_project_pattern = "kt_toolkits"
entity = None 

# 获取 Entity
try:
    api = wandb.Api() # 不指定 project 以便能获取所有 project
    if hasattr(api, 'default_entity') and api.default_entity:
        entity = api.default_entity
    else:
        try:
            entity = api.viewer.entity
        except:
            entity = api.viewer().entity
except:
    pass

mode = sys.argv[1] 
ids = sys.argv[2:]

# --- 打印表格的辅助函数 ---
def print_sweeps(sweeps, proj_name=""):
    # 定义列宽
    W_ID = 12
    W_STATE = 12
    W_RUNS = 8
    W_DATE = 12
    W_PROJ = 35 # 项目名列宽
    
    # 如果是 status-all，多显示一列 Project
    header_fmt = f"{'Sweep ID':<{W_ID}} | {'State':<{W_STATE}} | {'Runs':<{W_RUNS}} | {'Created':<{W_DATE}} | "
    if proj_name == "ALL":
        header_fmt += f"{'Project':<{W_PROJ}} | "
    header_fmt += "Name/Config"
    
    print("-" * 120)
    print(header_fmt)
    print("-" * 120)
    
    found = False
    for sweep in sweeps:
        found = True
        state = sweep.state
        
        try:
            run_count = sweep.run_count if hasattr(sweep, 'run_count') else len(list(sweep.runs))
        except:
            run_count = "?"

        created = sweep.created_at[:10] if hasattr(sweep, 'created_at') else "N/A"
        cfg_name = sweep.config.get('name', 'N/A')
        
        # 状态颜色
        state_padded = f"{state:<{W_STATE}}"
        if state in ["RUNNING", "FINISHED"]: state_disp = f"\033[92m{state_padded}\033[0m"
        elif state == "PAUSED": state_disp = f"\033[93m{state_padded}\033[0m"
        elif state in ["CANCELED", "KILLED", "STOPPED"]: state_disp = f"\033[91m{state_padded}\033[0m"
        else: state_disp = state_padded

        row_fmt = f"{sweep.id:<{W_ID}} | {state_disp} | {str(run_count):<{W_RUNS}} | {created:<{W_DATE}} | "
        if proj_name == "ALL":
            # 截断过长的项目名
            p_name = sweep.project
            if len(p_name) > W_PROJ - 1: p_name = p_name[:W_PROJ-3] + "..."
            row_fmt += f"{p_name:<{W_PROJ}} | "
        
        row_fmt += f"{cfg_name}"
        print(row_fmt)
    
    if not found:
        print("   (没有找到 Sweep)")
    print("-" * 120)


# ==========================
#  模式: STATUS (当前项目)
# ==========================
if mode == "status":
    print(f"\n📊 项目 [{curr_project}] 的状态:")
    try:
        sweeps = api.project(curr_project, entity=entity).sweeps()
        print_sweeps(sweeps)
    except Exception as e:
        print(f"❌ 获取失败: {e}")

# ==========================
#  模式: STATUS-ALL (所有项目)
# ==========================
elif mode == "status-all":
    print(f"\n🌍 全局扫描: Entity [{entity}] 下以 [{base_project_pattern}] 开头的项目")
    try:
        # 1. 获取所有项目
        projects = api.projects(entity=entity)
        
        target_projects = []
        for p in projects:
            if p.name.startswith(base_project_pattern):
                target_projects.append(p)
        
        if not target_projects:
            print("❌ 未找到匹配的项目。")
        else:
            # 2. 遍历项目获取 Sweep
            all_sweeps = []
            print(f"   -> 发现 {len(target_projects)} 个匹配项目，正在获取数据...")
            for p in target_projects:
                try:
                    # 仅获取最近的 Sweep，避免太慢
                    proj_sweeps = list(p.sweeps())
                    all_sweeps.extend(proj_sweeps)
                except:
                    pass
            
            # 3. 打印 (按时间倒序)
            # 简单的排序，如果 created_at 格式不对可能报错，这里简单 try一下
            try:
                all_sweeps.sort(key=lambda x: x.created_at, reverse=True)
            except:
                pass

            print_sweeps(all_sweeps, proj_name="ALL")

    except Exception as e:
        print(f"❌ 全局扫描失败: {e}")

# ==========================
#  模式: ACTION (CLI)
# ==========================
elif mode in ["pause", "resume", "stop"]:
    if not ids:
        print("❌ Error: 缺少 Sweep ID")
        sys.exit(1)
    
    action_map = {"pause": "--pause", "resume": "--resume", "stop": "--stop"}
    flag = action_map[mode]

    for sweep_id in ids:
        # 注意：这里我们优先尝试在当前 project 下操作
        # 如果是全局模式下看到的其他 project 的 ID，CLI 可能需要 user/proj/id 完整路径
        # 这里尝试简单处理：先用当前 project 上下文，失败则提示用户
        
        full_id = f"{entity}/{curr_project}/{sweep_id}" if entity else f"{curr_project}/{sweep_id}"
        print(f"🔄 尝试操作: {flag} {full_id}")
        
        exit_code = os.system(f"wandb sweep {flag} {full_id}")
        
        if exit_code != 0:
            print(f"⚠️  当前项目下失败，尝试直接操作 ID (适用于 wandb 自动推断): {sweep_id}")
            os.system(f"wandb sweep {flag} {sweep_id}")


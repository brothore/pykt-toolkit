import sys
import os
import warnings

# ==========================================
# 1. 强力屏蔽警告
# ==========================================
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
try:
    import pandas as pd
    pd.options.mode.chained_assignment = None
except ImportError:
    pass
import psutil
import wandb
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ==========================================
# 2. 颜色定义
# ==========================================
class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREY = '\033[90m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def colorize_status(status_str):
    if "Hybrid" in status_str:
        return f"{Colors.CYAN}{Colors.BOLD}{status_str}{Colors.RESET}"
    elif "Local Only" in status_str or "Unsynced" in status_str:
        return f"{Colors.YELLOW}{status_str}{Colors.RESET}"
    elif "Remote Only" in status_str:
        return f"{Colors.BLUE}{status_str}{Colors.RESET}"
    elif "Empty" in status_str:
        return f"{Colors.GREY}{status_str}{Colors.RESET}"
    elif "Finished" in status_str:
        return f"{Colors.RESET}{status_str}{Colors.RESET}"
    return status_str

def get_local_running_sweeps():
    """
    获取本地正在运行的 Sweep 信息。
    返回格式: 字典 { 'sweep_id': 'project_name' }
    """
    local_data = {}
    
    # 遍历所有进程，同时获取 cmdline 和 environ
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'environ']):
        try:
            pinfo = proc.info
            cmdline = pinfo['cmdline'] or []
            environ = pinfo['environ'] or {}
            
            # --- 方法 A: 检查环境变量 (最准) ---
            # 适用于实际的训练脚本 wandb_dkt_train.py
            w_sweep_id = environ.get('WANDB_SWEEP_ID')
            w_project = environ.get('WANDB_PROJECT')
            
            if w_sweep_id and w_project:
                local_data[w_sweep_id] = w_project
                continue # 找到了就跳过后续检查

            # --- 方法 B: 检查命令行 (针对 Agent 进程) ---
            # 格式通常是: wandb agent entity/project/sweep_id
            cmd_str = " ".join(cmdline)
            if 'wandb' in cmd_str and 'agent' in cmd_str:
                for arg in cmdline:
                    # 寻找包含 "/" 的参数，且假设它是 entity/project/sweep_id
                    if '/' in arg:
                        parts = arg.split('/')
                        if len(parts) >= 3:
                            # 提取最后一部分作为 sweep_id，倒数第二部分作为 project
                            s_id = parts[-1]
                            p_name = parts[-2]
                            local_data[s_id] = p_name
                            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    return local_data

def main():
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print(f"{Colors.RED}Error: WANDB_API_KEY not set.{Colors.RESET}")
        return

    print(f"{Colors.GREY}Scanning WandB and Local Processes...{Colors.RESET}", end='\r')
    
    try:
        api = wandb.Api()
        projects = list(api.projects())
    except Exception as e:
        print(f"\n{Colors.RED}Connection Error: {e}{Colors.RESET}")
        return

    # 获取本地数据 {sweep_id: project_name}
    local_running_map = get_local_running_sweeps()
    
    data = []

    for project in projects:
        project_name = project.name
        entity = project.entity
        
        try:
            sweeps = list(project.sweeps())
        except:
            sweeps = []
        
        total_sweeps = len(sweeps)
        active_sweeps_cloud = 0  # 云端显示 Running 的数量
        active_agents_local = 0  # 本地匹配到的数量
        
        for sweep in sweeps:
            # 1. 检查云端状态
            is_cloud_running = (sweep.state == "RUNNING")
            if is_cloud_running:
                active_sweeps_cloud += 1
            
            # 2. 检查本地状态
            # 只要 Sweep ID 存在于本地检测列表中，且 Project 名称对得上
            # (防止不同 Project 有相同 Sweep ID 的极低概率冲突)
            local_proj = local_running_map.get(sweep.id)
            if local_proj and local_proj == project_name:
                active_agents_local += 1
            
        # ===============================================
        # 状态判定逻辑
        # ===============================================
        has_local = (active_agents_local > 0)
        has_cloud = (active_sweeps_cloud > 0)

        if has_local and has_cloud:
            raw_status = "Hybrid Running 🚀" 
        elif has_local and not has_cloud:
            raw_status = "Local Unsynced ⚠️"
        elif not has_local and has_cloud:
            raw_status = "Remote Only ☁️"
        elif total_sweeps > 0:
            raw_status = "Finished/Stop ⚫"
        else:
            raw_status = "Empty ⚪"

        updated_at = getattr(project, 'updated_at', None)
        if updated_at:
            try:
                dt = pd.to_datetime(updated_at)
                updated_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                updated_str = "-"
        else:
            updated_str = "-"

        data.append({
            "Project": project_name,
            "Status": colorize_status(raw_status),
            "Cloud Sweeps": active_sweeps_cloud if active_sweeps_cloud > 0 else "-",
            "Local Agents": f"{Colors.GREEN}{active_agents_local}{Colors.RESET}" if active_agents_local > 0 else "-",
            "Total": total_sweeps,
            "Updated": updated_str
        })

    print(" " * 60, end='\r')

    if not data:
        print("No projects found.")
    else:
        df = pd.DataFrame(data)
        
        df['_sort_key'] = df['Status'].apply(lambda x: 0 if "Running" in x or "Unsynced" in x else 1)
        df['_date_key'] = pd.to_datetime(df['Updated'], errors='coerce')
        df = df.sort_values(by=['_sort_key', '_date_key'], ascending=[True, False])
        df = df.drop(columns=['_sort_key', '_date_key'])

        print(f"{Colors.BOLD}WandB Projects Status{Colors.RESET}")
        
        if HAS_TABULATE:
            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, stralign="center"))
        else:
            print(df.to_string(index=False))

if __name__ == "__main__":
    main()
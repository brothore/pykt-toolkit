import sys
import os
import warnings
import subprocess
import signal

# ==========================================
# 1. 强力屏蔽警告
# ==========================================
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import logging
    logging.getLogger("pydantic").setLevel(logging.ERROR)
    logging.getLogger("wandb").setLevel(logging.ERROR)
except:
    pass

import argparse
import psutil
import wandb

# ==========================================
# 2. 颜色定义
# ==========================================
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def get_project_sweeps(api, entity, project_name):
    try:
        project = api.project(project_name, entity=entity)
        sweeps = project.sweeps()
        return list(sweeps)
    except Exception as e:
        print(f"{Colors.RED}[Error] Could not access project {entity}/{project_name}: {e}{Colors.RESET}")
        sys.exit(1)

def kill_local_agents(project_name, entity):
    """
    查找并灭杀本地属于该项目的进程 (支持检测环境变量和命令行)
    """
    print(f"{Colors.YELLOW}[Local] Scanning for local agents/runs of '{project_name}'...{Colors.RESET}")
    killed_count = 0
    target_str = f"{entity}/{project_name}" # 用于命令行匹配
    
    # 获取当前用户ID，避免误杀其他用户的进程
    try:
        current_uid = os.getuid()
    except:
        current_uid = None

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'environ', 'uids']):
        try:
            # 安全检查：只杀自己用户的进程
            if current_uid is not None and proc.info['uids'] and proc.info['uids'].real != current_uid:
                continue

            pinfo = proc.info
            cmdline = pinfo['cmdline'] or []
            environ = pinfo['environ'] or {}
            is_target = False

            # --- 判定标准 A: 环境变量 (WANDB_PROJECT) ---
            # 这是识别训练脚本最准确的方法
            env_proj = environ.get('WANDB_PROJECT')
            if env_proj and env_proj == project_name:
                is_target = True

            # --- 判定标准 B: 命令行 (wandb agent) ---
            # 这是识别 Agent 进程的方法
            if not is_target:
                cmd_str = " ".join(cmdline)
                if 'wandb' in cmd_str and 'agent' in cmd_str:
                    # 检查是否包含 entity/project
                    if target_str in cmd_str or project_name in cmd_str:
                        is_target = True

            if is_target:
                print(f"  -> Killing Process PID {pinfo['pid']}: {' '.join(cmdline)[:100]}...")
                try:
                    # 尝试优雅终止
                    proc.terminate()
                    # 给一点时间
                    try:
                        proc.wait(timeout=1)
                    except psutil.TimeoutExpired:
                        proc.kill() # 强制杀死
                    killed_count += 1
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    print(f"     {Colors.RED}[Failed] Permission denied for PID {pinfo['pid']}{Colors.RESET}")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
            
    if killed_count > 0:
        print(f"{Colors.GREEN}[Local] Successfully killed {killed_count} local process(es).{Colors.RESET}")
    else:
        print(f"[Local] No running agents/train-scripts found for this project locally.")

def call_wandb_cli(action_flag, sweep_path):
    cmd = ["wandb", "sweep", action_flag, sweep_path]
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def update_sweeps_state(sweeps, action):
    action_map = { 'stop': '--stop', 'pause': '--pause', 'resume': '--resume' }
    cli_flag = action_map.get(action)
    if not cli_flag: return

    print(f"{Colors.CYAN}[Cloud] Updating {len(sweeps)} sweeps to state: {action.upper()}...{Colors.RESET}")
    
    cnt = 0
    for sweep in sweeps:
        if sweep.state in ['FINISHED', 'CANCELED'] and action != 'resume': continue
        if sweep.state == 'RUNNING' and action == 'resume': continue
        if sweep.state == 'PAUSED' and action == 'pause': continue

        sweep_path = f"{sweep.entity}/{sweep.project}/{sweep.id}"
        success, err_msg = call_wandb_cli(cli_flag, sweep_path)
        
        if success: cnt += 1
        else: print(f"  -> {Colors.RED}Failed {sweep.id}{Colors.RESET}: {err_msg}")
    
    if cnt > 0:
        print(f"{Colors.GREEN}[Cloud] Successfully sent '{action.upper()}' command to {cnt} sweeps.{Colors.RESET}")
    else:
        print(f"[Cloud] No sweeps needed updates.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--action", type=str, choices=['pause', 'resume', 'stop'], required=True)
    args = parser.parse_args()

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key: return

    try:
        api = wandb.Api()
        entity = api.default_entity
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        return

    print(f"Target: {Colors.BOLD}{entity}/{args.project}{Colors.RESET}")
    print(f"Action: {Colors.BOLD}{args.action.upper()}{Colors.RESET}")

    if args.action == 'stop':
        kill_local_agents(args.project, entity)
    
    sweeps = get_project_sweeps(api, entity, args.project)
    if not sweeps:
        print("No sweeps found.")
        return

    update_sweeps_state(sweeps, args.action)

    if args.action == 'resume':
        print(f"\n{Colors.YELLOW}[Tip] Cloud set to RUNNING. Restart local agents manually.{Colors.RESET}")

if __name__ == "__main__":
    main()
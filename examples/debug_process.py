import psutil
import os
import sys

def debug_processes():
    print(f"{'PID':<10} {'Name':<15} {'Status':<10} {'Command Line'}")
    print("="*100)
    
    found_any = False
    
    # 遍历所有进程
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'environ']):
        try:
            # 过滤条件：只看 python, wandb 相关，或者高 CPU 的进程
            pinfo = proc.info
            name = pinfo['name'].lower()
            cmdline = pinfo['cmdline'] or []
            cmd_str = " ".join(cmdline)
            
            is_target = False
            
            # 1. 检查是否是 python 进程
            if 'python' in name:
                is_target = True
            # 2. 检查是否是 wandb 相关进程
            elif 'wandb' in name or 'wandb' in cmd_str:
                is_target = True
            
            if is_target:
                found_any = True
                print(f"{pinfo['pid']:<10} {pinfo['name']:<15} {pinfo['status']:<10} {cmd_str}")
                
                # --- 深度诊断：尝试获取环境变量 ---
                # WandB 运行时通常会在环境变量里注入 WANDB_PROJECT 或 WANDB_SWEEP_ID
                # 这比命令行更靠谱
                try:
                    env = pinfo['environ']
                    if env:
                        w_proj = env.get('WANDB_PROJECT', 'N/A')
                        w_sweep = env.get('WANDB_SWEEP_ID', 'N/A')
                        w_run_id = env.get('WANDB_RUN_ID', 'N/A')
                        if w_proj != 'N/A' or w_sweep != 'N/A':
                            print(f"    └── [ENV Detected] Project: {w_proj} | SweepID: {w_sweep} | RunID: {w_run_id}")
                except (psutil.AccessDenied,  Exception):
                    pass # 没权限看环境变量就算了

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not found_any:
        print("No python or wandb processes found (Permission denied?).")

if __name__ == "__main__":
    # 确保有 root 权限或者与运行进程相同的用户权限
    debug_processes()
import subprocess
import os
import time
import itertools
from typing import List, Dict, Any

# python -u run_experiments.py > log.log

def run_experiment(params: Dict[str, Any]) -> None:
    """运行单个实验
    
    Args:
        params: 实验参数字典
    """
    # 构建命令行参数
    cmd = ["python", "main.py"]
    
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")
    
    # 创建实验输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{params['model_name']}_dim{params['intrinsic_dim']}_use_coupled_{params['use_coupled']}_{timestamp}"
    
    # 添加输出目录参数
    if "--output_dir" not in " ".join(cmd):
        output_dir = f"./experiments/{exp_name}"
        cmd.append(f"--output_dir={output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        # 如果已经指定了output_dir，确保目录存在
        for arg in cmd:
            if arg.startswith("--output_dir="):
                output_dir = arg.split("=")[1]
                os.makedirs(output_dir, exist_ok=True)
                break
    
    # 添加保存路径
    if "--save_path" not in " ".join(cmd):
        save_path = os.path.join(output_dir, "model_save.pth")
        cmd.append(f"--save_path={save_path}")
    
    # 运行命令
    print(f"运行实验: {' '.join(cmd)}")
    subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(f"实验完成，结果保存在: {output_dir}")

def grid_search(param_grid: Dict[str, List]) -> None:
    """进行网格搜索
    
    Args:
        param_grid: 参数网格字典，每个键对应一个参数名，值为该参数的可能取值列表
    """
    # 确保存在实验目录
    os.makedirs("./experiments", exist_ok=True)
    
    # 生成所有参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    experiments = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"将运行 {len(experiments)} 个实验...")
    
    # 运行每个实验
    for i, params in enumerate(experiments):
        print(f"\n运行实验 {i+1}/{len(experiments)}")
        run_experiment(params)
        # 实验之间暂停一段时间，避免资源竞争
        time.sleep(1)

if __name__ == "__main__":
    # 示例参数网格 - 使用完整参数集
    example_grid = {
        # 模型参数
        "model_name": ["deepseek-llama"],
        "intrinsic_dim": [8, 16, 24],
        "n_prompt_tokens": [10],  # 保持默认
        "hf_dir": ["/root/autodl-tmp/deepseek-llama"],  # 使用默认路径
        
        # 优化参数
        "n_init": [24],
        "n_iterations": [12],  # 减少迭代次数加快示例运行
        "batch_size": [12],
        "use_coupled": [True, False],
        
        # 控制参数
        "train_baysion": [True],  # 必须为True才能运行贝叶斯优化
        "train_scav": [False],    # 通常不需要重新训练SCAV
        "test_api": [False],      # 是否运行API测试
        "use_deepseek_api": [False],  # 是否使用DeepSeek API进行攻击测试
        "seed": [2142]            # 保持默认种子
    }
    
    # 默认运行实验
    print("运行实验...")
    instruct_dim_grid = {
        # 模型参数
        "model_name": ["deepseek-llama"],
        "intrinsic_dim": [8, 12, 16, 24],
        "n_prompt_tokens": [10],
        "hf_dir": ["/root/autodl-tmp/deepseek-llama"],
        
        # 优化参数
        "n_init": [24],
        "n_iterations": [12],
        "batch_size": [12],
        "use_coupled": [True],
        
        # 控制参数
        "train_baysion": [True],
        "train_scav": [False],
        "test_api": [False],
        "use_deepseek_api": [False],
        "seed": [2142],
    }
    # grid_search(instruct_dim_grid)
    
    coupled_grid = {
        # 模型参数
        "model_name": ["deepseek-llama"],
        "intrinsic_dim": [8, 12, 16],
        "n_prompt_tokens": [10],
        "hf_dir": ["/root/autodl-tmp/deepseek-llama"],
        
        # 优化参数
        "n_init": [24],
        "n_iterations": [12],
        "batch_size": [12],
        "use_coupled": [True, False],
        
        # 控制参数
        "train_baysion": [True],
        "train_scav": [False],
        "test_api": [False],
        "use_deepseek_api": [False],
        "seed": [2142],
    }
    # grid_search(coupled_grid)
    
    iter_grid = {
        # 模型参数
        "model_name": ["deepseek-llama"],
        "intrinsic_dim": [12, 16],
        "n_prompt_tokens": [10],
        "hf_dir": ["/root/autodl-tmp/deepseek-llama"],
        
        # 优化参数
        "n_init": [24],
        "n_iterations": [24],
        "batch_size": [12],
        "use_coupled": [True],
        
        # 控制参数
        "train_baysion": [True],
        "train_scav": [False],
        "test_api": [False],
        "use_deepseek_api": [False],
        "seed": [2142],
    }
    # grid_search(iter_grid)
    model_grid = {
        # 模型参数
        "model_name": ["deepseek-llama"],
        "intrinsic_dim": [12],
        "n_prompt_tokens": [10],
        "hf_dir": ["/root/autodl-tmp/deepseek-llama"],
        
        # 优化参数
        "n_init": [24],
        "n_iterations": [16],
        "batch_size": [12],
        "use_coupled": [True],
        
        # 控制参数
        "train_baysion": [True],
        "train_scav": [False],
        "test_api": [False],
        "use_deepseek_api": [True],
        "seed": [2142],
    }
    grid_search(model_grid)
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.font_manager import FontProperties
import argparse
from typing import List, Dict, Optional, Any, Tuple

# 设置中文字体，需要确保系统中安装了相应字体
try:
    # Windows系统
    font = FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    try:
        # Linux/Mac系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未能找到合适的中文字体，图表中的中文可能无法正确显示")

def plot_optimization_progress(data_path: str, output_dir: str):
    """绘制优化进度图
    
    Args:
        data_path: 优化进度数据文件路径
        output_dir: 输出目录
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plt.figure(figsize=(15, 7))
    
    # 绘制得分变化曲线
    plt.subplot(1, 2, 1)
    plt.plot(data["iteration"], data["best_score"], 'b-o', label='最佳得分')
    plt.plot(data["iteration"], data["avg_score"], 'r--^', label='平均得分')
    plt.xlabel('迭代次数')
    plt.ylabel('得分')
    plt.title('优化过程中得分变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制计算时间曲线
    plt.subplot(1, 2, 2)
    plt.plot(data["iteration"], data["computation_time"], 'g-o')
    plt.xlabel('迭代次数')
    plt.ylabel('计算时间 (秒)')
    plt.title('优化过程中计算时间变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_progress.png"), dpi=300)
    plt.close()
    
    print(f"优化进度图已保存至 {os.path.join(output_dir, 'optimization_progress.png')}")

def plot_attack_success_rate(data_path: str, output_dir: str):
    """绘制攻击成功率图
    
    Args:
        data_path: 攻击成功率数据文件路径
        output_dir: 输出目录
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data["call_nums"], data["success_rates"], 'b-o')
    plt.xlabel('调用次数')
    plt.ylabel('攻击成功率')
    plt.title(f'攻击成功率变化 (窗口大小: {data["window_size"]})')
    plt.grid(True)
    plt.ylim([0, 1.05])
    
    plt.savefig(os.path.join(output_dir, "attack_success_rate.png"), dpi=300)
    plt.close()
    
    print(f"攻击成功率图已保存至 {os.path.join(output_dir, 'attack_success_rate.png')}")

def plot_parameter_influence(data_path: str, output_dir: str):
    """绘制参数影响图
    
    Args:
        data_path: 参数影响数据文件路径
        output_dir: 输出目录
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plt.figure(figsize=(15, 7))
    
    # 绘制内在维度与最终得分的关系
    plt.subplot(1, 2, 1)
    plt.scatter(data["intrinsic_dim"], data["final_score"], s=100, c='blue', alpha=0.7)
    for i, (x, y) in enumerate(zip(data["intrinsic_dim"], data["final_score"])):
        plt.annotate(f"维度: {x}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('内在维度')
    plt.ylabel('最终得分')
    plt.title('内在维度对最终得分的影响')
    plt.grid(True)
    
    # 绘制批量大小与总时间的关系
    plt.subplot(1, 2, 2)
    plt.scatter(data["batch_size"], data["total_time"], s=100, c='red', alpha=0.7)
    for i, (x, y) in enumerate(zip(data["batch_size"], data["total_time"])):
        plt.annotate(f"批量: {x}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('批量大小')
    plt.ylabel('总计算时间 (秒)')
    plt.title('批量大小对计算时间的影响')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_influence.png"), dpi=300)
    plt.close()
    
    print(f"参数影响图已保存至 {os.path.join(output_dir, 'parameter_influence.png')}")

def plot_parameter_heatmap(data_path: str, output_dir: str):
    """绘制参数热力图
    
    Args:
        data_path: 参数热力图数据文件路径
        output_dir: 输出目录
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dim1 = data["dim1"]
    dim2 = data["dim2"]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data["x_values"], data["y_values"], 
                c=data["scores"], s=[s*50 for s in data["scores"]], 
                cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='得分')
    plt.xlabel(f'参数维度 {dim1}')
    plt.ylabel(f'参数维度 {dim2}')
    plt.title(f'参数空间得分分布 (维度 {dim1} vs 维度 {dim2})')
    plt.grid(True)
    
    # 标注部分高分点
    top_indices = np.argsort(data["scores"])[-5:]  # 取得分最高的5个点
    for i in top_indices:
        plt.annotate(f"{data['scores'][i]:.2f}", 
                    (data["x_values"][i], data["y_values"][i]), 
                    textcoords="offset points", 
                    xytext=(0,5), 
                    ha='center',
                    fontsize=9)
    
    plt.savefig(os.path.join(output_dir, f"parameter_heatmap_dim{dim1}_dim{dim2}.png"), dpi=300)
    plt.close()
    
    print(f"参数热力图已保存至 {os.path.join(output_dir, f'parameter_heatmap_dim{dim1}_dim{dim2}.png')}")

def plot_experiments_comparison(data_path: str, output_dir: str):
    """绘制实验对比图
    
    Args:
        data_path: 实验对比数据文件路径
        output_dir: 输出目录
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    experiments = data["experiments"]
    metrics = data["metrics"]
    
    # 为每个指标创建一个图
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=(12, 6))
        bars = plt.bar(experiments, metric_values, width=0.6)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xlabel('实验')
        plt.ylabel(metric_name)
        plt.title(f'不同实验的{metric_name}对比')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"experiments_comparison_{metric_name}.png"), dpi=300)
        plt.close()
        
        print(f"实验对比图({metric_name})已保存至 {os.path.join(output_dir, f'experiments_comparison_{metric_name}.png')}")

def plot_attack_metrics(data_path: str, output_dir: str):
    """绘制攻击指标分析图
    
    Args:
        data_path: 攻击指标数据文件路径
        output_dir: 输出目录
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    plt.figure(figsize=(15, 10))
    
    # 绘制毒性得分随时间变化
    plt.subplot(2, 2, 1)
    plt.plot(data["call_num"], data["toxicity_score"], 'r-o')
    plt.xlabel('调用次数')
    plt.ylabel('毒性得分')
    plt.title('毒性得分随时间变化')
    plt.grid(True)
    
    # 绘制输出长度分布
    plt.subplot(2, 2, 2)
    plt.scatter(data["call_num"], data["char_count"], alpha=0.7)
    plt.xlabel('调用次数')
    plt.ylabel('输出字符数')
    plt.title('输出长度分布')
    plt.grid(True)
    
    # 计算模板长度与毒性得分的关系
    plt.subplot(2, 2, 3)
    plt.scatter(data["template_length"], data["toxicity_score"], alpha=0.7)
    plt.xlabel('模板长度')
    plt.ylabel('毒性得分')
    plt.title('模板长度与毒性得分关系')
    plt.grid(True)
    
    # 检测行为类型统计
    if len(data["detected_behaviors"]) > 0:
        behaviors = {}
        for behavior_list in data["detected_behaviors"]:
            for behavior in behavior_list:
                if behavior in behaviors:
                    behaviors[behavior] += 1
                else:
                    behaviors[behavior] = 1
        
        if behaviors:
            plt.subplot(2, 2, 4)
            behavior_names = list(behaviors.keys())
            behavior_counts = list(behaviors.values())
            plt.bar(behavior_names, behavior_counts)
            plt.xlabel('行为类型')
            plt.ylabel('检测次数')
            plt.title('检测到的行为类型统计')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_metrics_analysis.png"), dpi=300)
    plt.close()
    
    print(f"攻击指标分析图已保存至 {os.path.join(output_dir, 'attack_metrics_analysis.png')}")

def plot_all_experiments_summary(experiments_dir: str, output_dir: str):
    """绘制所有实验的摘要对比图
    
    Args:
        experiments_dir: 包含多个实验目录的总目录
        output_dir: 输出目录
    """
    # 读取所有实验的摘要
    experiment_summaries = []
    experiment_names = []
    
    # 查找所有子目录中的experiment_summary.json文件
    summary_files = glob.glob(f"{experiments_dir}/*/experiment_summary.json")
    
    for summary_file in summary_files:
        exp_name = os.path.basename(os.path.dirname(summary_file))
        experiment_names.append(exp_name)
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            experiment_summaries.append(summary)
    
    if not experiment_summaries:
        print("未找到任何实验摘要文件")
        return
    
    # 提取关键指标
    intrinsic_dims = []
    batch_sizes = []
    use_coupled = []
    final_scores = []
    attack_success_rates = []
    safety_scores = []
    semantic_scores = []
    
    for summary in experiment_summaries:
        if "优化参数设置" in summary and "内在维度" in summary["优化参数设置"]:
            intrinsic_dims.append(summary["优化参数设置"]["内在维度"])
        else:
            intrinsic_dims.append(None)
            
        if "优化参数设置" in summary and "批量大小" in summary["优化参数设置"]:
            batch_sizes.append(summary["优化参数设置"]["批量大小"])
        else:
            batch_sizes.append(None)
            
        # 从实验名称中推断是否使用了耦合核
        if "use_coupled_True" in os.path.basename(os.path.dirname(summary_files[experiment_summaries.index(summary)])):
            use_coupled.append("耦合核")
        elif "use_coupled_False" in os.path.basename(os.path.dirname(summary_files[experiment_summaries.index(summary)])):
            use_coupled.append("标准核")
        else:
            use_coupled.append("未知")
            
        if "优化结果摘要" in summary and "最终最佳得分" in summary["优化结果摘要"]:
            final_scores.append(summary["优化结果摘要"]["最终最佳得分"])
        else:
            final_scores.append(None)
            
        if "攻击效果摘要" in summary and "攻击成功率" in summary["攻击效果摘要"]:
            attack_success_rates.append(summary["攻击效果摘要"]["攻击成功率"])
        else:
            attack_success_rates.append(None)
            
        if "攻击效果摘要" in summary and "平均安全绕过得分" in summary["攻击效果摘要"]:
            safety_scores.append(summary["攻击效果摘要"]["平均安全绕过得分"])
        else:
            safety_scores.append(None)
            
        if "攻击效果摘要" in summary and "平均语义相似度" in summary["攻击效果摘要"]:
            semantic_scores.append(summary["攻击效果摘要"]["平均语义相似度"])
        else:
            semantic_scores.append(None)
    
    # 创建图表
    plt.figure(figsize=(15, 12))
    
    # 1. 最终得分对比
    plt.subplot(2, 2, 1)
    bars = plt.bar(experiment_names, final_scores)
    plt.xlabel('实验')
    plt.ylabel('最终得分')
    plt.title('不同实验的最终得分对比')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        if height is not None:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 攻击成功率对比
    plt.subplot(2, 2, 2)
    bars = plt.bar(experiment_names, attack_success_rates)
    plt.xlabel('实验')
    plt.ylabel('攻击成功率')
    plt.title('不同实验的攻击成功率对比')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        if height is not None:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 安全绕过得分与语义相似度对比
    plt.subplot(2, 2, 3)
    x = np.arange(len(experiment_names))
    width = 0.35
    plt.bar(x - width/2, safety_scores, width, label='安全绕过得分')
    plt.bar(x + width/2, semantic_scores, width, label='语义相似度')
    plt.xlabel('实验')
    plt.ylabel('得分')
    plt.title('安全绕过得分与语义相似度对比')
    plt.xticks(x, experiment_names, rotation=45)
    plt.legend()
    
    # 4. 分组对比：内在维度/使用耦合核对最终得分的影响
    plt.subplot(2, 2, 4)
    # 创建DataFrame以便于分组
    df = pd.DataFrame({
        'experiment': experiment_names,
        'intrinsic_dim': intrinsic_dims,
        'use_coupled': use_coupled,
        'final_score': final_scores
    })
    
    if len(set(df['use_coupled'])) > 1:  # 如果有不同的核类型
        # 按核类型分组
        grouped_by_kernel = df.groupby('use_coupled')['final_score'].mean().reset_index()
        plt.bar(grouped_by_kernel['use_coupled'], grouped_by_kernel['final_score'], width=0.4)
        plt.xlabel('核类型')
        plt.ylabel('平均最终得分')
        plt.title('核类型对最终得分的影响')
    elif len(set(df['intrinsic_dim'])) > 1:  # 如果有不同的内在维度
        # 按内在维度分组
        valid_dims = [d for d in df['intrinsic_dim'] if d is not None]
        if valid_dims:
            dim_scores = []
            dims = []
            for dim in set(valid_dims):
                scores = [score for d, score in zip(intrinsic_dims, final_scores) if d == dim and score is not None]
                if scores:
                    dim_scores.append(np.mean(scores))
                    dims.append(dim)
            
            if dims:
                plt.bar([str(d) for d in dims], dim_scores, width=0.4)
                plt.xlabel('内在维度')
                plt.ylabel('平均最终得分')
                plt.title('内在维度对最终得分的影响')
    
    plt.tight_layout()
    
    # 保存图表
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "all_experiments_summary.png"), dpi=300)
    plt.close()
    
    print(f"所有实验摘要对比图已保存至 {os.path.join(output_dir, 'all_experiments_summary.png')}")

def plot_all_for_experiment(experiment_dir: str, output_dir: Optional[str] = None):
    """为单个实验绘制所有图表
    
    Args:
        experiment_dir: 实验目录
        output_dir: 输出目录，默认为实验目录下的plots子目录
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "plots")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 优化进度图
    opt_data_path = os.path.join(experiment_dir, "optimization_progress_plot_data.json")
    if os.path.exists(opt_data_path):
        plot_optimization_progress(opt_data_path, output_dir)
    
    # 攻击成功率图
    attack_data_path = os.path.join(experiment_dir, "attack_success_rate_data.json")
    if os.path.exists(attack_data_path):
        plot_attack_success_rate(attack_data_path, output_dir)
    
    # 参数影响图
    param_data_path = os.path.join(experiment_dir, "parameter_influence_plot_data.json")
    if os.path.exists(param_data_path):
        plot_parameter_influence(param_data_path, output_dir)
    
    # 攻击指标分析图
    metrics_data_path = os.path.join(experiment_dir, "attack_metrics.json")
    if os.path.exists(metrics_data_path):
        plot_attack_metrics(metrics_data_path, output_dir)
    
    # 参数热力图
    heatmap_files = glob.glob(os.path.join(experiment_dir, "parameter_heatmap_dim*_data.json"))
    for heatmap_file in heatmap_files:
        plot_parameter_heatmap(heatmap_file, output_dir)
    
    # 实验对比图
    comparison_data_path = os.path.join(experiment_dir, "experiments_comparison_plot_data.json")
    if os.path.exists(comparison_data_path):
        plot_experiments_comparison(comparison_data_path, output_dir)
    
    print(f"所有图表已保存至 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='绘制实验结果图表')
    parser.add_argument('--exp_dir', type=str, required=True, help='实验目录路径')
    parser.add_argument('--out_dir', type=str, default=None, help='输出目录路径，默认为实验目录下的plots子目录')
    parser.add_argument('--all_exps', action='store_true', help='是否绘制所有实验的摘要对比图')
    
    args = parser.parse_args()
    
    if args.all_exps:
        # 绘制所有实验的摘要对比图
        if args.out_dir is None:
            args.out_dir = os.path.join(args.exp_dir, "all_plots")
        plot_all_experiments_summary(args.exp_dir, args.out_dir)
    else:
        # 为单个实验绘制所有图表
        plot_all_for_experiment(args.exp_dir, args.out_dir)

if __name__ == "__main__":
    main()
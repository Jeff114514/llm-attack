import os
import csv
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Union

class ExperimentDataCollector:
    def __init__(self, output_dir="./experiment_data"):
        """初始化数据收集器
        
        Args:
            output_dir: 数据输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据存储
        self.optimization_data = {
            "iteration": [],
            "best_score": [],
            "avg_score": [],
            "computation_time": [],
            "ei_value": [],
            "batch_size": []
        }
        
        self.evaluation_data = {
            "call_num": [],
            "safety_evasion_score": [],
            "semantic_similarity": [],
            "attack_success": [],
            "template": []
        }
        
        self.hyperparameter_data = {
            "intrinsic_dim": [],
            "n_init": [],
            "n_iterations": [],
            "batch_size": [],
            "final_score": [],
            "total_time": []
        }
        
        self.best_points_data = {
            "iteration": [],
            "point_values": [],
            "score": []
        }
        
        # 实验开始时间
        self.start_time = time.time()
        self.current_iteration = 0
        
    def record_iteration_data(self, iteration: int, best_score: float, 
                              all_scores: List[float], computation_time: float,
                              ei_values: List[float]=None, batch_size: int=None):
        """记录每次迭代的数据
        
        Args:
            iteration: 迭代次数
            best_score: 当前迭代的最佳得分
            all_scores: 当前迭代的所有得分
            computation_time: 计算时间(秒)
            ei_values: 期望改进值(可选)
            batch_size: 批量大小(可选)
        """
        self.current_iteration = iteration
        self.optimization_data["iteration"].append(iteration)
        self.optimization_data["best_score"].append(best_score)
        self.optimization_data["avg_score"].append(np.mean(all_scores) if all_scores else 0)
        self.optimization_data["computation_time"].append(computation_time)
        
        if ei_values:
            avg_ei = np.mean(ei_values) if isinstance(ei_values, list) else ei_values
            self.optimization_data["ei_value"].append(avg_ei)
        else:
            self.optimization_data["ei_value"].append(None)
            
        if batch_size:
            self.optimization_data["batch_size"].append(batch_size)
        else:
            self.optimization_data["batch_size"].append(None)
            
    def record_evaluation_data(self, call_num: int, safety_score: float, 
                              semantic_score: float, attack_success: bool,
                              template: str):
        """记录评估结果数据
        
        Args:
            call_num: 调用次数
            safety_score: 安全绕过评分
            semantic_score: 语义相似度分数
            attack_success: 攻击是否成功
            template: 使用的模板
        """
        self.evaluation_data["call_num"].append(call_num)
        self.evaluation_data["safety_evasion_score"].append(safety_score)
        self.evaluation_data["semantic_similarity"].append(semantic_score)
        self.evaluation_data["attack_success"].append(1 if attack_success else 0)
        self.evaluation_data["template"].append(template)
        
    def record_best_point(self, iteration: int, point_values: List[float], score: float):
        """记录最佳点数据
        
        Args:
            iteration: 迭代次数
            point_values: 点的参数值
            score: 得分
        """
        self.best_points_data["iteration"].append(iteration)
        self.best_points_data["point_values"].append(point_values)
        self.best_points_data["score"].append(score)
        
    def record_hyperparameters(self, intrinsic_dim: int, n_init: int, 
                              n_iterations: int, batch_size: int,
                              final_score: float=None):
        """记录超参数数据
        
        Args:
            intrinsic_dim: 内在维度
            n_init: 初始采样点数量
            n_iterations: 迭代次数
            batch_size: 批量大小
            final_score: 最终得分(可选)
        """
        self.hyperparameter_data["intrinsic_dim"].append(intrinsic_dim)
        self.hyperparameter_data["n_init"].append(n_init)
        self.hyperparameter_data["n_iterations"].append(n_iterations)
        self.hyperparameter_data["batch_size"].append(batch_size)
        
        if final_score is not None:
            self.hyperparameter_data["final_score"].append(final_score)
        else:
            self.hyperparameter_data["final_score"].append(None)
            
        total_time = time.time() - self.start_time
        self.hyperparameter_data["total_time"].append(total_time)
        
    def record_method_comparison(self, method_name: str, scores: List[float], times: List[float]):
        """记录不同方法的比较数据
        
        Args:
            method_name: 方法名称
            scores: 得分列表
            times: 时间列表
        """
        # 创建新的比较数据文件
        filename = os.path.join(self.output_dir, f"method_comparison_{method_name}.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "score", "time"])
            for i, (score, t) in enumerate(zip(scores, times)):
                writer.writerow([i+1, score, t])
                
    def export_data(self):
        """导出所有数据为CSV格式"""
        # 优化过程数据
        opt_df = pd.DataFrame(self.optimization_data)
        opt_df.to_csv(os.path.join(self.output_dir, "optimization_process.csv"), index=False)
        
        # 评估结果数据
        eval_df = pd.DataFrame(self.evaluation_data)
        eval_df.to_csv(os.path.join(self.output_dir, "evaluation_results.csv"), index=False)
        
        # 超参数数据
        hyper_df = pd.DataFrame(self.hyperparameter_data)
        hyper_df.to_csv(os.path.join(self.output_dir, "hyperparameters.csv"), index=False)
        
        # 最佳点数据
        # 将点值列表转换为字符串
        best_points_data_copy = self.best_points_data.copy()
        best_points_data_copy["point_values"] = [str(p) for p in self.best_points_data["point_values"]]
        best_df = pd.DataFrame(best_points_data_copy)
        best_df.to_csv(os.path.join(self.output_dir, "best_points.csv"), index=False)
        
        # 返回总数据量统计
        stats = {
            "optimization_records": len(self.optimization_data["iteration"]),
            "evaluation_records": len(self.evaluation_data["call_num"]),
            "hyperparameter_records": len(self.hyperparameter_data["intrinsic_dim"]),
            "best_points_records": len(self.best_points_data["iteration"])
        }
        
        return stats
    
    def plot_optimization_progress(self):
        """保存优化进度数据，而不是直接绘图"""
        if not self.optimization_data["iteration"]:
            print("没有足够的优化数据来保存绘图数据")
            return
        
        # 准备绘图数据
        plot_data = {
            "iteration": self.optimization_data["iteration"],
            "best_score": self.optimization_data["best_score"],
            "avg_score": self.optimization_data["avg_score"],
            "computation_time": self.optimization_data["computation_time"]
        }
        
        # 保存绘图数据
        data_path = os.path.join(self.output_dir, "optimization_progress_plot_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, ensure_ascii=False, indent=2)
        
        print(f"优化进度绘图数据已保存到 {data_path}")
        
    def plot_parameter_influence(self):
        """保存参数影响数据，而不是直接绘图"""
        if len(self.hyperparameter_data["intrinsic_dim"]) <= 1:
            print("没有足够的超参数数据来保存绘图数据")
            return
        
        # 准备绘图数据
        plot_data = {
            "intrinsic_dim": self.hyperparameter_data["intrinsic_dim"],
            "batch_size": self.hyperparameter_data["batch_size"],
            "final_score": self.hyperparameter_data["final_score"],
            "total_time": self.hyperparameter_data["total_time"]
        }
        
        # 保存绘图数据
        data_path = os.path.join(self.output_dir, "parameter_influence_plot_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, ensure_ascii=False, indent=2)
        
        print(f"参数影响绘图数据已保存到 {data_path}")

    def save_bayesian_points(self, X_train, Y_train, coupled=False):
        """保存贝叶斯优化点和得分
        
        Args:
            X_train: 贝叶斯优化点(torch.Tensor或numpy.ndarray)
            Y_train: 贝叶斯优化点对应的得分(torch.Tensor或numpy.ndarray)
            coupled: 是否使用耦合核(如果是，Y_train可能是二维的)
        """
        # 确保数据是可序列化的列表格式
        if hasattr(X_train, 'tolist'):
            X_points = X_train.tolist()
        else:
            X_points = X_train
            
        # 保存贝叶斯点
        points_path = os.path.join(self.output_dir, "bayesian_points.json")
        with open(points_path, 'w', encoding='utf-8') as f:
            json.dump(X_points, f, ensure_ascii=False, indent=4)
        
        # 保存分数
        if coupled:
            # 处理耦合情况下的Y_train，可能是元组(Y1_train, Y2_train)
            if isinstance(Y_train, tuple) and len(Y_train) > 1:
                # 分别处理Y1和Y2
                Y1, Y2 = Y_train

                # 确保Y1是可序列化的格式
                if hasattr(Y1, 'tolist'):
                    Y1_scores = Y1.tolist()
                else:
                    Y1_scores = Y1

                # 保存Y1
                scores1_path = os.path.join(self.output_dir, "bayesian_scores1.json")
                with open(scores1_path, 'w', encoding='utf-8') as f:
                    json.dump(Y1_scores, f, ensure_ascii=False, indent=4)
                
                # 确保Y2是可序列化的格式
                if hasattr(Y2, 'tolist'):
                    Y2_scores = Y2.tolist()
                else:
                    Y2_scores = Y2

                # 保存Y2
                scores2_path = os.path.join(self.output_dir, "bayesian_scores2.json")
                with open(scores2_path, 'w', encoding='utf-8') as f:
                    json.dump(Y2_scores, f, ensure_ascii=False, indent=4)
            else:
                # 如果只有一个Y_train
                if hasattr(Y_train, 'tolist'):
                    Y_scores = Y_train.tolist()
                else:
                    Y_scores = Y_train
                    
                scores1_path = os.path.join(self.output_dir, "bayesian_scores1.json")
                with open(scores1_path, 'w', encoding='utf-8') as f:
                    json.dump(Y_scores, f, ensure_ascii=False, indent=4)
        else:
            # 非耦合情况
            if hasattr(Y_train, 'tolist'):
                Y_scores = Y_train.tolist()
            else:
                Y_scores = Y_train
                
            scores_path = os.path.join(self.output_dir, "bayesian_scores.json")
            with open(scores_path, 'w', encoding='utf-8') as f:
                json.dump(Y_scores, f, ensure_ascii=False, indent=4)
                
        print(f"Bayesian optimization points and scores have been saved to {self.output_dir}")
        
    def load_bayesian_points(self, coupled=False):
        """加载贝叶斯优化点和得分
        
        Args:
            coupled: 是否使用耦合核
            
        Returns:
            tuple: (X_train, Y_train) 点和分数
        """
        import torch
        
        # 加载贝叶斯点
        points_path = os.path.join(self.output_dir, "bayesian_points.json")
        with open(points_path, 'r', encoding='utf-8') as f:
            X_points = json.load(f)
        
        # 转换为tensor
        X_train = torch.tensor(X_points)
        
        # 加载分数
        if coupled:
            scores1_path = os.path.join(self.output_dir, "bayesian_scores1.json")
            with open(scores1_path, 'r', encoding='utf-8') as f:
                Y_scores1 = json.load(f)
                
            scores2_path = os.path.join(self.output_dir, "bayesian_scores2.json")
            if os.path.exists(scores2_path):
                with open(scores2_path, 'r', encoding='utf-8') as f:
                    Y_scores2 = json.load(f)
                return X_train, (torch.tensor(Y_scores1), torch.tensor(Y_scores2))
            else:
                return X_train, torch.tensor(Y_scores1)
        else:
            scores_path = os.path.join(self.output_dir, "bayesian_scores.json")
            with open(scores_path, 'r', encoding='utf-8') as f:
                Y_scores = json.load(f)
            return X_train, torch.tensor(Y_scores)

    def record_optimization_details(self, iteration, candidate_points, model_state=None):
        """记录每次迭代的优化详情
        
        Args:
            iteration: 当前迭代次数
            candidate_points: 当前迭代中评估的所有候选点
            model_state: GP模型的关键状态参数(如核函数参数)
        """
        if not hasattr(self, "optimization_details"):
            self.optimization_details = []
        
        # 记录详细信息
        detail = {
            "iteration": iteration,
            "candidate_points": [p.cpu().numpy().tolist() if hasattr(p, 'cpu') else p for p in candidate_points],
            "timestamp": time.time()
        }
        
        # 如果提供了模型状态，也记录下来
        if model_state:
            detail["model_state"] = {
                "lengthscale": model_state.get("lengthscale", None),
                "noise": model_state.get("noise", None),
                "outputscale": model_state.get("outputscale", None)
            }
        
        self.optimization_details.append(detail)
        
        # 导出详情
        details_path = os.path.join(self.output_dir, f"optimization_details_iter{iteration}.json")
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
    
    def record_attack_metrics(self, call_num, template, output, target_behaviors=None):
        """记录更详细的攻击效果指标
        
        Args:
            call_num: 调用次数
            template: 使用的模板
            output: 模型输出
            target_behaviors: 目标行为检测词典，格式为{行为类型:关键词列表}
        """
        if not hasattr(self, "attack_metrics"):
            self.attack_metrics = {
                "call_num": [],
                "word_count": [],
                "char_count": [],
                "detected_behaviors": [],
                "toxicity_score": [],
                "template_length": []
            }
        
        # 基础统计
        self.attack_metrics["call_num"].append(call_num)
        self.attack_metrics["word_count"].append(len(output.split()))
        self.attack_metrics["char_count"].append(len(output))
        self.attack_metrics["template_length"].append(len(template))
        
        # 行为检测
        if target_behaviors:
            detected = []
            for behavior, keywords in target_behaviors.items():
                if any(keyword in output.lower() for keyword in keywords):
                    detected.append(behavior)
            self.attack_metrics["detected_behaviors"].append(detected)
        else:
            self.attack_metrics["detected_behaviors"].append([])
        
        # 可以添加简单的毒性评分算法
        toxic_words = ["kill", "bomb", "attack", "weapon", "harmful", "illegal"]
        toxicity = sum(1 for word in toxic_words if word in output.lower()) / len(toxic_words)
        self.attack_metrics["toxicity_score"].append(toxicity)
        
        # 导出攻击指标数据
        metrics_path = os.path.join(self.output_dir, "attack_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.attack_metrics, f, ensure_ascii=False, indent=2)

    def compare_experiments(self, experiment_dirs, metrics=None):
        """比较多个实验结果
        
        Args:
            experiment_dirs: 实验目录列表
            metrics: 要比较的指标列表，默认为None表示比较所有指标
        
        Returns:
            比较结果数据框
        """
        import pandas as pd
        
        if metrics is None:
            metrics = ["best_score", "avg_score", "computation_time"]
        
        # 收集各实验的数据
        comparison_data = {metric: [] for metric in metrics}
        comparison_data["experiment"] = []
        
        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            comparison_data["experiment"].append(exp_name)
            
            # 读取该实验的优化过程数据
            try:
                df = pd.read_csv(os.path.join(exp_dir, "optimization_process.csv"))
                # 对每个指标，提取最后一次迭代的值
                for metric in metrics:
                    if metric in df.columns:
                        comparison_data[metric].append(df[metric].iloc[-1])
                    else:
                        comparison_data[metric].append(None)
            except Exception as e:
                print(f"读取实验 {exp_name} 数据失败: {e}")
                for metric in metrics:
                    comparison_data[metric].append(None)
        
        # 转换为数据框
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        comparison_path = os.path.join(self.output_dir, "experiments_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # 保存绘图数据而不是直接绘图
        plot_data_path = os.path.join(self.output_dir, "experiments_comparison_plot_data.json")
        with open(plot_data_path, 'w', encoding='utf-8') as f:
            json.dump({
                "experiments": comparison_data["experiment"],
                "metrics": {metric: comparison_data[metric] for metric in metrics}
            }, f, ensure_ascii=False, indent=2)
        
        return comparison_df
    
    def analyze_optimization_trends(self):
        """分析优化趋势并生成统计报告"""
        if not hasattr(self, "optimization_data") or not self.optimization_data["best_score"]:
            print("没有足够的优化数据进行分析")
            return {}
        
        # 计算基本统计量
        best_scores = self.optimization_data["best_score"]
        avg_scores = self.optimization_data["avg_score"]
        times = self.optimization_data["computation_time"]
        
        analysis = {
            "总迭代次数": len(best_scores),
            "最终最佳得分": best_scores[-1],
            "最佳得分提升": best_scores[-1] - best_scores[0],
            "平均得分提升": avg_scores[-1] - avg_scores[0],
            "总计算时间(秒)": sum(times),
            "平均每轮时间(秒)": sum(times) / len(times),
            "收敛速度": self._calculate_convergence_speed(best_scores)
        }
        
        # 保存分析结果
        analysis_path = os.path.join(self.output_dir, "optimization_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        return analysis
    
    def _calculate_convergence_speed(self, scores):
        """计算收敛速度，即达到最终得分90%所需的迭代次数"""
        final_score = scores[-1]
        target_score = 0.9 * final_score
        
        for i, score in enumerate(scores):
            if score >= target_score:
                return i
        
        return len(scores)  # 如果没有达到目标

    def save_attack_success_rate_data(self):
        """保存攻击成功率数据，而不是直接绘图"""
        if not hasattr(self, "evaluation_data") or not self.evaluation_data["attack_success"]:
            print("没有足够的评估数据来保存攻击成功率数据")
            return
        
        # 计算移动窗口下的攻击成功率
        window_size = 10
        success_rates = []
        call_nums = []
        
        for i in range(0, len(self.evaluation_data["attack_success"]), window_size):
            window = self.evaluation_data["attack_success"][i:i+window_size]
            if window:
                success_rate = sum(window) / len(window)
                success_rates.append(success_rate)
                call_nums.append(self.evaluation_data["call_num"][i])
        
        # 保存绘图数据
        data = {
            "call_nums": call_nums,
            "success_rates": success_rates,
            "window_size": window_size
        }
        
        data_path = os.path.join(self.output_dir, "attack_success_rate_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"攻击成功率绘图数据已保存到 {data_path}")
    
    def save_parameter_heatmap_data(self, dim1=0, dim2=1):
        """保存参数空间热力图数据，而不是直接绘图
        
        Args:
            dim1: 第一个维度索引
            dim2: 第二个维度索引
        """
        if not hasattr(self, "best_points_data") or not self.best_points_data["point_values"]:
            print("没有足够的最佳点数据来保存热力图数据")
            return
        
        # 提取所有点的指定维度值和对应分数
        x_values = []
        y_values = []
        scores = []
        
        for point, score in zip(self.best_points_data["point_values"], self.best_points_data["score"]):
            if len(point) > max(dim1, dim2):
                x_values.append(point[dim1])
                y_values.append(point[dim2])
                scores.append(score)
        
        # 保存绘图数据
        data = {
            "x_values": x_values,
            "y_values": y_values,
            "scores": scores,
            "dim1": dim1,
            "dim2": dim2
        }
        
        data_path = os.path.join(self.output_dir, f"parameter_heatmap_dim{dim1}_dim{dim2}_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"参数热力图数据已保存到 {data_path}")
    
    def record_model_internal_state(self, iteration, layer_outputs=None, attention_maps=None):
        """记录模型内部状态，用于分析模型行为
        
        Args:
            iteration: 当前迭代次数
            layer_outputs: 各层输出
            attention_maps: 注意力图
        """
        internal_state_dir = os.path.join(self.output_dir, "model_internal_states")
        os.makedirs(internal_state_dir, exist_ok=True)
        
        # 创建该迭代的子目录
        iter_dir = os.path.join(internal_state_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 保存层输出
        if layer_outputs is not None:
            for i, output in enumerate(layer_outputs):
                if hasattr(output, 'cpu'):
                    # 保存为numpy数组
                    np.save(os.path.join(iter_dir, f"layer_{i}_output.npy"), output.cpu().numpy())
        
        # 保存注意力图
        if attention_maps is not None:
            for i, attn_map in enumerate(attention_maps):
                if hasattr(attn_map, 'cpu'):
                    np.save(os.path.join(iter_dir, f"attention_map_{i}.npy"), attn_map.cpu().numpy())

    def generate_experiment_summary(self):
        """生成实验总结报告，包含所有关键数据和图表的汇总"""
        summary = {
            "实验基本信息": {
                "开始时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                "结束时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "总耗时(小时)": (time.time() - self.start_time) / 3600,
                "实验目录": self.output_dir
            },
            "优化参数设置": {
                "内在维度": self.hyperparameter_data["intrinsic_dim"][-1] if self.hyperparameter_data["intrinsic_dim"] else None,
                "初始采样点数": self.hyperparameter_data["n_init"][-1] if self.hyperparameter_data["n_init"] else None,
                "迭代次数": self.hyperparameter_data["n_iterations"][-1] if self.hyperparameter_data["n_iterations"] else None,
                "批量大小": self.hyperparameter_data["batch_size"][-1] if self.hyperparameter_data["batch_size"] else None
            },
            "优化结果摘要": self.analyze_optimization_trends() if hasattr(self, "optimization_data") else {},
            "攻击效果摘要": {
                "总评估次数": len(self.evaluation_data["call_num"]) if hasattr(self, "evaluation_data") else 0,
                "攻击成功率": sum(self.evaluation_data["attack_success"]) / max(len(self.evaluation_data["attack_success"]), 1) if hasattr(self, "evaluation_data") and self.evaluation_data["attack_success"] else 0,
                "平均安全绕过得分": sum(self.evaluation_data["safety_evasion_score"]) / max(len(self.evaluation_data["safety_evasion_score"]), 1) if hasattr(self, "evaluation_data") and self.evaluation_data["safety_evasion_score"] else 0,
                "平均语义相似度": sum(self.evaluation_data["semantic_similarity"]) / max(len(self.evaluation_data["semantic_similarity"]), 1) if hasattr(self, "evaluation_data") and self.evaluation_data["semantic_similarity"] else 0
            },
            "最佳模板示例": {
                "迭代": self.best_points_data["iteration"][-1] if hasattr(self, "best_points_data") and self.best_points_data["iteration"] else None,
                "得分": self.best_points_data["score"][-1] if hasattr(self, "best_points_data") and self.best_points_data["score"] else None,
                "模板": self.evaluation_data["template"][-1] if hasattr(self, "evaluation_data") and self.evaluation_data["template"] else None
            }
        }
        
        # 保存总结报告
        summary_path = os.path.join(self.output_dir, "experiment_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        # 生成纯文本格式报告
        self._generate_text_report(summary)
        
        return summary

    def _generate_text_report(self, summary):
        """生成纯文本格式的实验报告，避免HTML中的中文字体问题"""
        report_content = """
实验总结报告
===================

一、实验基本信息
-------------------
开始时间: {start_time}
结束时间: {end_time}
总耗时(小时): {total_time:.2f}
实验目录: {exp_dir}

二、优化参数设置
-------------------
内在维度: {intrinsic_dim}
初始采样点数: {n_init}
迭代次数: {n_iterations}
批量大小: {batch_size}

三、优化结果摘要
-------------------
""".format(
            start_time=summary['实验基本信息']['开始时间'],
            end_time=summary['实验基本信息']['结束时间'],
            total_time=summary['实验基本信息']['总耗时(小时)'],
            exp_dir=summary['实验基本信息']['实验目录'],
            intrinsic_dim=summary['优化参数设置']['内在维度'],
            n_init=summary['优化参数设置']['初始采样点数'],
            n_iterations=summary['优化参数设置']['迭代次数'],
            batch_size=summary['优化参数设置']['批量大小']
        )
        
        # 添加优化结果
        for key, value in summary['优化结果摘要'].items():
            report_content += f"{key}: {value}\n"
        
        report_content += """
四、攻击效果摘要
-------------------
总评估次数: {total_evals}
攻击成功率: {success_rate:.4f}
平均安全绕过得分: {avg_safety:.4f}
平均语义相似度: {avg_semantic:.4f}

五、最佳模板示例
-------------------
迭代: {best_iter}
得分: {best_score}
模板: {best_template}

六、生成的绘图数据文件
-------------------
优化进度数据: optimization_progress_plot_data.json
攻击成功率数据: attack_success_rate_data.json
参数影响数据: parameter_influence_plot_data.json
""".format(
            total_evals=summary['攻击效果摘要']['总评估次数'],
            success_rate=summary['攻击效果摘要']['攻击成功率'],
            avg_safety=summary['攻击效果摘要']['平均安全绕过得分'],
            avg_semantic=summary['攻击效果摘要']['平均语义相似度'],
            best_iter=summary['最佳模板示例']['迭代'],
            best_score=summary['最佳模板示例']['得分'],
            best_template=summary['最佳模板示例']['模板']
        )
        
        # 保存文本报告
        report_path = os.path.join(self.output_dir, "experiment_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

def collect_optimization_data(iterations, best_scores, all_scores, times, ei_values=None, batch_sizes=None):
    """
    收集优化过程数据并保存为CSV
    
    Args:
        iterations: 迭代次数列表
        best_scores: 每次迭代的最佳得分列表
        all_scores: 每次迭代的所有得分列表(嵌套列表)
        times: 每次迭代的计算时间列表
        ei_values: 期望改进值列表(可选)
        batch_sizes: 批量大小列表(可选)
    """
    output_dir = "./experiment_data"
    os.makedirs(output_dir, exist_ok=True)
    
    data = {"iteration": iterations, "best_score": best_scores}
    
    # 计算平均得分
    avg_scores = [np.mean(scores) if scores else 0 for scores in all_scores]
    data["avg_score"] = avg_scores
    
    # 添加时间数据
    data["computation_time"] = times
    
    # 添加EI值数据(如果有)
    if ei_values:
        data["ei_value"] = ei_values
    
    # 添加批量大小数据(如果有)
    if batch_sizes:
        data["batch_size"] = batch_sizes
    
    # 导出CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "optimization_process.csv"), index=False)
    
    return df

def collect_evaluation_data(call_nums, safety_scores, semantic_scores, attack_successes, templates):
    """
    收集评估结果数据并保存为CSV
    
    Args:
        call_nums: 调用次数列表
        safety_scores: 安全绕过评分列表
        semantic_scores: 语义相似度分数列表
        attack_successes: 攻击是否成功列表(布尔值)
        templates: 使用的模板列表
    """
    output_dir = "./experiment_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 将布尔值转换为0/1
    attack_successes_int = [1 if success else 0 for success in attack_successes]
    
    data = {
        "call_num": call_nums,
        "safety_evasion_score": safety_scores,
        "semantic_similarity": semantic_scores,
        "attack_success": attack_successes_int,
        "template": templates
    }
    
    # 导出CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    return df

def collect_method_comparison_data(methods, scores_by_method, times_by_method):
    """
    收集不同方法比较数据并保存为CSV
    
    Args:
        methods: 方法名称列表
        scores_by_method: 每种方法的得分列表(嵌套列表)
        times_by_method: 每种方法的时间列表(嵌套列表)
    """
    output_dir = "./experiment_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并数据
    data = {"method": []}
    max_iterations = max([len(scores) for scores in scores_by_method])
    
    for i in range(max_iterations):
        data[f"score_{i+1}"] = []
        data[f"time_{i+1}"] = []
    
    for i, method in enumerate(methods):
        data["method"].append(method)
        scores = scores_by_method[i]
        times = times_by_method[i]
        
        for j in range(max_iterations):
            if j < len(scores):
                data[f"score_{j+1}"].append(scores[j])
                data[f"time_{j+1}"].append(times[j])
            else:
                data[f"score_{j+1}"].append(None)
                data[f"time_{j+1}"].append(None)
    
    # 导出CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)
    
    return df

def collect_hyperparameter_data(intrinsic_dims, n_inits, n_iterations_list, batch_sizes, final_scores, total_times):
    """
    收集超参数实验数据并保存为CSV
    
    Args:
        intrinsic_dims: 内在维度列表
        n_inits: 初始采样点数量列表
        n_iterations_list: 迭代次数列表
        batch_sizes: 批量大小列表
        final_scores: 最终得分列表
        total_times: 总计算时间列表
    """
    output_dir = "./experiment_data"
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        "intrinsic_dim": intrinsic_dims,
        "n_init": n_inits,
        "n_iterations": n_iterations_list,
        "batch_size": batch_sizes,
        "final_score": final_scores,
        "total_time": total_times
    }
    
    # 导出CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "hyperparameters.csv"), index=False)
    
    return df

def plot_optimization_progress(iterations, best_scores, avg_scores, times):
    """
    绘制优化进度图并保存
    
    Args:
        iterations: 迭代次数列表
        best_scores: 每次迭代的最佳得分列表
        avg_scores: 每次迭代的平均得分列表
        times: 每次迭代的计算时间列表
    """
    output_dir = "./experiment_data"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制最佳得分变化
    plt.subplot(1, 2, 1)
    plt.plot(iterations, best_scores, 'b-', label='Best Score')
    plt.plot(iterations, avg_scores, 'r--', label='Average Score')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Score Changes During Optimization')
    plt.legend()
    plt.grid(True)
    
    # 绘制计算时间变化
    plt.subplot(1, 2, 2)
    plt.plot(iterations, times, 'g-')
    plt.xlabel('Iterations')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Changes During Optimization')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_progress.png"))
    plt.close() 
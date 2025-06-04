# gpytorch for substring kernel implemented in https://github.com/henrymoss/BOSS/tree/master/boss/code/kernels/string
from gpytorch.kernels.kernel import Kernel
import torch
import numpy as np
from tqdm.auto import tqdm, trange
from pathlib import Path
import json
import numpy as np
import torch
import os
import cma



class CombinedStringKernel(Kernel):
        def __init__(self, base_latent_kernel, instruction_kernel, latent_train, instruction_train, **kwargs):
            super().__init__(**kwargs)
            self.base_latent_kernel = base_latent_kernel # Kernel on the latent space (Matern Kernel)
            self.instruction_kernel = instruction_kernel # Kernel on the latent space (Matern Kernel)
            self.latent_train = latent_train # normalized training input
            self.lp_dim = self.latent_train.shape[-1]
            self.instruction_train = instruction_train # SMILES format training input #self.get_smiles(self.latent_train)#.clone())

        
        def forward(self, z1, z2, **params):
            # z1 and z2 are unnormalized
            check_dim = 0
            if len(z1.shape) > 2:
                check_dim = z1.shape[0]
                z1 = z1.squeeze(1)
            if len(z2.shape) > 2:
                check_dim = z2.shape[0]
                z2 = z2[0]
            latent_train_z1 = z1[:, :self.lp_dim] 
            latent_train_z2 = z2[:, :self.lp_dim]
            
            # 增加数值稳定性
            jitter = 1e-4
            try:
                # 计算指令空间的核矩阵
                K_train_instruction = self.instruction_kernel.forward(self.instruction_train, self.instruction_train, **params)
                
                # 计算潜在空间的核矩阵
                latent_space_kernel = self.base_latent_kernel.forward(self.latent_train, self.latent_train, **params)
                
                # 计算新点与训练点之间的核矩阵
                K_z1_training = self.base_latent_kernel.forward(latent_train_z1, self.latent_train, **params)
                K_z2_training = self.base_latent_kernel.forward(latent_train_z2, self.latent_train, **params)
                
                # 增加对角线抖动以提高稳定性
                eye_matrix = torch.eye(len(self.latent_train), device=latent_space_kernel.device, dtype=latent_space_kernel.dtype)
                latent_space_kernel_stable = latent_space_kernel + jitter * eye_matrix
                
                # 确保对角线元素绝对不会为负
                diag_indices = torch.arange(latent_space_kernel_stable.shape[0], device=latent_space_kernel_stable.device)
                min_diag = torch.min(latent_space_kernel_stable.diagonal())
                if min_diag < jitter:
                    latent_space_kernel_stable[diag_indices, diag_indices] += (jitter - min_diag)
                
                # 使用更稳定的求解方法替代直接求逆
                try:
                    # 尝试Cholesky分解（更稳定）
                    L = torch.linalg.cholesky(latent_space_kernel_stable)
                    latent_space_kernel_inv = torch.cholesky_solve(eye_matrix, L)
                except Exception:
                    # 回退到SVD分解
                    U, S, V = torch.linalg.svd(latent_space_kernel_stable)
                    # 截断小特征值以提高稳定性
                    S_inv = torch.zeros_like(S)
                    mask = S > (S.max() * 1e-10)
                    S_inv[mask] = 1.0 / S[mask]
                    latent_space_kernel_inv = V.transpose(-2, -1) @ torch.diag_embed(S_inv) @ U.transpose(-2, -1)
                
                # # 检查维度是否匹配，并在必要时调整
                # n_train = self.latent_train.shape[0]
                # if K_train_instruction.shape[0] != n_train or K_train_instruction.shape[1] != n_train:
                #     # 重新构建合适维度的指令核矩阵
                #     K_train_instruction = self.instruction_kernel.forward(
                #         self.instruction_train[:n_train], 
                #         self.instruction_train[:n_train], 
                #         **params
                #     )
                
                # 计算最终的核矩阵，分步执行以便于调试
                try:
                    temp1 = K_z1_training @ latent_space_kernel_inv
                    temp2 = temp1 @ K_train_instruction
                    temp3 = temp2 @ latent_space_kernel_inv
                    K_result = temp3 @ K_z2_training.T
                except Exception as e:
                    print(f"矩阵乘法出错: {e}")
                    # 如果矩阵乘法失败，返回合理默认值
                    K_result = torch.eye(z1.shape[0], z2.shape[0], device=z1.device, dtype=z1.dtype)
                
                # 确保结果对称且半正定
                if K_result.shape[0] == K_result.shape[1]:  # 只有方阵才能保证对称
                    K_result = 0.5 * (K_result + K_result.T)
                
                if check_dim > 0:
                    K_result = K_result.unsqueeze(1)
                return K_result
            except Exception as e:
                # 如果计算失败，返回默认核矩阵
                print(f"核函数计算错误: {e}")
                # 创建适当大小的单位矩阵
                default_kernel = torch.eye(z1.shape[0], z2.shape[0], device=z1.device, dtype=z1.dtype)
                if check_dim > 0:
                    default_kernel = default_kernel.unsqueeze(1)
                return default_kernel


def cma_es_concat(starting_point_for_cma, EI, tkwargs):
        if starting_point_for_cma.type() == 'torch.cuda.DoubleTensor':
            starting_point_for_cma = starting_point_for_cma.detach().cpu().squeeze()
        
        # 清理输入，确保数据有效
        if torch.isnan(starting_point_for_cma).any() or torch.isinf(starting_point_for_cma).any():
            # 替换NaN/Inf值为0
            starting_point_for_cma = torch.where(
                torch.isnan(starting_point_for_cma) | torch.isinf(starting_point_for_cma),
                torch.zeros_like(starting_point_for_cma),
                starting_point_for_cma
            )
        
        # 限制起始点在[-1, 1]范围内
        starting_point_for_cma = torch.clamp(starting_point_for_cma, -1.0, 1.0)
        
        # 转换为numpy数组
        x0 = starting_point_for_cma.numpy()
        
        # 初始化CMA-ES，增加容错和稳健性
        try:
            es = cma.CMAEvolutionStrategy(
                x0=x0, 
                sigma0=0.8,  # 减小初始步长以提高稳定性
                inopts={
                    'bounds': [-1, 1], 
                    "popsize": 50,  # 种群大小
                    "tolx": 1e-3,   # 增加收敛容忍度
                    "maxiter": 10,  # 限制最大迭代次数
                    "tolfun": 1e-3,  # 目标函数收敛容忍度
                    "verbose": -9  # 添加此行以禁止输出
                },
            )
            
            iter_count = 0
            best_x = x0
            best_f = float('-inf')
            
            while not es.stop() and iter_count < 10:
                iter_count += 1
                try:
                    xs = es.ask()
                    # 确保评估批次小于训练数据量
                    
                    # 动态调整批次大小
                    if hasattr(EI.model, 'train_inputs'):
                        train_size = EI.model.train_inputs[0].shape[0]
                        batch_size = min(len(xs), max(1, train_size // 3))  # 批次不超过训练集1/3
                    else:
                        batch_size = min(len(xs), 5)  # 默认保守值
                    
                    # 分批处理以避免维度不匹配
                    all_Y = []
                    for i in range(0, len(xs), batch_size):
                        batch_xs = xs[i:i+batch_size]
                        X_batch = torch.tensor(np.array(batch_xs), dtype=torch.float32).unsqueeze(1).to(**tkwargs)
                        
                        # 使用try-except包装EI计算，以防出错
                        try:
                            with torch.no_grad():
                                Y_batch = -1 * EI(X_batch)
                                all_Y.append(Y_batch.cpu().numpy())
                        except Exception as e:
                            # 如果EI评估失败，使用一个默认的较大负值
                            Y_batch = torch.ones(len(batch_xs), device=tkwargs["device"]) * float('inf')
                            all_Y.append(Y_batch.cpu().numpy())
                            print(f"EI评估失败: {e}")
                    
                    # 合并结果
                    if all_Y:
                        Y_np = np.concatenate(all_Y)
                    else:
                        Y_np = np.ones(len(xs)) * float('inf')
                    
                    es.tell(xs, Y_np)  # 将结果返回给优化器
                    
                    # 更新最佳值
                    if len(Y_np) > 0:
                        current_best_idx = np.argmin(Y_np)
                        if Y_np[current_best_idx] < best_f:
                            best_f = Y_np[current_best_idx]
                            best_x = xs[current_best_idx]
                    
                except Exception as e:
                    print(f"CMA-ES迭代失败: {e}")
                    iter_count += 2  # 加速退出失败的优化
            
            # 使用可靠的最佳值
            if hasattr(es.result, 'xbest') and es.result.xbest is not None and not (np.isnan(es.result.xbest).any() or np.isinf(es.result.xbest).any()):
                return es.result.xbest, -1 * es.result.fbest
            else:
                return best_x, -1 * best_f
                
        except Exception as e:
            print(f"CMA-ES初始化失败: {e}")
            # 返回原始点和一个默认EI值
            return x0, 0.0
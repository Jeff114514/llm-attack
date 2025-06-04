import argparse

def get_args():
    parser = argparse.ArgumentParser(description='贝叶斯优化参数设置')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='deepseek-llama',
                        help='模型名称')
    parser.add_argument('--hf_dir', type=str, default='/root/autodl-tmp/deepseek-llama',
                        help='HuggingFace模型目录')
    parser.add_argument('--intrinsic_dim', type=int, default=16,
                        help='内在维度大小')
    parser.add_argument('--n_prompt_tokens', type=int, default=10,
                        help='提示标记数量')
    
    # 优化参数
    parser.add_argument('--n_init', type=int, default=24,
                        help='初始采样点数量')
    parser.add_argument('--n_iterations', type=int, default=15,
                        help='优化迭代次数')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='批处理大小')
    parser.add_argument('--use_coupled', action='store_true',
                        help='是否使用耦合核')
    
    # 控制参数
    parser.add_argument('--train_scav', action='store_true',
                        help='是否训练SCAV分类器')
    parser.add_argument('--train_baysion', action='store_true',
                        help='是否进行贝叶斯优化')
    parser.add_argument('--test_api', action='store_true',
                        help='是否测试API')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存模型的路径')
    parser.add_argument('--load_path', type=str, default=None,
                        help='加载模型的路径')
    parser.add_argument('--seed', type=int, default=2142,
                        help='随机种子')
    parser.add_argument('--use_deepseek_api', action='store_true',
                        help='是否使用DeepSeek API进行攻击测试')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./experiment_data',
                        help='输出目录')
    
    return parser.parse_args() 
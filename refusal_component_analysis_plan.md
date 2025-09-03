# Refusal Component Analysis Plan

## 目标
计算与refusal direction平行的分量大小，量化模型在不同数据集上的'拒绝程度'。

## 核心思路

### 1. 技术流程

1. **加载已提取的refusal direction**: 
   - 从 `/pipeline/runs/Qwen3-14B/direction.pt` 加载预训练好的拒绝方向
   - 从 `/pipeline/runs/Qwen3-14B/direction_metadata.json` 获取layer信息(layer=25)

2. **数据准备**: 
   - 加载harmful数据集(3a)和harmless数据集(4a) 
   - 使用与run_pipeline.py相同的采样和过滤逻辑确保一致性

3. **两次前向传播**:
   - **第一次**: 使用Qwen3 thinking mode template (`QWEN3_CHAT_TEMPLATE_THINKING`)生成一个新token（预期是`<think>`）
   - **第二次**: 将新生成的token添加到输入序列中，使用forward hooks收集每一层的activations

4. **计算平行分量**: 
   ```python
   component_size = |activation · direction_normalized|
   ```

5. **存储和可视化**: 
   - 对比harmful vs harmless数据集上的分量大小分布
   - 生成多种可视化图表分析结果

### 2. 关键实现细节

#### 模板使用策略
- **第一次生成**: 使用 `QWEN3_CHAT_TEMPLATE_THINKING` (thinking mode)
- **第二次前向传播**: 在第一次生成token的基础上继续，提取新token位置的activations

#### Hook机制设计
```python
def get_parallel_component_hook(direction, results_cache, layer_idx, sample_idx):
    def hook_fn(module, input):
        activation = input[0]  # shape: [batch, seq, d_model]  
        direction_norm = direction / (direction.norm() + 1e-8)
        # 取最后一个token位置的activation
        last_token_activation = activation[:, -1, :]  # [batch, d_model]
        parallel_size = torch.abs(last_token_activation @ direction_norm)
        results_cache[sample_idx][layer_idx] = parallel_size.cpu()
    return hook_fn
```

#### 数据结构设计
```python
results = {
    "harmful": {
        "instructions": List[str],
        "parallel_components": torch.Tensor,  # [n_samples, n_layers]
        "generated_tokens": List[str],
        "token_ids": List[int]
    },
    "harmless": {
        "instructions": List[str], 
        "parallel_components": torch.Tensor,  # [n_samples, n_layers]
        "generated_tokens": List[str],
        "token_ids": List[int]
    }
}
```

### 3. 主要挑战与解决方案

#### 挑战1: Token位置确定
- **问题**: 需要确定在第二次前向传播时应该提取哪个position的activation
- **解决**: 提取新生成token位置的activation（sequence的最后一个位置）

#### 挑战2: 内存优化  
- **问题**: 大量activation数据可能导致OOM
- **解决**: 
  - 使用batch处理分批计算
  - 即时计算平行分量而不存储完整activations
  - 使用CPU存储结果释放GPU内存

#### 挑战3: 数据一致性
- **问题**: 确保使用与原pipeline相同的数据
- **解决**: 
  - 复用run_pipeline.py中的数据加载函数
  - 使用相同的random seed和采样参数

#### 挑战4: 生成验证
- **问题**: 验证第一次生成的token确实是预期的`<think>`
- **解决**: 
  - 记录生成的token并验证
  - 添加断言确保生成正确性

### 4. 文件架构

#### 4.1 主计算文件 - calculate_refusal_components.py
```python
def load_refusal_direction(direction_path, metadata_path):
    """加载refusal direction和元数据"""
    
def setup_model_and_data(model_path, cfg):
    """设置模型和数据集"""
    
def generate_first_token(model_base, instructions, batch_size=8):
    """第一次前向传播生成新token"""
    
def collect_activations_with_hooks(model_base, instructions, generated_tokens, direction, batch_size=8):
    """第二次前向传播收集activations并计算平行分量"""
    
def calculate_parallel_components(activations, direction):
    """计算与refusal direction的平行分量大小"""
    
def save_results(results, output_path):
    """保存计算结果到JSON和PyTorch格式"""
    
def main():
    """主函数：执行完整计算流程"""
```

#### 4.2 可视化文件 - visualize_refusal_components.py
```python
def load_results(results_path):
    """加载计算结果数据"""
    
def create_layer_comparison_plot(results, output_path):
    """生成层级对比图"""
    
def create_distribution_comparison_plot(results, output_path):
    """生成分布对比图（箱线图/小提琴图）"""
    
def create_heatmaps(results, output_dir):
    """生成harmful和harmless的热力图"""
    
def create_statistical_analysis_plot(results, output_path):
    """生成统计分析图"""
    
def generate_analysis_report(results, output_path):
    """生成文本分析报告"""
    
def main():
    """主函数：生成所有可视化图表"""
```

### 5. 可视化方案

#### 5.1 层级对比图
- **图表类型**: 折线图
- **内容**: 每一层上harmful vs harmless的平均平行分量大小
- **目的**: 显示不同层级的拒绝信号强度差异

#### 5.2 分布对比图  
- **图表类型**: 箱线图/小提琴图
- **内容**: 每个数据集在所有层级上的分量分布
- **目的**: 显示分布差异和统计显著性

#### 5.3 热力图
- **图表类型**: 2D热力图
- **内容**: 样本×层级的分量大小矩阵
- **目的**: 可视化个体样本的层级响应模式

#### 5.4 统计分析图
- **图表类型**: 柱状图
- **内容**: 各层级上harmful vs harmless的统计检验结果
- **目的**: 量化显著性差异

### 6. 预期输出文件

```
results/
├── refusal_components.json          # 原始计算结果
├── refusal_components.pt            # PyTorch tensor格式结果  
├── visualizations/
│   ├── layer_comparison.png         # 层级对比图
│   ├── distribution_comparison.png  # 分布对比图
│   ├── heatmap_harmful.png         # harmful热力图
│   ├── heatmap_harmless.png        # harmless热力图
│   └── statistical_analysis.png    # 统计分析图
└── analysis_report.txt              # 文本分析报告
```

### 7. 使用方法

#### 7.1 计算平行分量
```bash
python calculate_refusal_components.py \
    --model_path /path/to/Qwen3-14B \
    --direction_path /pipeline/runs/Qwen3-14B/direction.pt \
    --metadata_path /pipeline/runs/Qwen3-14B/direction_metadata.json \
    --output_dir ./results \
    --batch_size 8 \
    --n_samples 100
```

#### 7.2 生成可视化
```bash
python visualize_refusal_components.py \
    --results_path ./results/refusal_components.json \
    --output_dir ./results/visualizations
```

### 8. 验证方式

1. **生成验证**: 检查第一次生成的token是否为`<think>`
2. **数值验证**: 确保平行分量计算结果在合理范围内
3. **一致性验证**: 对比不同batch的结果保证稳定性
4. **可视化验证**: 检查图表是否符合预期的harmful > harmless模式
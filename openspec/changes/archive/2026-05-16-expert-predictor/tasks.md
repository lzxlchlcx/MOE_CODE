## 1. 创建 ExpertPredictor 模块

- [x] 1.1 在 40-myself/src/ 下创建 expert_predictor.py，定义 ExpertPredictor 抽象基类
- [x] 1.2 实现 GatePredictor（使用下一层 gate 网络预测）

## 2. 集成到 mDeepSeek

- [x] 2.1 在 deepseek.py 中导入 ExpertPredictor 和 GatePredictor
- [x] 2.2 在 __init__ 中初始化预测器
- [x] 2.3 修改 mixtral_forward，在 MoE 层循环中将预测与 attention 并行执行
- [x] 2.4 将预测结果存储为实例属性，供调度策略使用

## 3. 测试验证

- [x] 3.1 语法检查通过
- [x] 3.2 验证预测器输出格式正确

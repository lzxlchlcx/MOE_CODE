## 1. 准备工作

- [x] 1.1 添加 concurrent.futures 导入到 deepseek.py

## 2. 核心实现

- [x] 2.1 实现 GPU 专家执行函数 _execute_gpu_experts，返回结果张量
- [x] 2.2 实现 CPU 专家执行函数 _execute_cpu_experts，返回结果张量
- [x] 2.3 在 mixtral_forward 中使用 ThreadPoolExecutor 并行执行上述两个函数
- [x] 2.4 等待两个线程完成后，将结果在GPU上合并到 inps_after_experts

## 3. 测试验证

- [x] 3.1 运行相同输入，验证并行版本和串行版本的输出数值一致
- [x] 3.2 运行简单推理，检查没有数据竞争或死锁问题

## Context

当前 deepseek.py mixtral_forward 的 444-467 行中，GPU 专家和 CPU 专家是顺序执行。执行 GPU 专家 → 然后执行 CPU 专家。这种串行方式无法利用 CPU 和 GPU 的并行能力，造成了计算资源的浪费，增加了推理延迟。

## Goals / Non-Goals

**Goals:**
- 实现 GPU 和 CPU 专家并行执行，降低推理延迟
- 保持现有功能和输出一致性（数值一致）
- 保持代码可维护性和可读性

**Non-Goals:**
- 不修改专家调度策略
- 不引入新的依赖库（使用 Python 标准库）
- 不改变 generate 等外部接口

## Decisions

### Decision 1: 使用 concurrent.futures.ThreadPoolExecutor 实现并行

**选择:** 使用 Python 标准库 concurrent.futures.ThreadPoolExecutor，尽管存在 GIL。

**替代方案:**
- threading 原生 threading.Thread + queue
- multiprocessing
- 保持串行执行不做并行

**理由:** 
- PyTorch 的 GPU 操作会释放 GIL，所以 GPU 专家线程不会阻塞主线程
- CPU 专家的 PyTorch CPU 操作也会释放 GIL（大部分算子）
- multiprocessing 进程间数据复制开销更大，对于此场景得不偿失

### Decision 2: 每个专家执行逻辑拆分为独立函数

**选择:** 将 GPU 专家执行和 CPU 专家执行分别封装为独立的函数，接收共享的变量，返回结果。

**理由:** 便于并行执行，代码清晰，容易测试。

### Decision 3: 使用锁保护 inps_after_experts 的 index_add_？

**选择:** 每个线程使用独立的临时张量存储结果，最后在主线程合并。

**理由:** PyTorch 的 index_add_ 是原子操作？为了安全，我们使用线程本地变量 + 主线程合并的方式。

## Risks / Trade-offs

- **多线程 GIL 问题 → PyTorch 操作会释放 GIL，GPU/CPU 专家执行都不会长时间占用 GIL，并行仍有收益
- **CPU 专家结果回传 GPU 的开销 → CPU 计算完后，结果张量需要从 CPU 传回到 GPU，与 GPU 专家结果相加。这个传输开销可能抵消并行带来的收益。但考虑到专家结果通常较小（和 token 数量成正比），且 GPU 专家执行时间较长，总体应该仍有收益。
- **CPU 专家执行期间可能仍有 GIL 竞争 → 这是已知问题，需要通过性能测试验证实际收益
- **实现复杂度增加 → 需要仔细处理线程同步和结果合并

# PDScope → opensource 重构实验报告

## 对比对象
- `PDScope/deepseek.py`: 论文原始原型（单文件 1516 行）
- `opensource/deepseek.py` + `scheduler.py` + `config.py`: P0-P5 重构后（总计 ~2721 行）

## 总体结论

**opensource 版本远比 PDScope 更接近论文 Algorithm 1。** PDScope 是早期原型，只有基础 ondemand 逻辑，缺少算法核心调度架构，预取用 `time.sleep()` 模拟，且存在 7 个严重 bug。经过 P0-P5 六阶段重构，opensource 实现了完整的 Decode 四模式调度 + Prefill 三步调度法，修正了所有已知 bug。

---

## 一、P0: 无效代码清理（-70 行）

### 删除内容

| 类型 | PDScope 中的问题 | opensource 处理 |
|------|-----------------|----------------|
| 重复导入 | `import threading` 出现两次 | 删除重复 |
| 重复 raise | `raise` 后面又跟一个 `raise` | 删除多余的 |
| 未使用属性 | `expert_loc_now`、`expert_weight_accumulator`、`latency_cpu`、`latency_gpu` | 删除 |
| 只写不读 | `layer_time_details`、`layer_time_accumulator_details` | 删除 |
| 未初始化方法 | `is_expert_loading()` 引用未定义的 `self.expert_loading_status` | 删除 |
| 调试残留 | `print("oo")`、`return` 后死代码 | 删除 |
| 被覆盖函数 | 第一版 `process_experts_in_gpu()`（多线程版，被后面的串行版覆盖） | 删除 |
| 硬编码 | `n_shared_experts = 2`、`torch.set_num_threads(16)` | 改为从 config 读取 |

---

## 二、P1: Multi-stream 并行传输

### PDScope 原实现

```python
# PDScope _async_ondemand (line 223-243): 串行拷贝
for name in ['gate_proj', 'up_proj', 'down_proj']:
    dst = getattr(target_placeholder, name).weight.data
    src = getattr(expert, name).weight.data
    dst.copy_(src)  # 串行，无 CUDA Stream

# PDScope _async_load_expert (line 247-294): 同样串行
for name in ['gate_proj', 'up_proj', 'down_proj']:
    dst.copy_(src)  # 同上
```

### opensource 重构后

```python
# 合并为一个函数，3 个 CUDA Stream 并行传输
self._transfer_streams = [torch.cuda.Stream() for _ in range(3)]

for stream, name in zip(self._transfer_streams, ["gate_proj", "up_proj", "down_proj"]):
    with torch.cuda.stream(stream):
        dst.copy_(src, non_blocking=True)
for s in self._transfer_streams:
    torch.cuda.synchronize(s)
```

### 关键改进

| 维度 | PDScope | opensource |
|------|---------|-----------|
| 传输方式 | 串行 `copy_()` × 3 | 3 CUDA Stream 并行 + `non_blocking=True` |
| 函数数量 | 2 个（`_async_ondemand` + `_async_load_expert`） | 1 个（`_async_load_expert` 统一接口） |
| Ondemand 路径 | 单独函数，不支持外部传入 placeholder | `target_placeholder` 参数可选，prefetch/ondemand 共用 |
| Pin Memory | 每次调用重复 pin | 同样每次 pin（保持一致） |

理论加速：3 个矩阵并行拷贝可将传输时间从 ~3e 缩短到 ~e（受 PCIe 带宽上限约束，实际约 1.5-2x）。

---

## 三、P2: 统一占位专家管理

### PDScope 原实现

```python
# 6 个独立变量，命名不一致
self.expert_placeholder = ...    → self.expert_placeholder11_inused
self.expert_placeholder2 = ...   → self.expert_placeholder22_inused
self.expert_placeholder3 = ...   → self.expert_placeholder33_inused
self.expert_placeholder4 = ...   → self.expert_placeholder44_inused
self.expert_placeholder5 = ...   → self.expert_placeholder5_inused
self.expert_placeholder6 = ...   → self.expert_placeholder6_inused

# 分配：if/elif 链（line 1088-1112）
if not self.expert_placeholder5_inused:
    placeholder = self.expert_placeholder5
    ...
elif not self.expert_placeholder6_inused:
    ...
elif not self.expert_placeholder33_inused:  # ← 注意：复用了 3 号
    placeholder = self.expert_placeholder3
    ...

# 释放：只覆盖 4 个（line 296-306），placeholder 5/6 永远不释放
for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                          'expert_placeholder3', 'expert_placeholder4']:
    ...
```

### opensource 重构后

```python
# 3 个数组统一管理
self._placeholders = [...]  # 6 个占位专家对象
self._ph_in_use = [False] * 6
self._ph_mapping = [None] * 6  # (layer, expert) 映射

def _get_available_placeholder(self):
    for i, in_use in enumerate(self._ph_in_use):
        if not in_use:
            self._ph_in_use[i] = True
            return i, self._placeholders[i]
    return None, None

def _release_placeholder_by_index(self, idx):
    # 精确释放指定索引，清理双向映射
    ...

def release_placeholder(self, layer_idx, expert_id):
    # 遍历所有 6 个，不再遗漏
    for i in range(len(self._placeholders)):
        stored = self._ph_mapping[i]
        if stored and (stored[0] < layer_idx or ...):
            self._release_placeholder_by_index(i)
```

### 消除的问题

| PDScope 问题 | opensource 修复 |
|-------------|----------------|
| 6 个变量 + 6 个 flag（12 个属性） | 3 个数组（3 个属性） |
| `placeholder5_inused` vs `expert_placeholder5_inused` 命名不一致 | 统一 `_ph_in_use[5]` |
| ondemand 分配时 `expert_placeholder33_inused` 复用了 prefetch 的 3 号 | 统一池分配，无冲突 |
| `release_placeholder()` 只释放 4 个，5/6 永不释放 | 遍历全部 6 个 |
| `process_experts_remaining` ~75 行手动 if-elif + 线程管理 | ~35 行循环分配 |

---

## 四、P3: Decode 负载均衡调度

### PDScope 原实现

PDScope **没有独立的 Decode 调度**。Decode 和 Prefill 使用同一个 TG vs TC 循环（line 780-824）：

```python
e = 1.39   # 硬编码
tg = 0.95  # 硬编码
for i in range(n-1):
    TG = (1 + i) * e + tg       # 累积 GPU 成本
    TC = TC - cpu_time_table[...]  # 递减 CPU 成本
    if self.is_decode:
        if TG < TC:
            if token_count > 1:  # ← decode 特有：只对多 token 专家 ondemand
                ondemand_experts.append(expert_id)
            ...
```

### opensource: `decide_decode()` + `_compute_optimal_gpu_count()`

```python
def decide_decode(self, r_cur, r_next, k):
    n_g = self._compute_optimal_gpu_count(k)  # n_g^ρ = argmin max(n_g·tg, (k-n_g)·tc)
    cur_below = r_cur < n_g
    next_below = r_next < n_g
    # 四模式分支
    if not cur_below and not next_below: return "C", ...
    elif cur_below and not next_below:    return "A", ...
    elif not cur_below and next_below:    return "B", ...
    else:                                 return "default", ...
```

### 核心差异

| 维度 | PDScope | opensource |
|------|---------|-----------|
| 调度粒度 | 逐个专家贪心（TG vs TC） | 全局配额（$n_g^\rho$）+ 四模式带宽分配 |
| 跨层感知 | 不考虑下一层 | `r_next` = 下层 GPU 驻留数，影响模式选择 |
| 负载均衡目标 | 无明确目标 | $\min \max(n_g \cdot t_g, (k-n_g) \cdot t_c)$ |
| Ondemand 策略 | `token_count > 1` 时才搬运 | 模式 A 全力搬运，模式 default 按预算分配 |
| Prefetch 策略 | decode 用 TG vs TC 循环决定 | 模式 B 全力预取下层，模式 default 按预算分配 |
| Offload 策略 | 无 | 模式 B/C 卸载多余占位专家，释放带宽 |
| 参数来源 | 硬编码 e=1.39, tg=0.95 | `config.e`, `config.tg`, `config.get_tc()` |

---

## 五、P4: Prefill 三步调度法

### PDScope 原实现

PDScope 的 Prefill 调度就是同一段 TG vs TC 循环的 `else` 分支（line 817-824）：

```python
if TG < TC + cpu_time_table[...]:  # ← 比 decode 多加回当前专家的 CPU 时间
    ondemand_experts.append(expert_id)
else:
    break
```

这只是一个**单层贪心**策略，没有跨层视野，没有预取机制。

### opensource: `decide_prefill_schedule()` 三步法

#### Step 1: 全局排序 (scheduler.py)

```
合并 cur + next 两层非驻留专家 → 按 token 升序排列
从最小 token 开始扫描:
    T_G_cum += t_io (增量累积)
    T_C_all -= cpu_time_i (递减)
    当 T_G_cum + t_g < T_C_all → break，右侧大 token 专家进入 L_global
```

**与 PDScope 的本质区别**：PDScope 只看当前层，opensource 合并两层专家做全局权衡。

#### Step 2: 局部重排 (scheduler.py)

```
L_global ∩ E_cur → 只保留当前层专家
按 token 升序扫描 ondemand 边界 i':
    T_G_local_cum += t_io
    当 T_G_local_cum < T_C_local → 该专家加入 ondemand
```

**论文核心机制**：交集筛选 `L_global ∩ E_cur` 确保当前层不被下一层"饿死"（即使下一层有更大 token 的专家）。

#### Step 3: 置信度预取 (scheduler.py)

```
T_gap = T_C_final - T_G_final (CPU bubble)
f_real = T_gap / t_io
逐专家遍历 L_global_next:
    R_i = max(0.5, r_hit - i*0.1)  # 置信度随排序递减
    ξ = (2·R_i - 1)·t_io           # 前 f 个专家
    ξ = R_i·frac·t_io - (1-R_i)·(1-frac)·t_io  # 边际专家
    ξ ≤ 0 → break (自然实现 f-1 回退)
```

**PDScope 完全没有这三步**。其预取逻辑（line 832-883）是基于硬编码 token 阈值的简单判断：
```python
if token_count >= 3 and not self.is_expert_in_gpu_now(...):
    self.hot_experts[i_layer + 1].append(expert_id)
```

---

## 六、P5: P4 修正 + Offload 执行 + 命中率统计

### 修正 1: Step 1/2 改为增量式

P4 初版使用固定公式 `T_G = α + (n_total - i)·t_io + t_g`，P5 修正为与 Algorithm 1 一致的增量累积式 `T_G_cum += t_io`。删除了论文未定义的 `alpha` 项。

### 修正 2: Step 3 添加 f-1 回退

P4 初版只判断小数部分是否向上取整，P5 改为逐专家 ξ + break，当 ξ ≤ 0 时自然回退（等效于论文的 f-1 fallback）。

### 修正 3: Decode offload 实际执行

新增 Section 9.6：Mode B/C 时将 `offload_count` 个占位专家降级到 CPU，释放占位槽供下层使用。

### 修正 4: 命中率三分法

| PDScope | opensource |
|---------|-----------|
| 只有 `cnt_expert_hit / cnt_expert_all` 一个命中率 | 三个独立指标 |
| 热表命中 = 唯一指标 | `hot_hit_rate`: 热表常驻命中（expert_loc=1） |
| 预取成功被掩盖在"未命中"中 | `prefetch_hit_rate`: 预取/占位专家命中 |
| — | `gpu_available_rate`: 总 GPU 可用率 = hot + prefetch |

---

## 七、PDScope 严重 Bug（opensource 已全部修复）

| # | Bug | 位置 | 影响 | opensource 修复 |
|---|-----|------|------|----------------|
| 1 | placeholder 硬编码专家 `run_expert_at_gpu(12, 1, ...)` | line 1024 | 推理结果完全错误 | P2: 用 `placeholder(expert_input)` 替代 |
| 2 | loading 硬编码专家 `run_expert_at_gpu(2, 3, ...)` | line 1049 | 推理结果完全错误 | P2: 同上 |
| 3 | 预取用 `time.sleep(0.0014)` 模拟 | line 1266-1268 | 预取从未实际传输权重 | P1: 真正的 `copy_(non_blocking=True)` |
| 4 | ondemand 线程内立即 join | line 1119-1127 | 并行退化为串行 | P2: 统一分配，去掉不必要的线程 |
| 5 | `expert_placeholder_inused` 未定义 | line 56/261 | 预取时 AttributeError | P2: `_ph_in_use[6]` 数组统一管理 |
| 6 | 非最后层专家输出为零（结构 bug） | if/else 嵌套 | cpu_offload=1 时推理完全错误 | P3: 专家处理从 else 块移出 |
| 7 | `release_placeholder` 只释放 4/6 个 | line 296-306 | 占位 5/6 永远不释放，显存泄漏 | P2: 遍历全部 6 个 |

---

## 八、代码量统计

| 文件 | PDScope | opensource (P0-P5 后) | 变化 |
|------|---------|----------------------|------|
| deepseek.py | 1516 行（单文件） | 1925 行 | +409 行（含调度集成、offload、命中率重构） |
| scheduler.py | 不存在 | 246 行（新增） | +246 行 |
| config.py | 不存在 | ~300 行（新增） | +300 行 |
| logger.py | 不存在 | ~250 行（新增） | +250 行 |
| **总计** | **1516 行** | **~2721 行** | **+1205 行 (+79%)** |

deepseek.py 增长主要来自：(1) 专家分类逻辑按四模式重写；(2) offload 执行代码；(3) 三项命中率统计；(4) 更完善的日志和错误处理。

---

## 九、论文 Algorithm 1 最终一致性

| Algorithm 1 特性 | PDScope | opensource (P5 后) | 状态 |
|-----------------|---------|-------------------|------|
| Decode 四模式 (A/B/C/default) | ❌ | ✅ `decide_decode()` | 完全实现 |
| $n_g^\rho$ 最优配额 floor/ceil | ❌ | ✅ `_compute_optimal_gpu_count()` | 完全实现 |
| Prefill Step 1: 全局排序 + 增量 TG | ❌ | ✅ 增量 T_G_cum 累积 | 完全实现 |
| Prefill Step 2: L_global ∩ E_cur | ❌ | ✅ 交集筛选当前层 | 完全实现 |
| Prefill Step 3: T_gap + ξ + f-1 回退 | ❌ | ✅ 逐专家 ξ + break | 完全实现 |
| R_hit 衰减 | ❌ | ✅ `max(0.5, r_hit - i*0.1)` | 完全实现 |
| Offload (L_un) 实际执行 | ❌ | ✅ Section 9.6 降级到 CPU | 完全实现 |
| 真实异步权重传输 | ❌ (sleep 模拟) | ✅ CUDA Stream + non_blocking | 完全实现 |
| Multi-stream 并行传输 | ❌ | ✅ 3 CUDA Stream gate/up/down | 完全实现 |
| CPU 时间表参数化 | 硬编码 | ✅ config + microbench | 完全实现 |

### 残余偏差（低优先级）

| # | 偏差 | 说明 |
|---|------|------|
| 1 | `max_ondemand=4` 硬编码 | 论文未设上限，实际受 placeholder 数量约束 |
| 2 | `r_hit=0.7` 初始值 | 论文要求动态计算，后续接入 PhasePer 预测器后可调整 |
| 3 | `cpu_time_table` 前 6 项为估算值 | microbench 平滑算法导致，需实测校准 |

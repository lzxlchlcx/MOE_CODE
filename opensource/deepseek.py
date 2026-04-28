"""
DeepSeek-V2 MoE 推理框架

功能: 实现DeepSeek-V2模型的GPU/CPU混合推理框架
特点:
    - 支持MoE(Mixture of Experts)稀疏路由
    - GPU/CPU混合专家调度
    - 热点专家预取
    - On-Demand动态加载

架构:
    1. 模型加载与初始化
    2. 专家位置管理 (GPU/CPU)
    3. MoE路由机制
    4. GPU-CPU并行计算
    5. 性能统计与优化
"""

import copy
import threading
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.nn.utils.rnn import pad_sequence
import transformers

from config import ModelConfig, load_cpu_time_table_from_file
from logger import BenchmarkLogger
from scheduler import ExpertScheduler


class FiddlerDeepSeekV2:
    """
    DeepSeek-V2 MoE 推理主类

    属性:
        model: 预训练的DeepSeek-V2模型
        n_expert: 路由专家数量 (64)
        n_shared_experts: 共享专家数量 (2)
        n_layer: Transformer层数 (26)
        expert_loc: 专家位置矩阵 [n_layer, n_expert], 0=CPU, 1=GPU
        placeholder: 占位专家，用于异步加载
    """

    def __init__(self, args_or_config):
        """
        初始化DeepSeek-V2推理框架

        参数:
            args_or_config: 命令行args对象 或 ModelConfig 对象
        """
        # ========== Config 处理 ==========
        if isinstance(args_or_config, ModelConfig):
            self.config = args_or_config
        else:
            self.config = ModelConfig.from_args(args_or_config)

        # ========== 基础配置 ==========
        self.dtype = torch.bfloat16  # 使用BF16精度减少显存
        self.dev = torch.device("cuda:0")  # GPU设备
        self.hot_experts = {}  # 热点专家缓存

        # ========== Logger ==========
        self.logger = BenchmarkLogger(self.config)

        # ========== Scheduler ==========
        self.scheduler = ExpertScheduler(self.config)
        self.scheduler.tc = self.config.get_tc()
        # 加载CPU时间表
        if not self.config.cpu_time_table and os.path.exists(
            self.config.cpu_time_table_file
        ):
            cpu_table = load_cpu_time_table_from_file(self.config.cpu_time_table_file)
            self.config.cpu_time_table = cpu_table
            self.scheduler.set_cpu_time_table(cpu_table)

        # ========== 模型加载 ==========
        # 加载预训练的DeepSeek-V2模型
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self.dtype,
            use_cache=True,  # 启用KV缓存加速推理
            trust_remote_code=True,
        )

        # Monkey-patch DynamicCache for compatibility with cached modeling file
        if not hasattr(transformers.cache_utils.DynamicCache, "get_usable_length"):
            transformers.cache_utils.DynamicCache.get_usable_length = (
                lambda self, seq_len, layer_idx: self.get_seq_length(layer_idx)
            )

        # ========== 批处理配置 ==========
        self.batch_size = self.config.batch_size
        self.cache = self.config.cache

        # ========== 模型组件提取 ==========
        self.lm_head = self.model.lm_head  # 输出层
        # 只保留model部分(去除lm_head)
        self.model = self.model.model

        # 获取第一层的MLP作为模板(用于创建占位专家)
        first_layer_mlp = self.model.layers[1].mlp

        # ========== 占位专家初始化 ==========
        self._placeholders = [
            copy.deepcopy(first_layer_mlp.experts[i]).to(self.dev)
            for i in range(6)
        ]
        self._ph_in_use = [False] * 6
        self._ph_mapping = [None] * 6  # 每个占位专家当前存储的 (layer_idx, expert_id)

        # 反向映射: (layer_idx, expert_id) -> 占位专家对象
        self.expert_to_placeholder = {}

        # ========== 预取相关状态 ==========
        self.prefetch_layers = 0  # 预取层数
        self.is_decode = False  # 是否处于decode阶段
        self.prefetch_list = {}  # 已预取的专家列表
        self.prefetching_list = {}  # 正在预取的专家列表
        self._prefetch_thread = None  # 异步预取 Future
        self._prefetch_lock = threading.Lock()  # 预取锁
        self._prefetch_stream = torch.cuda.Stream()  # 专用CUDA Stream用于预取拷贝
        self._transfer_streams = [torch.cuda.Stream() for _ in range(3)]  # gate/up/down 并行传输

        # ========== 分词器初始化 ==========
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ========== KV缓存初始化 ==========
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        # ========== 推理配置 ==========
        self.cpu_offload = self.config.cpu_offload  # CPU卸载开关
        self.beam_width = self.config.beam_width  # Beam搜索宽度

        # prefil_pre 标志位
        self.prefil_pre = False

        # ========== MoE配置 ==========
        self.n_layer = len(self.model.layers)  # Transformer层数 (26层)
        self.n_expert = self.model.config.n_routed_experts  # 路由专家数量 (64)
        self.n_shared_experts = self.config.n_shared_experts  # 共享专家数量

        # ========== 统计信息 ==========
        self.expert_selection_stats = []  # 记录专家选择情况
        self.expert_time_stats = []  # 记录专家处理时间

        # 初始化每层的专家选择历史和命中统计
        self.expert_selection_history = {}
        self.hit_stats = {}
        for i in range(1, 27):
            self.expert_selection_history[i] = []
            self.hit_stats[i] = {"hits": 0, "total": 0}

        # 每层CPU专家处理时间累计
        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(1, 27)}

        # 专家命中统计
        self.cnt_expert_hit = 0
        self.cnt_prefetch_hit = 0
        self.cnt_gpu_available = 0
        self.cnt_expert_all = 0

        # ========== GPU资源分配 ==========
        # 将非专家组件(Embedding, Attention, LayerNorm等)移到GPU
        self.bring_non_expert_to_gpu()
        self._print_gpu_memory_breakdown("非专家组件加载后")

        # 专家位置矩阵: [n_layer, n_expert], 0=CPU, 1=GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)

        # 计算可容纳到GPU的专家数量
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{(self.n_layer - 1) * self.n_expert}"
        )

        # 设置专家位置并加载热点专家到GPU
        self.set_expert_loc(n_expert_on_gpu)
        self._print_gpu_memory_breakdown("热点专家加载后")

        # ========== S1: Pre-pin all CPU expert weights ==========
        for layer in self.model.layers:
            if not hasattr(layer.mlp, "experts"):
                continue
            for expert in layer.mlp.experts:
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    w = getattr(expert, name, None)
                    if w is not None and not w.weight.data.is_cuda and not w.weight.data.is_pinned():
                        w.weight.data = w.weight.data.pin_memory()

        # ========== S2: Thread pool ==========
        self._executor = ThreadPoolExecutor(max_workers=6)

        # ========== 层时间统计 ==========
        self.layer_time_stats = []
        self.layer_time_accumulator = {}
        for i in range(1, 27):
            self.layer_time_accumulator[i] = 0.0

        # 专家统计: 记录上一次和当前迭代的专家选择
        self.last_iter_expert_stats = {
            i: {"expert_ids": [], "token_counts": []} for i in range(1, 27)
        }
        self.current_iter_expert_stats = {
            i: {"expert_ids": [], "token_counts": []} for i in range(1, 27)
        }

        # 层数据收集
        self.layer_data = {}

        # ========== 加载专家到GPU ==========
        tick = time.time()
        self.bring_expert_to_gpu()
        self._print_gpu_memory_breakdown("所有GPU专家加载后")
        print(f"专家 移动总耗时: {(time.time() - tick) * 1000:.2f}ms")
        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """
        将非专家组件移动到GPU

        说明:
            将模型中不需要动态加载的组件固定保留在GPU上:
            - Embedding层
            - Attention层
            - LayerNorm层
            - Gate投影层
            - 共享专家(Shared Experts)
        """
        # 输出层 lm_head
        self.lm_head.to(self.dev)
        # 词嵌入层
        self.model.embed_tokens.to(self.dev)
        # 输出层归一化
        self.model.norm.to(self.dev)
        # 第0层整体移到GPU (第0层是共享层)
        self.model.layers[0].to(self.dev)

        # 遍历1-26层，将非专家组件移到GPU
        for i in range(len(self.model.layers)):
            if i != 0:
                # 自注意力机制
                self.model.layers[i].self_attn.to(self.dev)
                # 输入层归一化
                self.model.layers[i].input_layernorm.to(self.dev)
                # Gate投影层 (用于路由决策)
                self.model.layers[i].mlp.gate.to(self.dev)
                # 注意力后归一化
                self.model.layers[i].post_attention_layernorm.to(self.dev)

        # 共享专家每层都在GPU上计算
        for i in range(1, 27):
            self.model.layers[i].mlp.shared_experts.to(self.dev)

    def get_hot_expert(self):
        """
        获取每层热点专家 (按处理token数量降序排序)

        说明:
            - 热点专家是指处理token数量最多的专家
            - 热点专家会被优先加载到GPU
            - 仅在decode阶段调用

        返回:
            dict: {layer_id: [expert_id_0, expert_id_1, ...]} 按token数量排序的专家列表
        """
        # 仅在decode阶段进行热点分析
        if not hasattr(self, "is_decode") or not self.is_decode:
            return {}

        hot_experts = {}

        # 遍历每一层
        for layer_id in range(self.n_layer):
            if layer_id > 0:
                # 获取当前层的专家统计
                expert_ids = self.current_iter_expert_stats[layer_id]["expert_ids"]
                token_counts = self.current_iter_expert_stats[layer_id]["token_counts"]

                # 合并专家ID和对应token数量
                expert_data = list(zip(expert_ids, token_counts))

                # 按token数量降序排序 (处理越多token的专家越热点)
                sorted_experts = sorted(expert_data, key=lambda x: x[1], reverse=True)

                # 提取排序后的专家ID列表
                hot_experts[layer_id] = [expert[0] for expert in sorted_experts]

                # 保存当前迭代结果到last_iter，供下一次参考
                self.last_iter_expert_stats[layer_id] = {
                    "expert_ids": expert_ids.copy(),
                    "token_counts": token_counts.copy(),
                }

                # 清空当前迭代记录，为下一token准备
                self.current_iter_expert_stats[layer_id]["expert_ids"].clear()
                self.current_iter_expert_stats[layer_id]["token_counts"].clear()

        # 保存热点专家到成员变量
        self.hot_experts = hot_experts
        return hot_experts

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """
        设置专家位置 (哪些专家在GPU上，哪些在CPU上)

        参数:
            n_expert_on_gpu: GPU上可容纳的专家数量
            popular_experts: 可选的热点专家列表 [(layer, expert), ...]

        说明:
            - 默认从hot/deep.txt文件加载历史热点专家
            - 如果文件不存在，使用默认策略:每层前40个专家加载到GPU
            - 设置expert_loc矩阵，1=GPU，0=CPU
        """
        # 如果没有指定热点专家，从文件加载或使用默认策略
        if popular_experts is None:
            hot_experts_file = self.config.hot_expert_file
            if os.path.exists(hot_experts_file):
                # 从文件加载热点专家
                try:
                    with open(hot_experts_file, "r") as f:
                        popular_experts = [
                            tuple(map(int, line.strip().split(",")))
                            for line in f
                            if line.strip()
                        ]
                    # print(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    print(f"Error loading hot experts: {e}")
            else:
                # 使用默认热点: 每层前40个专家
                popular_experts = []
                for layer in range(1, 27):  # 1-26层
                    for expert in range(40):  # 每层前40个专家
                        popular_experts.append((layer, expert))

        # 设置GPU上的专家数量上限
        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))

        # 更新expert_loc矩阵
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1  # 1=在GPU上

    def _get_available_placeholder(self):
        """获取一个可用的占位专家，返回 (index, placeholder) 或 (None, None)"""
        for i, in_use in enumerate(self._ph_in_use):
            if not in_use:
                self._ph_in_use[i] = True
                return i, self._placeholders[i]
        return None, None

    def _release_placeholder_by_index(self, idx):
        """按索引释放占位专家"""
        if idx is not None and 0 <= idx < len(self._placeholders):
            stored = self._ph_mapping[idx]
            if stored in self.expert_to_placeholder:
                del self.expert_to_placeholder[stored]
            self._ph_mapping[idx] = None
            self._ph_in_use[idx] = False

    def _is_expert_available_on_gpu(self, i_layer, expert_id):
        """检查专家是否可在GPU上使用（驻留/已预取/正在预取/占位专家中）"""
        if self.is_expert_in_gpu_now(i_layer, expert_id):
            return True
        if (
            i_layer in self.prefetch_list
            and expert_id in self.prefetch_list[i_layer]
        ):
            return True
        if (
            i_layer in self.prefetching_list
            and expert_id in self.prefetching_list[i_layer]
        ):
            return True
        for stored in self._ph_mapping:
            if stored and stored == (i_layer, expert_id):
                return True
        return False

    def _count_gpu_residents(self, i_layer, expert_ids):
        """统计指定层的专家列表中有多少已在GPU上可用"""
        count = 0
        for eid in expert_ids:
            if self._is_expert_available_on_gpu(i_layer, eid):
                count += 1
        return count

    def _async_load_expert(self, layer_idx, expert_id, target_placeholder=None, ph_idx=None):
        """
        加载专家权重到GPU占位专家

        参数:
            layer_idx: 层索引
            expert_id: 专家ID
            target_placeholder: 可选的目标占位专家对象
            ph_idx: 可选的占位专家索引（与 target_placeholder 配对使用）
                - 都为 None: 自动分配占位专家并记录映射（prefetch路径）
                - 都提供: 直接使用（ondemand路径）

        说明:
            - 使用Pinned Memory加速CPU->GPU传输
            - 使用3个CUDA Stream并行传输gate/up/down_proj
            - 当自动分配时，记录双向映射关系
        """
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]

        if next(expert.parameters()).is_cuda:
            return

        # ===== 1. 分配占位专家（如果未提供）=====
        allocated = False
        if target_placeholder is None:
            ph_idx, target_placeholder = self._get_available_placeholder()
            if target_placeholder is None:
                raise RuntimeError("No available expert placeholder")
            allocated = True

        # ===== 2. 记录映射（仅自动分配时）=====
        if allocated:
            self._ph_mapping[ph_idx] = (layer_idx, expert_id)
            self.expert_to_placeholder[(layer_idx, expert_id)] = target_placeholder

        # ===== 3. Multi-stream 并行传输 =====
        for stream, name in zip(self._transfer_streams, ["gate_proj", "up_proj", "down_proj"]):
            with torch.cuda.stream(stream):
                dst = getattr(target_placeholder, name).weight.data
                src = getattr(expert, name).weight.data
                dst.copy_(src, non_blocking=True)
        for s in self._transfer_streams:
            torch.cuda.synchronize(s)

        return ph_idx, target_placeholder

    def release_placeholder(self, layer_idx, expert_id):
        """
        释放已用完的占位专家

        参数:
            layer_idx: 当前处理的层索引
            expert_id: 当前处理的专家ID

        说明:
            - 当处理的层号大于占位专家存储的层号时，释放占位专家
        """
        for i in range(len(self._placeholders)):
            stored = self._ph_mapping[i]
            if stored and (
                stored[0] < layer_idx
                or (stored[0] == self.n_layer - 1 and layer_idx <= 1)
            ):
                self._release_placeholder_by_index(i)

    def is_expert_loaded(self, layer_id, expert_id):
        """
        检查指定层的专家是否已完成加载

        参数:
            layer_id: 层ID
            expert_id: 专家ID
        返回:
            bool: 是否已完成加载 (不在prefetching_list中)
        """
        return (
            layer_id not in self.prefetching_list
            or expert_id not in self.prefetching_list[layer_id]
        )

    def bring_expert_to_gpu(self):
        """
        将热点专家加载到GPU

        说明:
            - 根据expert_loc矩阵，将标记为GPU的专家加载到GPU
            - 记录成功加载的专家数量到test.txt
            - 处理显存溢出情况
        """
        expert_count = 0
        try:
            # 遍历所有层和专家
            for i in range(self.n_layer):
                for j in range(self.n_expert):
                    # 如果专家位置标记为GPU，则加载
                    if self.is_expert_in_gpu(i, j):
                        self.model.layers[i].mlp.experts[j].to(self.dev)
                        expert_count += 1

            # 记录成功加载的专家数量
            with open("test.txt", "a") as f:
                f.write(
                    f"模型: deep, batch_size: {self.batch_size}, 成功加载专家数量: {expert_count}\n"
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 记录显存溢出时的专家数量
                with open("test.txt", "a") as f:
                    f.write(
                        f"模型: deep, batch_size: {self.batch_size}, 显存溢出时专家数量: {expert_count}\n"
                    )
                raise
            else:
                raise
                raise  # 其他异常直接抛出

    def is_expert_in_gpu(self, i_layer, i_expert):
        """
        判断专家是否在GPU上 (根据expert_loc矩阵)

        参数:
            i_layer: 层索引
            i_expert: 专家索引

        返回:
            bool: 是否标记为在GPU上
        """
        return self.expert_loc[i_layer, i_expert] == 1

    def is_expert_in_gpu_now(self, i_layer, i_expert):
        """
        检查专家当前是否实际在GPU上 (包括占位专家)

        参数:
            i_layer: 层索引
            i_expert: 专家索引

        返回:
            bool: 是否实际在GPU上
        """
        # 检查专家参数是否在GPU上
        expert = self.model.layers[i_layer].mlp.experts[i_expert]
        return next(expert.parameters()).is_cuda

    def calc_n_expert_on_gpu(self):
        """
        计算GPU上可容纳的专家数量

        说明:
            - 根据GPU显存大小计算可容纳的专家数量
            - 不同batch_size有不同的策略
            - 使用启发式方法估算

        返回:
            int: 可在GPU上容纳的专家数量
        """
        # 获取单个专家的参数量
        fine_expert = self.model.layers[1].mlp.experts[0]
        n_param = sum(p.numel() for p in fine_expert.parameters())

        # 获取GPU显存信息
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        total_mem_gb = total_mem / (1024**3)
        used_mem = torch.cuda.memory_allocated(self.dev)
        used_mem_gb = used_mem / (1024**3)
        # 预留20%显存，使用80%
        free_mem = total_mem * 0.8 - used_mem
        free_mem_gb = free_mem / (1024**3)

        # 单个专家的显存占用 (bfloat16 = 2 bytes per param)
        expert_mem_bytes = n_param * 2
        expert_mem_mb = expert_mem_bytes / (1024**2)
        expert_mem_gb = expert_mem_bytes / (1024**3)

        # 调试信息
        print(f"[DEBUG calc_n_expert_on_gpu]")
        print(f"  GPU总显存: {total_mem_gb:.2f} GB")
        print(f"  已占用显存: {used_mem_gb:.2f} GB")
        print(f"  可用显存(80%): {free_mem_gb:.2f} GB")
        print(f"  单个专家参数量: {n_param:,}")
        print(f"  单个专家显存占用: {expert_mem_mb:.2f} MB ({expert_mem_gb:.4f} GB)")
        print(f"  模型总专家数: {(self.n_layer - 1) * self.n_expert}")

        # 根据batch_size使用不同策略
        # if self.batch_size == 64:
        #     n_expert = 893
        # elif self.batch_size == 32:
        #     n_expert = 993
        # elif self.batch_size == 16:
        #     n_expert = 693
        # else:
        # 计算能容纳的专家数量，不使用硬编码的magic number
        calculated_n = int(free_mem / expert_mem_bytes)
        # 预留一定buffer给KV缓存和中间计算，根据batch_size调整
        if self.batch_size >= 16:
            buffer_factor = 0.4  # 大batch需要更多KV缓存空间
        elif self.batch_size >= 8:
            buffer_factor = 0.5  # 中等batch
        else:
            buffer_factor = 0.7  # 小batch
        n_expert = int(calculated_n * buffer_factor)
        print(f"  计算的专家数量(无buffer): {calculated_n}")
        print(f"  buffer_factor: {buffer_factor}")

        # 确保返回正数
        n_expert = max(n_expert, 1)
        # 限制最大值为总专家数
        max_experts = (self.n_layer - 1) * self.n_expert
        n_expert = min(n_expert, max_experts)

        print(f"  最终GPU专家数量: {n_expert}/{max_experts}")
        print(f"  预计GPU显存占用: {n_expert * expert_mem_mb:.2f} MB")

        return n_expert

    def _print_gpu_memory_breakdown(self, stage: str):
        total = torch.cuda.get_device_properties(self.dev).total_memory
        used = torch.cuda.memory_allocated(self.dev)
        reserved = torch.cuda.memory_reserved(self.dev)
        gb = lambda b: b / (1024 ** 3)
        mb = lambda b: b / (1024 ** 2)
        print(f"\n{'='*60}")
        print(f"[GPU显存] {stage}")
        print(f"  GPU总显存:        {gb(total):8.2f} GB")
        print(f"  已分配(PyTorch):  {gb(used):8.2f} GB ({mb(used):.0f} MB)")
        print(f"  已预留(CUDA):     {gb(reserved):8.2f} GB ({mb(reserved):.0f} MB)")
        print(f"  剩余可用(估算):   {gb(total - reserved):8.2f} GB")
        print(f"{'='*60}\n")

    def initial_beam_tensor(self, input_tensor):
        """
        Beam搜索张量初始化

        参数:
            input_tensor: 输入张量 (batch*beam, seq, beam)

        返回:
            调整后的张量 (batch*beam, 1)
        """
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [
                i * self.beam_width
                for i in range(input_tensor.shape[0] // self.beam_width)
            ]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        """
        文本生成主函数

        参数:
            text: 输入文本，str或list
            output_token: 生成的token数量
            input_token: 输入token数量限制

        返回:
            tuple: (prefill_time, decode_time, hit_rate, stats)
        """
        torch.set_num_threads(self.config.cpu_threads)

        # 初始化KV缓存
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        # 初始化统计
        self.cnt_expert_hit = 0
        self.cnt_prefetch_hit = 0
        self.cnt_gpu_available = 0
        self.cnt_expert_all = 0
        self.expert_selection_stats = []
        self.expert_time_stats = []

        self._gate_predictions = {}
        self._gate_pred_stats = {
            'total_layers': 0,
            'total_actual': 0,
            'total_predicted': 0,
            'total_overlap': 0,
        }

        # 处理输入文本
        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        # 分词
        input_ids, position_ids, attention_mask = self.tokenize(text, input_token)

        # 截取指定长度的输入
        if input_token is not None:
            input_ids = torch.stack(
                [
                    ids[:input_token] if len(ids) > input_token else ids
                    for ids in input_ids
                ]
            )
            position_ids = torch.stack(
                [
                    pos[:input_token] if len(pos) > input_token else pos
                    for pos in position_ids
                ]
            )
            attention_mask = attention_mask[:, :, :, :input_token]

        # 开始生成
        tick = time.time()
        self.is_decode = False  # 先处于prefill阶段
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        self.token_decode_times = []
        self.perf_stats = {
            "token_embedding": [],
            "self_attention": [],
            "moe_gating": [],
            "expert_compute": [],
            "expert_compute-cpu": [],
        }

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,  # 跳过前1次迭代
                warmup=3,  # 预热1次迭代
                active=1,  # 记录3次迭代
                repeat=1,  # 只执行1轮
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        prof.start()

        for i_token in range(output_token):
            prof.step()
            token_start_time = time.time()  # 记录单个token开始时间
            # if self.beam_width == 1:
            # print(self.tokenizer.decode(input_ids[0]))
            # TODO: streaming output for beam search
            if self.is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(
                        input_ids[i, :].tolist()
                    )
            # new_mask = torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=self.dev)
            # attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            #
            if self.is_decode:
                # decode阶段处理
                if (
                    attention_mask.dim() == 4
                    and attention_mask.shape[-1] == attention_mask.shape[-2]
                ):
                    # 从方阵mask转换为序列mask [1,1,seq_len,seq_len] -> [1,1,1,seq_len]
                    seq_len = attention_mask.shape[-1]
                    attention_mask = attention_mask[..., :1, :]  # 取第一行

                # 扩展序列长度维度
                seq_len = attention_mask.shape[-1]
                new_attention_mask = torch.ones(
                    (attention_mask.shape[0], 1, 1, seq_len + 1),
                    dtype=torch.bool,
                    device=self.dev,
                )
                new_attention_mask[..., :seq_len] = attention_mask[..., :seq_len]
                attention_mask = new_attention_mask
            else:
                # prefill阶段保持原有方阵mask [1,1,seq_len,seq_len]
                pass
            new_position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + input_ids.shape[1],
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask)

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # normalize logits
            logits = F.softmax(logits, dim=-1)

            # greedy search:
            # output = torch.argmax(logits, dim=-1)

            # beam_search:
            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            # input_ids.shape: (batch_size, seq_len=1)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )
            token_time = time.time() - token_start_time
            self.token_decode_times.append(token_time)
            # print(f"Token {i_token} decode time: {token_time * 1000:.2f}ms")
            self.logger.log_token_decode(i_token, token_time)

            # position_ids.shape: (1, 1)
            if not self.is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            self.is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        print("\nToken decode time summary:")
        for i, t in enumerate(self.token_decode_times):
            print(f"Token {i}: {t * 1000:.2f}ms")
        print(f"Total decode time: {decode_time * 1000:.2f}ms")
        print(
            f"Average per token: {decode_time * 1000 / len(self.token_decode_times):.2f}ms"
        )
        print("--------------------")

        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")
        prof.stop()

        n = self.cnt_expert_all if self.cnt_expert_all > 0 else 1
        hit_rates = {
            "hot_hit_rate": self.cnt_expert_hit / n,
            "prefetch_hit_rate": self.cnt_prefetch_hit / n,
            "gpu_available_rate": self.cnt_gpu_available / n,
            "hot_hits": self.cnt_expert_hit,
            "prefetch_hits": self.cnt_prefetch_hit,
            "gpu_available": self.cnt_gpu_available,
            "total_experts": self.cnt_expert_all,
        }
        print(f"  Hot表命中率: {hit_rates['hot_hit_rate']:.2%} ({self.cnt_expert_hit}/{self.cnt_expert_all})")
        print(f"  预取命中率:  {hit_rates['prefetch_hit_rate']:.2%} ({self.cnt_prefetch_hit}/{self.cnt_expert_all})")
        print(f"  GPU可用率:   {hit_rates['gpu_available_rate']:.2%} ({self.cnt_gpu_available}/{self.cnt_expert_all})")

        gs = self._gate_pred_stats
        if gs['total_layers'] > 0:
            precision = gs['total_overlap'] / gs['total_predicted'] if gs['total_predicted'] > 0 else 0
            recall = gs['total_overlap'] / gs['total_actual'] if gs['total_actual'] > 0 else 0
            print(f"  门控预测准确率: {precision:.2%} (命中{gs['total_overlap']}/预测{gs['total_predicted']}, "
                  f"召回{recall:.2%}, {gs['total_layers']}层)")

        return (
            prefill_time,
            decode_time,
            hit_rates,
            {
                "perf_stats": self.perf_stats,
                "expert_selection": self.expert_selection_stats,
                "expert_time": self.expert_time_stats,
                "layer_time": self.layer_time_stats,
                "outputs": decode_strings,
                "expert_hot_stats": self.get_expert_stats(),
                "layer_time_avg": {
                    i: self.layer_time_accumulator[i]
                    / max(
                        1, len([x for x in self.layer_time_stats if x["layer_id"] == i])
                    )
                    for i in range(1, 27)
                },
            },
        )

    def tokenize(self, text, input_token):
        """
        文本分词函数

        参数:
            text: 输入文本，str或list
            input_token: 最大token长度

        返回:
            tuple: (input_ids, position_ids, attention_mask)
        """
        # 处理输入格式
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")

        # 确保文本数量与batch_size匹配
        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[: self.batch_size]

        # 分词
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt",
        )

        # 提取input_ids和attention_mask
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.bool().to(self.dev)

        # 生成 position_ids
        seq_length = input_ids.shape[1]
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=self.dev)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )

        # 修正attention_mask形状为4D (batch, num_heads, seq_len, seq_len)
        # 注意：这里需要根据模型的实际头数(32)来扩展
        if attention_mask.dim() == 2:
            # 从(batch, seq_len)扩展到(batch, 1, 1, seq_len)
            # attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # # 然后扩展到(batch, num_heads, seq_len, seq_len)
            # attention_mask = attention_mask.expand(-1, 1, -1, -1)
            attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(-1)  # (batch, 1, seq_len, 1)
            attention_mask = attention_mask.expand(
                -1, -1, -1, seq_length
            )  # (batch, 1, seq_len, seq_len)
        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask):
        """
        模型前向传播核心函数

        处理流程:
        1. Embedding: 将token id转换为hidden states
        2. 遍历各层:
            - Self-Attention: 自注意力计算
            - MoE Layer: 专家路由与计算
        3. Norm + LM_Head: 输出logits

        参数:
            input_ids: 输入token IDs
            position_ids: 位置IDs
            attention_mask: 注意力掩码

        返回:
            lm_logis: 预测的logits
        """
        # 获取隐藏层维度
        hidden_dim = self.model.config.hidden_size

        # ===== Embedding层 =====
        tick = time.time()
        inps = self.model.embed_tokens(input_ids)
        self.perf_stats["token_embedding"].append(time.time() - tick)

        # 如果是decode阶段，初始化层时间统计
        if self.is_decode:
            total_decode_start = time.time()
            layer_times = {i: 0.0 for i in range(1, 27)}
            layer_times_fwd = {i: 0.0 for i in range(1, 27)}
            layer_times_mid = {i: 0.0 for i in range(1, 27)}
            layer_times_final = {i: 0.0 for i in range(1, 27)}

        # 获取batch和序列长度
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 层处理开始
        layer_start_time = time.time()
        layer_total_time = 0.0
        isprefetch = False

        # ===== 遍历每一层 =====
        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()
            if i_layer == 0:
                # 第0层是共享层，直接计算
                inps = layer.mlp(inps)
            else:
                # ===== 1. 释放占位专家 =====
                self.release_placeholder(i_layer, 0)

                # 等待上一层的异步预取完成（Future + CUDA Stream）
                if self._prefetch_thread is not None:
                    self._prefetch_thread.result()
                torch.cuda.synchronize(self._prefetch_stream)

                # 保存残差
                original_inps_shape = inps.shape
                self.cpu_expert_time_per_layer[i_layer] = 0
                inps_residual = inps

                # ===== 2. Input LayerNorm =====
                inps = layer.input_layernorm(inps)

                # ===== 3. Self-Attention =====
                inps = inps.view(batch_size, seq_len, hidden_dim)
                tick = time.time()
                attn_output = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                )
                #      data collection

                torch.cuda.synchronize()  # 确保当前操作完成
                self.perf_stats["self_attention"].append(time.time() - tick)
                # print(f"输出attn_outputtime: {time.time() - tick}...\n")
                # 根据返回类型处理结果
                if isinstance(attn_output, tuple):
                    if len(attn_output) == 2:  # 只有attn_output和present_key_value
                        inps, present_key_value = attn_output
                        self_attn_weights = None
                    else:  # 3个返回值
                        inps, self_attn_weights, present_key_value = attn_output
                else:  # 只有attn_output
                    inps = attn_output
                    self_attn_weights = None
                    present_key_value = None
                # inps.shape: (batch_size, seq_len/token_num, embed_dim)
                # if self.past_key_values is None:
                #     self.past_key_values = [None] * self.n_layer

                # # 存储当前层的缓存
                # self.past_key_values[i_layer] = present_key_value
                inps = inps_residual + inps
                inps_residual = inps
                inps = layer.post_attention_layernorm(inps)
                inps = inps.view(-1, hidden_dim)
                # inps.shape: (batch_size*seq_len*embed_dim/hidden_dim, hidden_dim)
                layer_idx = i_layer
                if layer_idx not in self.layer_data:
                    self.layer_data[layer_idx] = {
                        "hidden_states": [],
                        "expert_indices": [],
                    }
                inps = inps.view(batch_size, seq_len, hidden_dim)  # 恢复3D形状
                # ===== 4. MoE路由 (Gate Network) =====
                # 获取专家处理前的hidden states
                pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)

                # 记录 gating 时间
                tick = time.time()
                # Gate网络: 计算每个token对各专家的亲和分数，选择Top-K专家
                # selected_experts: 选中的专家ID [batch*seq, top_k]
                # routing_weights: 各专家的权重 [batch*seq, top_k]
                selected_experts, routing_weights, _ = layer.mlp.gate(inps)
                torch.cuda.synchronize()
                self.perf_stats["moe_gating"].append(time.time() - tick)

                # 记录专家选择
                layer_expert_stats = {
                    "layer_id": i_layer,
                    "expert_ids": selected_experts.tolist(),
                }
                self.expert_selection_stats.append(layer_expert_stats)

                # ===== 5. 共享专家计算 =====
                # 共享专家(Shared Experts)每层都参与计算，输出加到最终结果
                inps_after_experts = torch.zeros_like(inps, device=self.dev)
                experts = layer.mlp.experts
                shared_output = torch.zeros_like(inps)
                expert_out = self.model.layers[i_layer].mlp.shared_experts(inps)
                shared_output += expert_out

                # ===== 6. 统计每个专家处理的token数量 =====
                expert_token_counts = {}
                for expert_id in selected_experts.unique():
                    # 统计选中该专家的token数量
                    mask = (selected_experts == expert_id).any(dim=1)
                    expert_token_counts[expert_id.item()] = mask.sum().item()

                # 按token数量降序排序 (热点专家优先)
                sorted_experts = sorted(
                    expert_token_counts.items(), key=lambda x: x[1], reverse=True
                )
                # 保存到当前层的专家统计中
                self.current_iter_expert_stats[i_layer] = {
                    "expert_ids": [e[0] for e in sorted_experts],  # 专家ID
                    "token_counts": [e[1] for e in sorted_experts],  # 对应token数量
                }
                layer_i_stats = self.current_iter_expert_stats[i_layer]
                # for expert_id, token_count in zip(layer_i_stats['expert_ids'], layer_i_stats['token_counts']):
                #     print(f"专家 {expert_id} 处理了 {token_count} 个token")
                filtered_expert_ids = []
                filtered_token_counts = []
                gpu_onloaded = []
                for expert_id, token_count in zip(
                    layer_i_stats["expert_ids"], layer_i_stats["token_counts"]
                ):
                    # 检查专家是否已经在GPU上
                    expert_in_gpu = False
                    if self.is_expert_in_gpu_now(i_layer, expert_id):
                        expert_in_gpu = True
                    elif (
                        i_layer in self.prefetch_list
                        and expert_id in self.prefetch_list[i_layer]
                    ) or (
                        i_layer in self.prefetching_list
                        and expert_id in self.prefetching_list[i_layer]
                    ):
                        expert_in_gpu = True
                    else:
                        for stored in self._ph_mapping:
                            if stored and stored == (i_layer, expert_id):
                                expert_in_gpu = True
                                break

                    if not expert_in_gpu:
                        filtered_expert_ids.append(expert_id)
                        filtered_token_counts.append(token_count)
                    # else:
                    #     gpu_onloaded.append(expert_id)
                # 直接使用过滤后的结果，不需要再次排序
                sorted_experts = list(zip(filtered_expert_ids, filtered_token_counts))

                # ===== 7. Next layer prediction =====
                selected_expert_ids = selected_experts.unique().tolist()

                if hasattr(self, '_gate_predictions') and i_layer in self._gate_predictions:
                    predicted = self._gate_predictions.pop(i_layer)
                    actual_set = set(selected_expert_ids)
                    predicted_set = set(predicted)
                    overlap = actual_set & predicted_set
                    self._gate_pred_stats['total_layers'] += 1
                    self._gate_pred_stats['total_actual'] += len(actual_set)
                    self._gate_pred_stats['total_predicted'] += len(predicted_set)
                    self._gate_pred_stats['total_overlap'] += len(overlap)

                next_predicted_expert_ids = []
                next_sorted_experts = []
                next_sorted_experts_filtered = []
                if i_layer < self.n_layer - 1:
                    next_layer = self.model.layers[i_layer + 1]
                    with torch.no_grad():
                        next_predicted_experts, next_routing_weights, _ = (
                            next_layer.mlp.gate(inps)
                        )

                    next_expert_token_counts = {}
                    for batch_idx in range(batch_size * seq_len):
                        for expert in next_predicted_experts[batch_idx]:
                            next_expert_token_counts[expert.item()] = (
                                next_expert_token_counts.get(expert.item(), 0) + 1
                            )
                    next_sorted_experts = sorted(
                        next_expert_token_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    next_predicted_expert_ids = [
                        expert[0] for expert in next_sorted_experts
                    ]

                    for eid, tokens in next_sorted_experts:
                        if not self._is_expert_available_on_gpu(i_layer + 1, eid):
                            next_sorted_experts_filtered.append((eid, tokens))

                    self._gate_predictions[i_layer + 1] = next_predicted_expert_ids

                # ===== 8. Scheduling decision =====
                if self.is_decode:
                    k = len(selected_expert_ids)
                    r_cur = self._count_gpu_residents(i_layer, selected_expert_ids)
                    r_next = (
                        self._count_gpu_residents(
                            i_layer + 1, next_predicted_expert_ids
                        )
                        if i_layer < self.n_layer - 1
                        else k
                    )
                    mode, ondemand_count, prefetch_count, offload_count = (
                        self.scheduler.decide_decode(r_cur, r_next, k)
                    )
                    ondemand_experts = []
                    prefetch_experts = []
                else:
                    mode = "prefill"
                    r_cur = self._count_gpu_residents(i_layer, selected_expert_ids)
                    ondemand_experts, prefetch_experts, prefetch_count = (
                        self.scheduler.decide_prefill_schedule(
                            sorted_experts,
                            next_sorted_experts_filtered,
                            cur_gpu_resident=r_cur,
                            is_last_layer=(i_layer >= self.n_layer - 1),
                        )
                    )
                    ondemand_count = len(ondemand_experts)

                # ===== 9. Expert classification =====
                experts_in_placeholder = []
                self.prefil_pre = (
                    self.prefil_pre if hasattr(self, "prefil_pre") else False
                )

                inps_after_experts = torch.zeros_like(inps, device=self.dev)
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts,
                    num_classes=self.n_expert + self.n_shared_experts,
                ).permute(2, 1, 0)

                cpu_experts = []
                gpu_experts = []
                experts_in_gpu = []
                experts_loading = []
                experts_remaining = []
                experts_remaining2 = []
                decode_ondemand_used = 0

                if self.is_decode:
                    laymid = time.time() - layer_tick
                    laypid = time.time()

                for i_expert in selected_expert_ids:
                    self.cnt_expert_all += 1

                    is_hot = (
                        self.expert_loc[i_layer, i_expert] == 1
                        and self.is_expert_in_gpu_now(i_layer, i_expert)
                    )
                    is_prefetched = (
                        i_layer in self.prefetch_list
                        and i_expert in self.prefetch_list[i_layer]
                    )
                    is_loading = (
                        i_layer in self.prefetching_list
                        and i_expert in self.prefetching_list[i_layer]
                    )
                    is_placeholder = any(
                        stored and stored == (i_layer, i_expert)
                        for stored in self._ph_mapping
                    )

                    if is_hot:
                        self.cnt_expert_hit += 1
                    if is_hot or is_prefetched or is_placeholder:
                        self.cnt_gpu_available += 1
                        self.cnt_prefetch_hit += (not is_hot)
                    self.logger.log_expert_hit(is_hot or is_prefetched or is_placeholder)

                    if self.is_expert_in_gpu_now(i_layer, i_expert):
                        gpu_experts.append(i_expert)
                        experts_in_gpu.append(i_expert)
                        self.logger.log_expert_class("gpu")
                        continue

                    if i_layer not in self.prefetch_list:
                        self.prefetch_list[i_layer] = []
                    if i_layer not in self.prefetching_list:
                        self.prefetching_list[i_layer] = []

                    if i_expert in self.prefetch_list[i_layer]:
                        experts_in_placeholder.append(i_expert)
                        self.logger.log_expert_class("prefetch")
                        continue

                    if i_expert in self.prefetching_list[i_layer]:
                        experts_loading.append(i_expert)
                        self.logger.log_expert_class("loading")
                        continue

                    if self.is_decode:
                        if mode in ("A", "default") and decode_ondemand_used < ondemand_count:
                            experts_remaining.append(i_expert)
                            self.logger.log_expert_class("ondemand")
                            decode_ondemand_used += 1
                        else:
                            cpu_experts.append(i_expert)
                            self.logger.log_expert_class("cpu")
                    else:
                        if i_expert in ondemand_experts:
                            experts_remaining.append(i_expert)
                            self.logger.log_expert_class("ondemand")
                        else:
                            cpu_experts.append(i_expert)
                            self.logger.log_expert_class("cpu")

                # ===== 9.5 Release unused prefetched experts =====
                # 释放当前层已预取但未选中的专家，释放占位槽供下层使用
                if i_layer in self.prefetch_list:
                    selected_set = set(selected_expert_ids)
                    unused_prefetched = [
                        eid for eid in self.prefetch_list[i_layer]
                        if eid not in selected_set
                    ]
                    for eid in unused_prefetched:
                        key = (i_layer, eid)
                        if key in self.expert_to_placeholder:
                            ph_obj = self.expert_to_placeholder[key]
                            for idx, ph in enumerate(self._placeholders):
                                if ph is ph_obj:
                                    self._release_placeholder_by_index(idx)
                                    break
                        self.prefetch_list[i_layer].remove(eid)

                # ===== 9.6 Offload excess placeholder experts (Mode B/C) =====
                # 当 offload_count > 0 时，将占位专家中的冗余选中专家降级到 CPU
                # 释放占位槽给下层预取使用
                if self.is_decode and offload_count > 0 and experts_in_placeholder:
                    to_offload = min(offload_count, len(experts_in_placeholder))
                    offloaded = experts_in_placeholder[-to_offload:]
                    experts_in_placeholder = experts_in_placeholder[:-to_offload]
                    for eid in offloaded:
                        key = (i_layer, eid)
                        if key in self.expert_to_placeholder:
                            ph_obj = self.expert_to_placeholder[key]
                            for idx, ph in enumerate(self._placeholders):
                                if ph is ph_obj:
                                    self._release_placeholder_by_index(idx)
                                    break
                        if i_layer in self.prefetch_list and eid in self.prefetch_list[i_layer]:
                            self.prefetch_list[i_layer].remove(eid)
                        cpu_experts.append(eid)

                # ===== 10. Next layer prefetch =====
                do_prefetch = False
                experts_to_prefetch = []
                if i_layer < self.n_layer - 1 and self.config.prefetch_enabled:
                    if self.is_decode and mode in ("B", "default"):
                        if prefetch_count > 0:
                            experts_to_prefetch = next_predicted_expert_ids[:prefetch_count]
                        else:
                            experts_to_prefetch = [
                                e[0] for e in next_sorted_experts[: self.cache]
                            ]
                        do_prefetch = True
                    elif not self.is_decode and prefetch_experts:
                        experts_to_prefetch = prefetch_experts
                        do_prefetch = True

                if do_prefetch:
                    next_layer_idx = i_layer + 1
                    if next_layer_idx not in self.prefetch_list:
                        self.prefetch_list[next_layer_idx] = []
                    if next_layer_idx not in self.prefetching_list:
                        self.prefetching_list[next_layer_idx] = []
                    prefetch_tasks = []
                    for expert_id in experts_to_prefetch:
                        if not self.is_expert_in_gpu_now(next_layer_idx, expert_id):
                            if (
                                expert_id not in self.prefetch_list[next_layer_idx]
                                and expert_id
                                not in self.prefetching_list[next_layer_idx]
                            ):
                                prefetch_tasks.append(expert_id)

                    if prefetch_tasks:
                        self.prefetching_list[next_layer_idx].extend(prefetch_tasks)

                        def _do_prefetch(tasks, layer_idx):
                            for eid in tasks:
                                try:
                                    self._async_load_expert(layer_idx, eid)
                                    with self._prefetch_lock:
                                        if layer_idx in self.prefetch_list:
                                            self.prefetch_list[layer_idx].append(eid)
                                        if (
                                            layer_idx in self.prefetching_list
                                            and eid in self.prefetching_list[layer_idx]
                                        ):
                                            self.prefetching_list[layer_idx].remove(eid)
                                except RuntimeError:
                                    with self._prefetch_lock:
                                        if (
                                            layer_idx in self.prefetching_list
                                            and eid in self.prefetching_list[layer_idx]
                                        ):
                                            self.prefetching_list[layer_idx].remove(eid)
                                    break

                        if (
                            self._prefetch_thread is not None
                        ):
                            self._prefetch_thread.result()
                        self._prefetch_thread = self._executor.submit(
                            _do_prefetch, prefetch_tasks, next_layer_idx
                        )

                # ===== 11. Expert processing (GPU thread + CPU thread) =====
                gpu_results = []
                cpu_results = []
                gpu_time = 0.0
                cpu_time = 0.0
                self._prefetch_thread_started = False

                def process_gpu_experts():
                    nonlocal gpu_time
                    start_time = time.time()

                    def process_experts_in_gpu():
                        results = []
                        threads = []
                        lock = threading.Lock()

                        def process_single_expert(i_expert):
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                return

                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)

                            tick = time.time()
                            expert_output = self.run_expert_at_gpu(
                                i_layer, i_expert, expert_input
                            )
                            self.perf_stats["expert_compute"].append(
                                time.time() - tick
                            )

                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1,
                                (selected_experts[flat_mask] == i_expert)
                                .long()
                                .argmax(dim=1, keepdim=True),
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)

                            with lock:
                                results.append((mask_index, expert_output))

                        for i_expert in experts_in_gpu:
                            t = threading.Thread(
                                target=process_single_expert, args=(i_expert,)
                            )
                            threads.append(t)
                            t.start()

                        for t in threads:
                            t.join()

                        return results

                    def process_experts_in_gpu():
                        results = []
                        for i_expert in experts_in_gpu:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)
                            tick = time.time()
                            expert_output = self.run_expert_at_gpu(
                                i_layer, i_expert, expert_input
                            )
                            self.perf_stats["expert_compute"].append(
                                time.time() - tick
                            )
                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1,
                                (selected_experts[flat_mask] == i_expert)
                                .long()
                                .argmax(dim=1, keepdim=True),
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results

                    def process_experts_in_placeholder():
                        results = []

                        for i_expert in experts_in_placeholder:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)
                            tick = time.time()
                            placeholder = self.expert_to_placeholder.get(
                                (i_layer, i_expert)
                            )
                            if placeholder is not None:
                                torch.cuda.synchronize(self._prefetch_stream)
                                expert_output = placeholder(expert_input)
                            else:
                                ph_idx, placeholder = self._get_available_placeholder()
                                if placeholder is not None:
                                    self._async_load_expert(
                                        i_layer, i_expert, target_placeholder=placeholder, ph_idx=ph_idx
                                    )
                                    expert_output = placeholder(expert_input)
                                else:
                                    expert_input_cpu = expert_input.to("cpu")
                                    expert_output = self.run_expert_at_cpu(
                                        i_layer, i_expert, expert_input_cpu
                                    )
                                    expert_output = expert_output.to(self.dev)
                            torch.cuda.synchronize()
                            self.perf_stats["expert_compute"].append(
                                time.time() - tick
                            )
                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1,
                                (selected_experts[flat_mask] == i_expert)
                                .long()
                                .argmax(dim=1, keepdim=True),
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results

                    def process_experts_loading():
                        results = []

                        for i_expert in experts_loading:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)
                            tick = time.time()
                            placeholder = self.expert_to_placeholder.get(
                                (i_layer, i_expert)
                            )
                            if placeholder is not None:
                                torch.cuda.synchronize(self._prefetch_stream)
                                expert_output = placeholder(expert_input)
                            else:
                                ph_idx, placeholder = self._get_available_placeholder()
                                if placeholder is not None:
                                    self._async_load_expert(
                                        i_layer, i_expert, target_placeholder=placeholder, ph_idx=ph_idx
                                    )
                                    expert_output = placeholder(expert_input)
                                else:
                                    expert_input_cpu = expert_input.to("cpu")
                                    expert_output = self.run_expert_at_cpu(
                                        i_layer, i_expert, expert_input_cpu
                                    )
                                    expert_output = expert_output.to(self.dev)
                            torch.cuda.synchronize()
                            self.perf_stats["expert_compute"].append(
                                time.time() - tick
                            )
                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1,
                                (selected_experts[flat_mask] == i_expert)
                                .long()
                                .argmax(dim=1, keepdim=True),
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results

                    def process_experts_remaining():
                        results = []

                        def process_single_expert(i_expert, ph_idx, placeholder):
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                self._release_placeholder_by_index(ph_idx)
                                return

                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)
                            self._async_load_expert(i_layer, i_expert, target_placeholder=placeholder, ph_idx=ph_idx)
                            expert_output = placeholder(expert_input)

                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1,
                                (selected_experts[flat_mask] == i_expert)
                                .long()
                                .argmax(dim=1, keepdim=True),
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                            self._release_placeholder_by_index(ph_idx)

                        for i_expert in experts_remaining:
                            ph_idx, placeholder = self._get_available_placeholder()
                            if placeholder is None:
                                experts_remaining2.append(i_expert)
                                continue
                            process_single_expert(i_expert, ph_idx, placeholder)

                        for i_expert in experts_remaining2:
                            ph_idx, placeholder = self._get_available_placeholder()
                            if placeholder is None:
                                continue
                            process_single_expert(i_expert, ph_idx, placeholder)

                        return results

                    threads = []
                    results = []

                    def run_and_collect(func):
                        res = func()
                        results.extend(res)

                    threads.append(
                        threading.Thread(
                            target=run_and_collect, args=(process_experts_in_gpu,)
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=run_and_collect,
                            args=(process_experts_in_placeholder,),
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=run_and_collect, args=(process_experts_loading,)
                        )
                    )
                    threads.append(
                        threading.Thread(
                            target=run_and_collect,
                            args=(process_experts_remaining,),
                        )
                    )

                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    gpu_results.extend(results)
                    gpu_time = time.time() - start_time

                def process_cpu_experts():
                    nonlocal cpu_time
                    start_time = time.time()

                    if cpu_experts:
                        if (
                            not hasattr(self, "_cpu_expert_warmed_up")
                            or not self._cpu_expert_warmed_up
                        ):
                            dummy_input = torch.randn(
                                1, hidden_dim, device="cpu"
                            ).to(self.dtype)
                            _ = self.model.layers[i_layer].mlp.experts[
                                cpu_experts[0]
                            ](dummy_input)
                            self._cpu_expert_warmed_up = True

                    for i_expert in cpu_experts:
                        mask = (selected_experts == i_expert).any(dim=1)
                        if not mask.any():
                            continue

                        batch_mask = mask.view(batch_size, seq_len)
                        expert_input = (
                            inps[batch_mask].view(-1, hidden_dim).to("cpu")
                        )
                        tick = time.time()
                        expert_output = self.run_expert_at_cpu(
                            i_layer, i_expert, expert_input
                        )
                        self.perf_stats["expert_compute-cpu"].append(
                            time.time() - tick
                        )
                        flat_mask = mask.view(-1)
                        weights = (
                            routing_weights[flat_mask]
                            .gather(
                                1,
                                (selected_experts[flat_mask] == i_expert)
                                .long()
                                .argmax(dim=1, keepdim=True),
                            )
                            .to("cpu")
                        )
                        expert_output = expert_output * weights
                        mask_index = mask.nonzero().squeeze(1)
                        cpu_results.append((mask_index, expert_output))
                    cpu_time = time.time() - start_time

                parallel_start = time.time()
                gpu_future = self._executor.submit(process_gpu_experts)
                cpu_future = self._executor.submit(process_cpu_experts)
                gpu_future.result()
                cpu_future.result()
                parallel_time = time.time() - parallel_start

                max_thread_time = max(gpu_time, cpu_time)
                parallel_degree = (
                    (gpu_time + cpu_time) / parallel_time
                    if parallel_time > 0
                    else 1.0
                )

                self.logger.log_layer_stats(
                    i_layer, gpu_time, cpu_time, parallel_time
                )

                if self.is_decode:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    stats_text = (
                        f"\n[{timestamp}] Layer {i_layer} Thread Time Stats:\n"
                    )
                    stats_text += (
                        f"[{timestamp}] GPU Thread Time: {gpu_time * 1000:.2f}ms\n"
                    )
                    stats_text += (
                        f"[{timestamp}] CPU Thread Time: {cpu_time * 1000:.2f}ms\n"
                    )
                    stats_text += f"[{timestamp}] Parallel Time: {parallel_time * 1000:.2f}ms\n"
                    stats_text += (
                        f"[{timestamp}] Parallel Degree: {parallel_degree:.2f}x\n"
                    )

                    os.makedirs("./log", exist_ok=True)
                    with open("./log/linshi.txt", "a") as f:
                        f.write(stats_text)

                # ===== 12. Merge results =====
                for mask_index, expert_output in gpu_results:
                    expert_output = expert_output.view(-1, hidden_dim)
                    inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                    inps_after_experts.index_add_(
                        0,
                        mask_index,
                        expert_output.to(inps_after_experts.dtype),
                    )
                inps_after_experts = inps_after_experts.view(
                    batch_size, seq_len, hidden_dim
                )

                for mask_index, expert_output in cpu_results:
                    expert_output = expert_output.view(-1, hidden_dim)
                    inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                    inps_after_experts.index_add_(
                        0,
                        mask_index.to(self.dev),
                        expert_output.to(self.dev).to(inps_after_experts.dtype),
                    )
                    inps_after_experts = inps_after_experts.view(
                        batch_size, seq_len, hidden_dim
                    )

                # addition because there's residual connection over moe layer
                total_expert_output = shared_output + inps_after_experts

                # 残差连接（恢复原始形状）
                inps = inps_residual + total_expert_output.reshape(
                    batch_size, seq_len, hidden_dim
                )
                # inps = inps_residual + inps_after_experts.reshape(batch_size, seq_len, hidden_dim)
                if self.is_decode:
                    layer_time = time.time() - layer_tick
                    layer_time_final = time.time() - laypid

                    self.layer_time_stats.append(
                        {
                            "layer_id": i_layer,
                            "time": layer_time,
                            "token_step": self.past_key_values_length,
                        }
                    )

                # end of one layer

                # 清理当前层已消费的预取列表
                if i_layer in self.prefetch_list:
                    del self.prefetch_list[i_layer]
                if i_layer in self.prefetching_list:
                    del self.prefetching_list[i_layer]

                if self.is_decode:
                    layer_time = time.time() - layer_tick
                    layer_times[i_layer] += layer_time
                    layer_times_mid[i_layer] += laymid
                    layer_times_final[i_layer] += layer_time_final
                    self.layer_time_accumulator[i_layer] += layer_time

                    # 记录到统计信息
        hot_experts = self.get_hot_expert()
        # for layer_id in hot_experts:
        # print(f"层 {layer_id} 热点专家: {hot_experts[layer_id][0]}")
        if self.is_decode:
            # 打印层时间统计
            # print("\n各层处理时间统计(ms):")
            for i in range(1, 27):
                # cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                layer_time = layer_times[i] * 1000
                layer_times_mids = layer_times_mid[i] * 1000
                layer_times_finals = layer_times_final[i] * 1000
                # cpu_ratio = (cpu_time / layer_time * 100) if layer_time > 0 else 0
                # print(f"层 {i}: {layer_time:.2f}ms (前向: {layer_times_fwd:.2f}ms, 后向: {layer_times_final.2f}ms)")
                # cpu_ratio = (cpu_time / layer_time * 100) if layer_time > 0 else 0

            # 计算并打印平均层时间
            avg_layer_time = sum(layer_times.values()) / self.n_layer
            total_cpu_time = sum(self.cpu_expert_time_per_layer.values())
            avg_cpu_ratio = (
                (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0
            )
            # print(f"平均每层时间: {avg_layer_time*1000:.2f}ms (CPU专家平均占比: {avg_cpu_ratio:.1f}%)")
            os.makedirs("./log", exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open("./log/expert_stats.txt", "a") as f:
                f.write(f"\n[{timestamp}] 各层处理时间统计(ms):\n")
                for i in range(1, 27):
                    cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                    layer_time = layer_times[i] * 1000
                    layer_time_mids = layer_times_mid[i] * 1000
                    layer_time_finals = layer_times_final[i] * 1000
                    # cpu_ratio = (cpu_time / layer_time * 100) if layer_time > 0 else 0
                    f.write(
                        f"[{timestamp}] 层 {i}: {layer_time:.2f}ms (, 中间层: {layer_time_mids:.2f}ms, 最终层: {layer_time_finals:.2f}ms, %)\n"
                    )
                    # f.write(f"层 {i}: {layer_time:.2f}ms (CPU专家: {cpu_time:.2f}ms, 占比: {cpu_ratio:.1f}%)\n")

                # f.write(f"平均每层时间: {avg_layer_time*1000:.2f}ms (CPU专家平均占比: {avg_cpu_ratio:.1f}%)\n")
        other_ops_start = time.time()
        # print("ok")
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start
            # print(f"\nToken处理时间统计:")
            # print(f"总时长: {total_decode_time*1000:.2f}ms")
            # print(f"  - {self.n_layer}层处理: {layer_total_time*1000:.2f}ms (平均每层: {layer_total_time*1000/self.n_layer:.2f}ms)")
            # print(f"  - 其他操作(norm+lm_head): {other_ops_time*1000:.2f}ms")
        self.present_key_value = present_key_value
        return lm_logis

        # def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        # return self.model.layers[i_layer].mlp.experts[i_expert](
        #     inps, routing_weights
        # )

    def run_expert_at_gpu(self, i_layer, i_expert, inps):
        """
        在GPU上运行专家计算

        参数:
            i_layer: 层索引
            i_expert: 专家索引
            inps: 输入tensor [token_count, hidden_dim]

        返回:
            expert_output: 专家输出

        说明:
            - 专家状态分类:
              - normal: 处理1个token
              - hot: 处理2个token
              - veryhot: 处理>2个token
        """
        start_time = time.time()

        # 执行专家计算
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        torch.cuda.synchronize()  # 确保GPU操作完成
        elapsed = time.time() - start_time

        # 统计处理的token数量，判断热点状态
        token_count = inps.shape[0]
        expert_status = "normal"
        if token_count > 2:
            expert_status = "veryhot"  # 非常热点
        elif token_count > 1:
            expert_status = "hot"  # 热点

        # 记录当前迭代的专家统计
        self.current_iter_expert_stats[i_layer]["expert_ids"].append(i_expert)
        self.current_iter_expert_stats[i_layer]["token_counts"].append(token_count)

        # 记录专家处理时间和状态
        self.expert_time_stats.append(
            {
                "layer_id": i_layer,
                "expert_id": i_expert,
                "time": elapsed,
                "device": "gpu",
                "token_count": token_count,
                "status": expert_status,
            }
        )
        return result

    def run_expert_at_cpu(self, i_layer, i_expert, inps):
        """
        在CPU上运行专家计算

        参数:
            i_layer: 层索引
            i_expert: 专家索引
            inps: 输入tensor [token_count, hidden_dim]

        返回:
            expert_output: 专家输出

        说明:
            - 与GPU版本类似，但记录CPU处理时间
            - 累计每层的CPU处理时间用于性能分析
        """
        start_time = time.time()

        # 执行专家计算 (在CPU上)
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # 累计CPU处理时间
        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed

        # 统计token数量和热点状态
        token_count = inps.shape[0]
        expert_status = "normal"
        if token_count > 2:
            expert_status = "veryhot"
        elif token_count > 1:
            expert_status = "hot"

        self.current_iter_expert_stats[i_layer]["expert_ids"].append(i_expert)
        self.current_iter_expert_stats[i_layer]["token_counts"].append(token_count)

        self.expert_time_stats.append(
            {
                "layer_id": i_layer,
                "expert_id": i_expert,
                "time": elapsed,
                "device": "cpu",
                "token_count": token_count,
                "status": expert_status,
            }
        )
        return result

    def get_expert_stats(self):
        """
        获取专家热度统计信息

        返回:
            dict: 包含以下键的统计字典
                - hot_experts: 每层热点专家统计
                - hot_counts: 热点专家数量分布
                - token_distribution: token数量分布
        """
        stats = {
            "hot_experts": {
                i: {"count": 0, "hot": 0, "veryhot": 0} for i in range(1, 27)
            },
            "hot_counts": {2: 0, 3: 0, 4: 0, 5: 0},
            "token_distribution": {},
        }

        # 按层统计hot/veryhot专家
        for record in self.expert_time_stats:
            layer = record["layer_id"]
            token_count = record["token_count"]
            expert_status = record["status"]

            stats["hot_experts"][layer]["count"] += 1
            if expert_status == "hot":
                stats["hot_experts"][layer]["hot"] += 1
            elif expert_status == "veryhot":
                stats["hot_experts"][layer]["veryhot"] += 1

        # 统计hot专家数量分布
        layer_hot_counts = {i: 0 for i in range(1, 27)}
        for record in self.expert_time_stats:
            if record["status"] in ["hot", "veryhot"]:
                layer_hot_counts[record["layer_id"]] += 1

        for count in layer_hot_counts.values():
            if count >= 2 and count <= 5:
                stats["hot_counts"][count] += 1

        # 重置token分布统计，只统计当前迭代
        stats["token_distribution"] = {}
        token_counts = [r["token_count"] for r in self.expert_time_stats]
        unique_counts = set(token_counts)

        for count in unique_counts:
            stats["token_distribution"][count] = token_counts.count(count)

        return stats
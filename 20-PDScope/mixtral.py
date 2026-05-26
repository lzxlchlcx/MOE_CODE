import copy
import threading
import time
import os  # 【PDScope新增】Fiddler 无此导入，用于读取 hot 专家文件和写日志
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers

class FiddlerMixtral:
    """
    PDScope 推理引擎 — 在 Fiddler 基础上增加了以下关键机制：
    
    【与 Fiddler 的核心差异】
    1. 6 个占位专家（Fiddler 只有 1 个），支持预取和 ondemand 并行
    2. TG/TC 贪心调度（Fiddler 用 2^8=256 暴力枚举）
    3. 三线程并行：GPU线程 + CPU线程 + 预取线程（Fiddler 串行执行）
    4. 下一层专家预测 + 异步预取（Fiddler 无预取）
    5. 多 batch 支持（Fiddler 仅支持单条/beam search）
    6. 实测 cpu_time_table 查表（Fiddler 用固定 latency_cpu/latency_gpu）
    
    【Fiddler 对照参考】Fiddler 的核心创新：将激活值传到 CPU 执行专家计算，
    而非将权重搬到 GPU，因为激活值(batch×4096)远小于权重(3×4096×14336)。
    PDScope 在此基础上增加了多级缓存和预测预取机制。
    """

    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        # 【PDScope新增】每层预测的热门专家列表，用于异步预取。Fiddler 无此概念。
        # 格式: {layer_id: [expert_id, ...]}，在 mixtral_forward 中由下一层路由预测填充
        self.hot_experts = {}

        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            use_cache=True,
        )
        # 【PDScope新增】batch_size 和 cache（预取数量）两个参数
        # Fiddler 只有 beam_width，不支持多 batch 推理
        # cache 控制预测下一层时取 top-k 个候选专家用于预取
        self.cache = args.cache
        self.batch_size = args.batch_size  # 【PDScope新增】Fiddler 不支持多batch
        # 根据 batch_size 调整预取数量：batch 越小，GPU 显存越充裕，可预取更多
        if self.batch_size == 4:
            self.cache = 5
        else:
            self.cache = 7
        self.lm_head = self.model.lm_head
        self.model = self.model.model

        # ===== 【PDScope核心差异】6 个 GPU 占位专家（Fiddler 只有 1 个 expert_placeholder）=====
        #
        # 设计思路：
        # - 4 个主 placeholder（预取用）：按奇偶层分离管理，避免相邻层争用
        #   偶数层用 placeholder/placeholder2，奇数层用 placeholder3/placeholder4
        # - 2 个 ondemand placeholder：用于实时加载当前层需要的非预取专家
        #
        # Fiddler 只有 1 个 placeholder，每次 load_state_dict 会覆盖前一次的权重，
        # 无法并行加载多个专家，也无法区分"正在预取"和"已完成预取"。
        # PDScope 的 6 个 placeholder 支持预取线程和 GPU 计算线程并行工作。
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.expert_placeholder2 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[1]
        ).to(self.dev)
        self.expert_placeholder3 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[2]
        ).to(self.dev)
        self.expert_placeholder4 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[3]
        ).to(self.dev)
        self.prefil_pre = False  # prefill 阶段预取标记

        # 2 个 ondemand 专用 placeholder：用于实时加载当前层需要的非预取专家
        # 【PDScope新增】Fiddler 完全没有 ondemand 概念
        # ondemand 指在 TG/TC 贪心调度中被判定为"值得加载到 GPU"的专家，
        # 通过 _async_ondemand 将权重从 CPU 复制到这两个 placeholder 中执行
        self.expert_placeholder5 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[5]
        ).to(self.dev)
        self.expert_placeholder6 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[6]
        ).to(self.dev)
        self.expert_placeholder5_inused = False
        self.expert_placeholder6_inused = False

        # 【PDScope新增】placeholder 管理基础设施（Fiddler 完全没有）
        # expert_to_placeholder: 反向映射，从 (layer, expert) 查找对应的 placeholder 对象
        # placeholder_to_expert: 正向映射，从 placeholder 名称查找其当前装载的 (layer, expert)
        # expert_loading_status: placeholder 的加载状态，True=正在加载中，用于等待/同步
        # placeholder_lock: 多线程分配 placeholder 时用的互斥锁（预取线程 vs GPU线程）
        self.expert_to_placeholder = {}  # (layer, expert) → placeholder对象
        self.expert_placeholder_inused = False
        self.expert_placeholder2_inused = False
        self.expert_placeholder3_inused = False
        self.expert_placeholder4_inused = False
        self.placeholder_lock = threading.Lock()  # 多线程分配锁
        self.prefetch_layers = 0  # 当前预取目标层
        self.is_decode = False  # 【PDScope新增】标记当前是 prefill 还是 decode 阶段
        # Fiddler 将 is_decode 作为 mixtral_forward 的参数传入，
        # PDScope 改为实例变量，方便多线程和各方法共享状态

        # 【PDScope新增】placeholder名 → (layer, expert) 映射
        # 用于查找某个 placeholder 里当前装载的是哪一层的哪个专家
        # Fiddler 只有 1 个 placeholder，无需映射
        self.placeholder_to_expert = {
            'expert_placeholder': None,
            'expert_placeholder2': None,
            'expert_placeholder3': None,
            'expert_placeholder4': None
        }
        # 【PDScope新增】placeholder 加载状态：True=正在加载中（不可用），False=已就绪（可用）
        # 预取线程开始加载时设为 True，加载完成后设为 False
        # GPU 线程在执行前会检查此状态，避免使用半加载的权重
        # 初始值为 True，表示刚开始时所有 placeholder 都"不可用"
        self.expert_loading_status = {
            'expert_placeholder': True,
            'expert_placeholder2': True,
            'expert_placeholder3': True,
            'expert_placeholder4': True
        }

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)  # Mixtral-8x7B: 32层
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)  # 8个专家

        # ===== 【PDScope新增】大量性能统计基础设施（Fiddler 只有 cnt_expert_hit/cnt_expert_all）=====
        # 这些统计用于：1) 分析预取命中率 2) 各层耗时分析 3) 专家热度分布
        self.expert_selection_stats = []  # 每层每步的专家选择记录
        self.expert_time_stats = []       # 每个专家的计算耗时
        self.prefetch_list = {}           # 已完成预取的专家: {layer: [expert_id, ...]}
        self.prefetching_list = {}        # 正在预取中的专家: {layer: [expert_id, ...]}
        self.expert_selection_history = {}  # 历史专家选择（记录每次迭代的路由结果）
        self.hit_stats = {}               # 每层命中统计: {'hits': n, 'total': m}
        for i in range(self.n_layer):
            self.expert_selection_history[i] = []
            self.hit_stats[i] = {'hits': 0, 'total': 0}

        # 【未使用】路由权重累积器，仅供分析
        self.expert_weight_accumulator = {}
        for i in range(self.n_layer):
            self.expert_weight_accumulator[i] = torch.zeros(8, device=self.dev)

        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(self.n_layer)}

        # 延迟参数
        # 【对比差异】Fiddler: latency_cpu=7（每token的CPU延迟）, latency_gpu=70（权重搬运固定延迟）
        # PDScope 在 mixtral_forward 中实际使用 e=27（单次GPU加载延迟）, tg=4.22（token级GPU开销）
        # 和 cpu_time_table 查表（micromixtral.txt 中预实测的 CPU 延迟表），不再用这两个固定值
        self.latency_cpu = 5
        self.latency_gpu = 45

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()
        #expert location table: 1 表示该专家常驻 GPU，0 表示常驻 CPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)  # 【PDScope新增但未使用】
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)
        
        # ===== 【PDScope新增】分层耗时统计（Fiddler 无此基础设施）=====
        # 表1: layer_time_accumulator — 每层 decode 总耗时累加器
        # 用途: 在 mixtral_forward 每层执行后累加耗时，generate() 末尾除以次数得每层平均耗时
        # 消费方: generate() L799-801 → layer_time_avg
        self.layer_time_accumulator = {}  
        for i in range(self.n_layer):
            self.layer_time_accumulator[i] = 0.0


        # 表2: layer_time_details — 按调度场景(all_gpu/all_cpu/mixed)记录每层耗时明细
        # 每个场景对应一个列表，存储 {layer_id, time, token_step} 字典
        # 用途: 分析不同专家调度策略下各层的性能差异
        # 消费方: generate() L797 返回给调用方；L803-808 计算 layer_time_avg_details
        self.layer_time_details = {
            'all_gpu': [],    
            'all_cpu': [],    
            'mixed': []       
        }


        # 表3: layer_time_accumulator_details — 按场景分别累加每层耗时
        # 用途: 与 layer_time_accumulator 类似，但区分三种场景，计算每种场景下每层平均耗时
        # 消费方: generate() L803-808 → layer_time_avg_details
        # NOTE:搜索整个文件只有初始化 (L198) 和读取 (L829)，没有 += 赋值。所以它实际上是死代码
        self.layer_time_accumulator_details = {
            'all_gpu': {i: 0.0 for i in range(self.n_layer)},
            'all_cpu': {i: 0.0 for i in range(self.n_layer)},
            'mixed': {i: 0.0 for i in range(self.n_layer)}
        }


        # 表4: last_iter_expert_stats — 上一轮 decode 各层专家的 {expert_ids, token_counts}
        # 【注意】当前为死代码：get_hot_expert() 的返回值未被任何代码消费，
        # 且 last_iter_expert_stats 本身也没有被读取。
        # 真正驱动 prefetch 的是 self.hot_experts（基于当前层 gate 即时分析，L1165-1189）。
        self.last_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []} 
            for i in range(self.n_layer)
        }
        # 表5: current_iter_expert_stats — 当前轮 decode 各层专家的 {expert_ids, token_counts}
        # 填充方: run_expert_at_gpu (L1764-1765), run_expert_at_cpu (L1803-1804)
        # 消费方: get_hot_expert() 按 token_counts 降序排序
        # 【注意】当前为死代码：get_hot_expert() 返回值 (L1695) 未被任何代码使用。
        # 与 self.hot_experts (L33, L1165-1189) 是两套独立的热度机制，
        # self.hot_experts 基于当前层 gate 实时统计，才是实际驱动 prefetch 的数据源。
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(self.n_layer)
        }


        # 表5: current_iter_expert_stats — 当前轮 decode 各层专家的 {expert_ids, token_counts}
        # 用途: 实时记录每层各专家被路由的 token 数，用于热度排名
        # 填充方: run_expert_at_gpu (L1726-1727), run_expert_at_cpu (L1765-1766)
        # 消费方: get_hot_expert() 按 token_counts 降序排序 → 返回 hot_experts → prefetch 决策
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(self.n_layer)
        }

        self.layer_data = {}  
        
        # 【PDScope新增】专家选择计数矩阵(n_layer × n_expert)，用于热度分析
        self.expert_selection_count = np.zeros((self.n_layer, self.n_expert), dtype=int)
        tick = time.time()        
        # 【与 Fiddler 相同】将标记为 GPU 的专家实际加载到 GPU 显存
        # 但 PDScope 版本增加了 OOM 异常捕获
        self.bring_expert_to_gpu()
        
        print("Model is ready.")



    def bring_non_expert_to_gpu(self):
        """将非专家层加载到 GPU。与 Fiddler 基本相同。
        包括 lm_head、embed_tokens、norm、每层的 self_attn、
        input_layernorm、gate（路由器）、post_attention_layernorm。
        专家层保留在 CPU，后续按需加载。
        """
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            

    def get_hot_expert(self):
        """【PDScope新增】收集当前迭代的专家热度排名。
        根据 current_iter_expert_stats 中记录的各专家被路由的 token 数排序，
        将上一轮的统计保存到 last_iter_expert_stats，然后清空当前轮。
        返回 {layer_id: [expert_id, ...]} 按热度降序。
        Fiddler 无此功能——它不做运行时专家热度分析。
        """
        
        if not hasattr(self, 'is_decode') or not self.is_decode:
            return {}
        
        hot_experts = {}
        
        for layer_id in range(self.n_layer):
            
            expert_ids = self.current_iter_expert_stats[layer_id]['expert_ids']
            token_counts = self.current_iter_expert_stats[layer_id]['token_counts']
            
            
            expert_data = list(zip(expert_ids, token_counts))
            
            
            sorted_experts = sorted(expert_data, key=lambda x: x[1], reverse=True)
            
            
            hot_experts[layer_id] = [expert[0] for expert in sorted_experts]
            
            
            self.last_iter_expert_stats[layer_id] = {
                'expert_ids': expert_ids.copy(),
                'token_counts': token_counts.copy()
            }
            
            
            self.current_iter_expert_stats[layer_id]['expert_ids'].clear()
            self.current_iter_expert_stats[layer_id]['token_counts'].clear()
        
        
        
        return hot_experts
    
    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """根据热度排序将前 n_expert_on_gpu 个专家标记为 GPU 常驻。
        
        【对比差异】Fiddler 只使用硬编码的 popular_experts 列表（368个，覆盖所有256个专家）。
        PDScope 增加了从文件 './hot/mix.txt' 动态加载的能力，
        但默认 fallback 列表只列出了 82 个专家（Fiddler 列出完整 368 个），
        这意味着 PDScope 的 GPU 常驻专家列表更短/更精简。
        """
        if popular_experts is None:
            # 【PDScope新增】优先从文件加载热专家列表，支持运行时动态配置
            # Fiddler 直接使用下方硬编码列表
            
            hot_experts_file = './hot/mix.txt'
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, 'r') as f:
                        popular_experts = [tuple(map(int, line.strip().split(','))) 
                                         for line in f if line.strip()]
                    print(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    print(f"Error loading hot experts: {e}")
            else:
                popular_experts = [
                    (9, 5),
                    (11, 2),
                    (10, 4),
                    (28, 0),
                    (13, 1),
                    (17, 7),
                    (12, 1),
                    (8, 6),
                    (16, 1),
                    (9, 0),
                    (14, 5),
                    (19, 5),
                    (26, 2),
                    (30, 7),
                    (7, 1),
                    (3, 7),
                    (23, 4),
                    (22, 1),
                    (29, 3),
                    (1, 5),
                    (13, 0),
                    (5, 1),
                    (18, 0),
                    (4, 7),
                    (10, 3),
                    (1, 2),
                    (3, 0),
                    (8, 3),
                    (11, 0),
                    (11, 5),
                    (11, 1),
                    (31, 4),
                    (21, 0),
                    (25, 1),
                    (15, 5),
                    (22, 4),
                    (27, 5),
                    (16, 7),
                    (15, 1),
                    (13, 2),
                    (15, 4),
                    (21, 1),
                    (27, 7),
                    (9, 7),
                    (7, 4),
                    (31, 5),
                    (2, 1),
                    (11, 6),
                    (12, 3),
                    (2, 4),
                    (24, 2),
                    (28, 2),
                    (0, 2),
                    (30, 2),
                    (6, 0),
                    (6, 7),
                    (15, 6),
                    (6, 2),
                    (14, 2),
                    (2, 0),
                    (17, 2),
                    (19, 2),
                    (24, 0),
                    (10, 0),
                    (19, 4),
                    (1, 4),
                    (26, 3),
                    (31, 7),
                    (17, 6),
                    (25, 3),
                    (12, 6),
                    (0, 0),
                    (26, 0),
                    (29, 7),
                    (27, 2),
                    (19, 6),
                    (5, 0),
                    (18, 2),
                    (20, 1),
                    (12, 4),
                    (17, 5),
                    (5, 4),
                    (30, 6),
                    (20, 5),
                ]

        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1
    

    def _async_ondemand(self, layer_idx, expert_id, target_placeholder):
        """【PDScope新增】ondemand 加载：将 CPU 专家权重异步复制到指定的 GPU placeholder。
        
        与 _async_load_expert 的区别：
        - _async_ondemand 接受外部指定的 target_placeholder（通常是 placeholder5/6）
        - _async_load_expert 内部自行分配 placeholder（1-4，按奇偶层）
        
        两者都使用 pin_memory + copy_ 进行 CPU→GPU 的权重搬运。
        Fiddler 使用 load_state_dict 进行同步加载（只有1个 placeholder，无需异步）。
        """
        expert = self.model.layers[layer_idx].block_sparse_moe.experts[expert_id]
    
        
        if next(expert.parameters()).is_cuda:
            return 
        
        
        for name in ['w1', 'w2', 'w3']:
            w = getattr(self.model.layers[layer_idx].block_sparse_moe.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data 
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned

        tick = time.time()
        for name in ['w1', 'w2', 'w3']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].block_sparse_moe.experts[expert_id], name).weight.data
            dst.copy_(src)
            
        copytime = time.time() - tick
        


    def _async_load_expert(self, layer_idx, expert_id):
        """【PDScope新增】异步加载专家到预取 placeholder（1-4）。
        
        核心逻辑：
        1. 若专家已在 GPU 则跳过
        2. pin_memory 固定 CPU 权重（加速后续 copy_）
        3. 按奇偶层分配 placeholder：奇数层→3/4，偶数层→1/2
        4. copy_ 将权重从 CPU 搬到 GPU placeholder
        5. 更新双向映射和加载状态
        
        Fiddler 的 load_state_dict 是同步的，只加载到唯一一个 placeholder。
        """
        expert = self.model.layers[layer_idx].block_sparse_moe.experts[expert_id]
    
        
        if next(expert.parameters()).is_cuda:
            return 
        
        for name in ['w1', 'w2', 'w3']:
            w = getattr(self.model.layers[layer_idx].block_sparse_moe.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data 
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned
        
        target_placeholder = None
        
        if layer_idx % 2 == 1:
            if not self.expert_placeholder3_inused:
                target_placeholder = self.expert_placeholder3
                self.expert_placeholder3_inused = True
                self.expert_loading_status['expert_placeholder3'] = True
            elif not self.expert_placeholder4_inused:
                target_placeholder = self.expert_placeholder4
                self.expert_placeholder4_inused = True
                self.expert_loading_status['expert_placeholder4'] = True
        
        else:
            if not self.expert_placeholder_inused:
                target_placeholder = self.expert_placeholder
                self.expert_placeholder_inused = True
                self.expert_loading_status['expert_placeholder'] = True
            elif not self.expert_placeholder2_inused:
                target_placeholder = self.expert_placeholder2
                self.expert_placeholder2_inused = True
                self.expert_loading_status['expert_placeholder2'] = True
        
        if target_placeholder is None:
            raise RuntimeError("No available expert placeholder")

        
        if target_placeholder == self.expert_placeholder:
            self.placeholder_to_expert['expert_placeholder'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder2:
            self.placeholder_to_expert['expert_placeholder2'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder3:
            self.placeholder_to_expert['expert_placeholder3'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder4:
            self.placeholder_to_expert['expert_placeholder4'] = (layer_idx, expert_id)

        
        tick = time.time()
        for name in ['w1', 'w2', 'w3']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].block_sparse_moe.experts[expert_id], name).weight.data
            dst.copy_(src)
        
        
        if target_placeholder == self.expert_placeholder:
            self.expert_loading_status['expert_placeholder'] = False
        elif target_placeholder == self.expert_placeholder2:
            self.expert_loading_status['expert_placeholder2'] = False
        elif target_placeholder == self.expert_placeholder3:
            self.expert_loading_status['expert_placeholder3'] = False
        elif target_placeholder == self.expert_placeholder4:
            self.expert_loading_status['expert_placeholder4'] = False
            
        copytime = time.time() - tick
        
        self.expert_to_placeholder[(layer_idx, expert_id)] = target_placeholder
    def is_expert_loading(self, placeholder_name):
        """【PDScope新增】检查指定 placeholder 是否正在加载中。"""
        
        return self.expert_loading_status.get(placeholder_name, False)
    def is_expert_loaded(self, layer_id, expert_id):
        """【PDScope新增】检查专家是否已完成预取加载（不在 prefetching_list 中）。"""
        
        
            
        return (layer_id not in self.prefetching_list or 
                expert_id not in self.prefetching_list[layer_id])     
    def release_placeholder(self, layer_idx, expert_id):
        """【PDScope新增】释放已用完的 placeholder。
        当当前层已经过了某个 placeholder 所装载的专家所在层时，
        释放该 placeholder 以便后续层复用。
        条件：stored_expert[0] < layer_idx 或循环到第0层时释放第31层的。
        Fiddler 无需释放——它只有1个 placeholder，每次 load_state_dict 直接覆盖。
        """
        for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                              'expert_placeholder3', 'expert_placeholder4']:
            stored_expert = self.placeholder_to_expert[placeholder_name]
            if stored_expert and (stored_expert[0] < layer_idx or 
                    (stored_expert[0] == self.n_layer - 1 and layer_idx <= 1)):
                
                setattr(self, f"{placeholder_name}_inused", False)
                self.placeholder_to_expert[placeholder_name] = None

    def bring_expert_to_gpu(self):
        """将 expert_loc 中标记为 GPU 的专家实际加载到 GPU 显存。
        【对比差异】Fiddler 直接遍历加载，无异常处理。
        PDScope 增加了 OOM 捕获（try/except），但在当前实现中捕获后仍 raise。
        """
        expert_count = 0
        try:
            for i in range(self.n_layer):
                for j in range(self.n_expert):
                    if self.is_expert_in_gpu(i, j):
                        self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)
                        expert_count += 1
                        #NOTE 【注意】这里没有乘以 routing_weights，与 Fiddler 不同
                        # Fiddler: expert(current_state, routing_weights[...])
                        # PDScope: placeholder(current_state) ← 缺少权重乘法
                        # 可能是因为专家 forward 中已包含权重处理，或是遗漏
                
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                
                
                    
                
            
                raise  
    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1
    def is_expert_in_gpu_now(self, i_layer, i_expert):
        """【PDScope新增】检查专家是否实际在 GPU 上（通过检查参数设备）。
        与 is_expert_in_gpu 的区别：
        - is_expert_in_gpu: 查 expert_loc 静态表（初始化时设定，后续不变）
        - is_expert_in_gpu_now: 检查参数实际的 device（反映预取/ondemand等动态变化）
        Fiddler 只有 is_expert_in_gpu，因为它没有运行时动态加载。
        """
        
        
        
        
        
        
        
                
        
        expert = self.model.layers[i_layer].block_sparse_moe.experts[i_expert]
        return next(expert.parameters()).is_cuda


    def calc_n_expert_on_gpu(self):
        """根据 GPU 剩余显存计算能容纳多少个专家。
        
        【对比差异】Fiddler 动态计算: free_mem // (n_param * 2)，通用但需确保显存足够。
        PDScope 使用硬编码值，根据 batch_size 返回不同数值：
        - batch=64 → 62, batch=32 → 70, else → 74
        batch 越大，KV Cache 占用越多，留给专家的显存越少。
        保留了 Fiddler 的参数统计代码（n_param）但仅用于打印。
        """
        
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        print(f"Number of parameters in a single expert: {n_param}")
        
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev) 
        if self.batch_size==64:
            return 62
        elif self.batch_size==32:
            return 70
        else:
            return 74


    def initial_beam_tensor(self, input_tensor):
        """在 beam search 第一步中，将 topk 结果正确转换为 (beam_width, 1)。
        与 Fiddler 相同。"""
        
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor
    
    def _process_single_cpu_expert(self, i_layer, i_expert, combined_inps, combined_weights, combined_selected, expert_mask, combined_mask):
        """【PDScope新增】处理单个 CPU 专家的辅助方法（当前未在主流程中调用）。"""
        expert_input = combined_inps[expert_mask]
        
        tick = time.time()
        expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input)
        self.perf_stats['expert_compute-cpu'].append(time.time() - tick)
        
        expert_weights = combined_weights[expert_mask].gather(
            1, (combined_selected[expert_mask] == i_expert).long().argmax(dim=1, keepdim=True)
        )
        expert_output = expert_output * expert_weights
        
        mask_index = combined_mask.nonzero().squeeze(1)[expert_mask.nonzero().squeeze(1)]
        return mask_index, expert_output        
     
    def generate_heatmap(self):
        """【PDScope新增】将专家选择计数导出为 CSV 热力图。Fiddler 无此功能。"""
        
        import numpy as np
        import os
        os.makedirs('./log', exist_ok=True)

        np.savetxt('./log/expert_selection_count.csv', 
                  self.expert_selection_count, 
                  delimiter=',',
                  fmt='%d')
        
        np.savetxt('./log/expert_high_weight_count.csv',
                  self.expert_high_weight_count,
                  delimiter=',',
                  fmt='%d')

        return self.expert_selection_count

    def generate(self, text=None, output_token=20, input_token=None):
        """文本生成主循环。
        
        【对比 Fiddler 的 generate】
        1. Fiddler: 单条文本输入，复制 beam_width 份做 beam search
           PDScope: 支持 batch_size 条并行输入（多 batch 推理）
        2. Fiddler: tokenize 返回 (input_ids, position_ids)
           PDScope: tokenize 返回 (input_ids, position_ids, attention_mask)，
                    attention_mask 支持 padding 和 KV Cache 增量扩展
        3. Fiddler: 返回 (prefill_time, decode_time, hit_rate) 三元组
           PDScope: 返回四元组，第四个是包含大量性能统计的 dict
        4. PDScope 增加了 torch.profiler 集成和逐 token 计时
        """
        torch.set_num_threads(16) 
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 1
        
        self.expert_selection_stats = []
        self.expert_time_stats = []
        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        input_ids, position_ids, attention_mask = self.tokenize(text,input_token)
        
        if input_token is not None:
            
            input_ids = torch.stack([
                ids[:input_token] if len(ids) > input_token else ids 
                for ids in input_ids
            ])
            position_ids = torch.stack([
                pos[:input_token] if len(pos) > input_token else pos
                for pos in position_ids
            ])
            
            attention_mask = attention_mask[:, :, :, :input_token]
        
        tick = time.time()
        self.is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        self.token_decode_times = []
        # 【PDScope新增】perf_stats 用于记录各阶段耗时（token_embedding, self_attention, moe_gating, expert_compute 等）
        # Fiddler 只有全局的 prefill/decode 时间

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,  
                warmup=3,  
                active=1,  
                repeat=1  
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        for i_token in range(output_token):
            
            token_start_time = time.time()  
            
            if self.is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])
            
            if self.is_decode:
                
                new_mask = torch.ones(
                    (attention_mask.shape[0], attention_mask.shape[1], 1, 1),
                    dtype=torch.bool,
                    device=self.dev
                )
                
                attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            ).unsqueeze(0).expand(input_ids.shape[0], -1)         
            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask )
            # 【对比差异】Fiddler: mixtral_forward(input_ids, position_ids, is_decode)
            # PDScope: mixtral_forward(input_ids, position_ids, attention_mask)
            # PDScope 传入 attention_mask 以支持多 batch padding 和 KV Cache 增量更新
            # is_decode 改为实例变量 self.is_decode

            logits = logits.to("cpu")
            logits = F.softmax(logits, dim=-1)

            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            
            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            
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
            
            if not self.is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            self.is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")
        
        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
            {
                'perf_stats': self.perf_stats,
                'expert_selection': self.expert_selection_stats,
                'expert_time': self.expert_time_stats,
                'layer_time': self.layer_time_stats,
                'outputs': decode_strings,  
                'layer_time_details': self.layer_time_details,  
                'expert_hot_stats': self.get_expert_stats(), 
                'layer_time_avg': {
                    i: self.layer_time_accumulator[i] / max(1, len([x for x in self.layer_time_stats if x['layer_id'] == i]))
                    for i in range(self.n_layer)
                },
                'layer_time_avg_details': {  
                    case: {
                        i: self.layer_time_accumulator_details[case][i] / max(1, len([x for x in self.layer_time_details[case] if x['layer_id'] == i]))
                        for i in range(self.n_layer)
                    }
                    for case in ['all_gpu', 'all_cpu', 'mixed']
                }
            }
        )
    def tokenize(self, text, input_token):
        """将输入文本 tokenize。
        
        【对比差异】
        Fiddler: 单条文本 → 复制 beam_width 份 → pad_sequence 拼接
                 返回 (input_ids, position_ids)，无 attention_mask
        PDScope: 支持多条文本列表（batch_size 条），padding 到相同长度
                 返回 (input_ids, position_ids, attention_mask)
                 attention_mask 扩展为 (batch, 1, 32, seq_len) 以适配 Flash Attention
        """
        # 将单条文本转为列表，统一后续处理逻辑
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")
        
        # 补齐/截断文本数量到 batch_size，保证输入维度一致
        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[:self.batch_size]

        # tokenize + padding：将不定长文本填充到相同长度，用于 batch 推理
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt"
        )
        # 将 token ids 和 mask 移到 GPU
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.bool().to(self.dev)

        # 生成位置 id：[0, 1, ..., seq_len-1]，广播到 batch 维度
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # 将 2D mask (batch, seq_len) 扩展为 4D (batch, 1, 32, seq_len)
        # 以适配 Flash Attention 的 attention_mask 格式要求
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(-1, 32, -1, -1)
        
        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask ):
        """自定义前向传播——PDScope 的核心调度引擎。
        
        【与 Fiddler 的核心差异】
        Fiddler 的 mixtral_forward 流程：
          1. 遍历8个专家，统计每个专家的 token 数
          2. 用固定 latency_cpu/latency_gpu 估算 CPU/GPU 延迟
          3. 暴力枚举 2^8=256 种 CPU/GPU 分配方案，选最小总延迟
          4. 先串行执行 GPU 组，再串行执行 CPU 组
        
        PDScope 的 mixtral_forward 流程：
          1. 统计非 GPU 常驻专家的 token 数，按热度排序
          2. 用实测 cpu_time_table 查表 + TG/TC 贪心算法决定哪些专家 ondemand 加载
          3. 用下一层路由预测热门专家，异步预取到 placeholder 1-4
          4. 将专家分为5类：gpu常驻 / placeholder已预取 / 预取加载中 / remaining(ondemand) / cpu
          5. 三线程并行：GPU线程(含4个子线程) + CPU线程 + 预取线程
        
        参数 attention_mask 是 PDScope 新增的，Fiddler 用 is_decode 布尔参数。
        """
        hidden_dim = self.model.config.hidden_size
        tick = time.time()
        
        # Token Embedding —— 与 Fiddler 相同
        inps = self.model.embed_tokens(input_ids)
        self.perf_stats['token_embedding'].append(time.time() - tick)
        
        # 【PDScope新增】decode 阶段初始化分层计时器
        if self.is_decode:
            total_decode_start = time.time()
            layer_times = {i: 0.0 for i in range(self.n_layer)}
            layer_times_fwd = {i: 0.0 for i in range(self.n_layer)}        
            layer_times_mid = {i: 0.0 for i in range(self.n_layer)}       
            layer_times_final = {i: 0.0 for i in range(self.n_layer)}                  
        
        # 【PDScope新增】RoPE 位置编码，Fiddler 直接传 position_ids 给 self_attn
        # PDScope 先计算 position_embeddings，再传给 self_attn 的 position_embeddings 参数
        # 这是因为 transformers 库版本差异，新版 Mixtral 需要 pre-computed RoPE
        position_embeddings = self.model.rotary_emb(inps, position_ids)
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        
        
        layer_start_time = time.time()        
        layer_total_time = 0.0
        isprefetch=False  # 【PDScope新增但未使用】

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()            

            # 【PDScope新增】每层开始时释放已用完的 placeholder
            # 当执行到第 i_layer 层时，placeholder 中装载的 < i_layer 层的专家已无用了
            # Fiddler 无此操作——它每次 load_state_dict 直接覆盖唯一 placeholder
            self.release_placeholder(i_layer, 0)
            
            laymid = time.time() - layer_tick             
            # ===== Self-Attention 部分（与 Fiddler 类似，但 API 有差异）=====
            # Fiddler: layer.self_attn(inps, position_ids=..., past_key_value=..., use_cache=True)
            #          返回 (attn_output, self_attn_weights, present_key_value) 三元组
            # PDScope: layer.self_attn(hidden_states=inps, attention_mask=..., 
            #          position_embeddings=..., past_key_value=..., use_cache=True)
            #          使用关键字参数，新增 attention_mask 和 position_embeddings
            #          返回值可能是 2 元组或 3 元组，需灵活处理
            self.cpu_expert_time_per_layer[i_layer] =0
            inps_residual = inps
            inps = layer.input_layernorm(inps)

            inps = inps.view(batch_size, seq_len, hidden_dim)            
            tick = time.time()
            attn_output = layer.self_attn(
                hidden_states=inps,
                attention_mask=attention_mask,  # 【PDScope新增】支持多 batch padding mask
                position_embeddings=position_embeddings,  # 【PDScope新增】替代 position_ids
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            

            torch.cuda.synchronize()  # 【PDScope新增】确保 GPU 操作完成，精确计时
            self.perf_stats['self_attention'].append(time.time() - tick)
            
            # 【PDScope新增】灵活处理 self_attn 返回值（2元组或3元组）
            # Fiddler 直接解包为三元组: inps, self_attn_weights, present_key_value
            # PDScope 更防御性地处理不同返回格式
            if isinstance(attn_output, tuple):
                if len(attn_output) == 2:  
                    inps, present_key_value = attn_output
                    self_attn_weights = None
                else:  
                    inps, self_attn_weights, present_key_value = attn_output
            else:  
                inps = attn_output
                self_attn_weights = None
                present_key_value = None
            
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            
            layer_idx=i_layer
            if layer_idx not in self.layer_data:
                self.layer_data[layer_idx] = {
                    "hidden_states": [],
                    "expert_indices": []
                }
            
            
            pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)
            
            # ===== MoE (Mixture of Experts) 路由与调度 =====
            # 以下代码是 PDScope 与 Fiddler 差异最大的部分
            
            tick = time.time()
            router_logits = layer.block_sparse_moe.gate(inps)
            torch.cuda.synchronize()  
            self.perf_stats['moe_gating'].append(time.time() - tick)

            routing_weights = F.softmax(router_logits, dim=1)
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            expert_token_counts = {}
            for expert_id in selected_experts.unique():
                mask = (selected_experts == expert_id).any(dim=1)
                expert_token_counts[expert_id.item()] = mask.sum().item()
            
            
            sorted_experts = sorted(
                expert_token_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )            
            
            self.current_iter_expert_stats[i_layer] = {
                'expert_ids': [e[0] for e in sorted_experts],  
                'token_counts': [e[1] for e in sorted_experts]  
            }
            layer_i_stats = self.current_iter_expert_stats[i_layer]
            
                
            # ===== 【PDScope核心差异】TG/TC 贪心调度算法 =====
            #
            # Fiddler 的调度：暴力枚举 2^8=256 种 CPU/GPU 分配方案
            #   - 对每种方案计算 total_cost = Σ(expert_i 的 CPU 或 GPU 延迟)
            #   - 选 total_cost 最小的方案
            #   - 缺点：复杂度 O(2^n)，且假设 GPU 执行是串行的
            #
            # PDScope 的调度：TG/TC 贪心算法
            #   - TG = GPU 累计延迟（每多加载一个专家，增加一次 e + tg）
            #   - TC = CPU 累计延迟（从总 CPU 延迟中逐步减去被移到 GPU 的专家）
            #   - 按热度降序遍历专家，如果 TG < TC，则该专家值得 ondemand 加载到 GPU
            #   - 否则该专家及后续所有专家都放 CPU（break）
            #
            # 关键参数：
            #   e = 27.0: 单次专家权重从 CPU→GPU 加载的延迟（ms）
            #   tg = 4.22: GPU 上执行单个专家的 token 级延迟开销
            #   cpu_time_table: 预实测的 CPU 延迟表，索引为 token 数
            
            # 过滤掉已经在 GPU 上的专家（常驻 + 预取 + placeholder 中已有的）
            # 只对不在 GPU 的专家做 TG/TC 决策
            for expert_id, token_count in zip(layer_i_stats['expert_ids'], layer_i_stats['token_counts']):
                
                expert_in_gpu = False
                if self.is_expert_in_gpu_now(i_layer, expert_id):
                    expert_in_gpu = True
                elif (i_layer in self.prefetch_list and expert_id in self.prefetch_list[i_layer]) or \
                    (i_layer in self.prefetching_list and expert_id in self.prefetching_list[i_layer]):
                    expert_in_gpu = True
                else:
                    
                    for placeholder_name in ['expert_placeholder', 'expert_placeholder2', 
                                          'expert_placeholder3', 'expert_placeholder4']:
                        stored_expert = self.placeholder_to_expert[placeholder_name]
                        if stored_expert and stored_expert == (i_layer, expert_id):
                            expert_in_gpu = True
                            break
                
                if not expert_in_gpu:
                    filtered_expert_ids.append(expert_id)
                    filtered_token_counts.append(token_count)
            
            
            # 排序后的非 GPU 专家列表：(expert_id, token_count)，按 token 数降序
            sorted_experts = list(zip(filtered_expert_ids, filtered_token_counts))
            
            # TG/TC 贪心调度参数
            e = 27.0   # 单次专家权重加载延迟（CPU→GPU copy 延迟）
            tg = 4.22   # GPU 上每多加载一个专家的额外 token 处理开销
            n = len(sorted_experts)
            ondemand_experts = []  # 被判定为"值得加载到 GPU"的专家
            
            # 【PDScope核心】从预实测文件加载 CPU 延迟查找表
            # cpu_time_table[i] = CPU 上执行 i 个 token 的专家前向传播耗时
            # Fiddler 使用简单的 token_count * latency_cpu 线性模型
            cpu_time_table = [float(line.strip())  for line in open('micromixtral.txt')]
            tic=time.time()
            # TA = 所有非 GPU 专家在 CPU 上执行的总延迟
            TA = sum(cpu_time_table[min(tokens, 1498)] for expert_id, tokens in sorted_experts[0:n])
            TC = TA   # TC 初始化为总 CPU 延迟，随专家被分配到 GPU 而逐步减少
            experts_in_placeholder = []  # token 数 ≤ 1 的专家，不值得 ondemand，放 placeholder
            
            # 【PDScope核心】贪心遍历：按热度降序，逐个判断专家是否值得 ondemand 加载
            for i in range(n-1):
                expert_id, token_count = sorted_experts[i]                
                
                print("e,t,n,i",expert_id, token_count,n,i)
                # TG = 累计 GPU 延迟：加载 i+1 个专家的权重开销
                TG = (1 + i) * e + tg
                # TC = 剩余 CPU 延迟：减去当前专家后的 CPU 总延迟
                TC = TC-cpu_time_table[min(token_count, 1498)]                
                
                if self.is_decode:
                    if TG < TC:
                        if token_count>1:
                            ondemand_experts.append(expert_id)
                            
                        else:
                            experts_in_placeholder.append(expert_id)
                    else:
                        
                        break 
                else:
                    if TG < TC+cpu_time_table[min(token_count, 1498)]:
                        ondemand_experts.append(expert_id)
                        
                        if i==n-2:
                            if TC-TG>e :
                                
                                expert_id2, token_count2 = sorted_experts[i+1]
                                ondemand_experts.append(expert_id2)                 
                            elif TC-TG>e/2:
                                self.prefil_pre=True                        
                    else:
                        
                        
                        break        
            print(f"time: {(time.time() - tic)*1000:.2f}ms")
            
            
                
                

            # ===== 【PDScope核心差异】下一层专家预测 + 预取决策 =====
            # Fiddler 完全没有预取机制——每层独立决策，无跨层信息。
            # PDScope 在处理第 i 层时，用当前 hidden states 预测第 i+1 层的路由结果，
            # 提前将热门专家加载到 placeholder 中，实现计算与预取的 overlap。
            if i_layer < self.n_layer - 1:  
                next_next_layer = self.model.layers[i_layer + 1]
                
                # 用当前 hidden states 提前运行下一层的 gate，预测下一层的路由结果
                #NOTE 注意：这是近似预测——实际 i+1 层的输入会经过当前层的 MoE 后变化
                with torch.no_grad():
                    next_next_router_logits = next_next_layer.block_sparse_moe.gate(inps)
                    next_next_routing_weights = F.softmax(next_next_router_logits, dim=1)
                    _, next_next_predicted_experts = torch.topk(next_next_routing_weights, 2, dim=-1)
                
                
            # 【PDScope新增】统计每个专家被路由到的 token 数，按热度降序排列
            # 这是 TG/TC 贪心调度的基础数据，Fiddler 不需要此统计
            # （Fiddler 用 2^8 暴力枚举，只需每个专家的 token 数 * latency_cpu 即可）
                for batch_idx in range(batch_size * seq_len):
                    for expert in next_next_predicted_experts[batch_idx]:
                        expert_token_counts[expert.item()] = expert_token_counts.get(expert.item(), 0) + 1        
                
                sorted_experts = sorted(expert_token_counts.items(), key=lambda x: x[1], reverse=True)
                # 统计下一层各专家被选中的 token 数，取 top-k 作为预取候选
                top3_experts = [expert[0] for expert in sorted_experts[:self.cache]]
                
                # 根据不同 batch_size 应用不同的预取策略：
                # - 小 batch (4/8/16): 只有 token 数 ≥ 3 且不在 GPU 的专家才值得预取
                # - 大 batch: 所有 top-k 专家都预取
                # 这是因为小 batch 时预取收益较小，需要更谨慎
                self.hot_experts[i_layer + 1] = []
                for expert_id in top3_experts:
                    token_count = expert_token_counts[expert_id]
                    if self.batch_size==4:
                        if token_count >= 3 and not self.is_expert_in_gpu_now(i_layer + 1, expert_id) and i_layer + 1<32:
                            
                            
                            
                            self.hot_experts[i_layer + 1].append(expert_id)
                        elif len([e for e in top3_experts if expert_token_counts.get(e, 0) >= 2 and 
                                not self.is_expert_in_gpu_now(i_layer + 1, e)]) >= 3:
                            
                            self.hot_experts[i_layer + 1].append(expert_id)
                    elif self.batch_size==8 or self.batch_size==16:
                        if token_count >= 3 and not self.is_expert_in_gpu_now(i_layer + 1, expert_id) and i_layer + 1<32:
                            
                            
                            
                            self.hot_experts[i_layer + 1].append(expert_id)
                        elif len([e for e in top3_experts if expert_token_counts.get(e, 0) >= 3 and 
                                not self.is_expert_in_gpu_now(i_layer + 1, e)]) >= 3:
                            
                            self.hot_experts[i_layer + 1].append(expert_id)                            
                    else:
                        self.hot_experts[i_layer + 1]=top3_experts


            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            layer_expert_stats = {
                'layer_id': i_layer,
                'expert_ids': selected_experts.tolist()
            }
            self.expert_selection_stats.append(layer_expert_stats)

            
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            if self.cpu_offload == 0:
                # ===== 模式一：基线模式（cpu_offload=0）=====
                # 与 Fiddler 的基线模式基本相同，但有以下差异：
                # 1. Fiddler 将 routing_weights 传入 expert()，PDScope 在外部乘
                # 2. PDScope 缺少 GPU 常驻专家的 routing_weights 乘法（潜在 bug）
                
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        continue
                    
                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(current_state)
                    else:
                        current_state = current_state * routing_weights[top_2_list, idx_list, None]
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")

                    

            else:
                # ===== 模式二：PDScope 核心 CPU-GPU 协同调度模式 =====
                # 【与 Fiddler 的根本差异】
                #
                # Fiddler 的协同模式：
                #   1. 统计8个专家各自的 CPU/GPU 延迟（固定 latency_cpu/latency_gpu）
                #   2. 暴力枚举 2^8=256 种分配方案，选最小总延迟
                #   3. 串行执行 GPU 组 → 串行执行 CPU 组
                #
                # PDScope 的协同模式：
                #   1. 将专家分为 5 类（详见下方注释）
                #   2. GPU 线程内分 4 个子线程并行处理 4 类 GPU 侧专家
                #   3. CPU 线程独立处理剩余专家
                #   4. 预取线程异步为下一层加载专家
                #   5. 三线程并行，最大化 CPU-GPU overlap
                
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # 【PDScope核心】5 类专家分类（Fiddler 只有 cpu_experts 和 gpu_experts 两类）
                cpu_experts = []        # 第5类：在 CPU 上执行的专家（不值得 GPU 加载）
                gpu_experts = []        # （未使用，PDScope 用 experts_in_gpu 代替）
                experts_in_gpu = []     # 第1类：GPU 常驻专家（expert_loc 标记为1的）
                
                experts_loading = []    # 第3类：预取中的专家（在 prefetching_list 中，尚未完成）
                experts_remaining = []  # 第4类：ondemand 专家（TG/TC 判定值得加载，但尚未预取）                
                            
                selected_expert_ids = selected_experts.unique().tolist()

                # 遍历所有被选中的专家，按状态分为 5 类
                self._prefetch_thread_started=False  # 【PDScope新增】控制预取线程是否已启动
                for i_expert in selected_expert_ids:
                    
                    # 第1类：GPU 常驻专家 — 直接在 GPU 上执行
                    if self.is_expert_in_gpu_now(i_layer, i_expert):
                        gpu_experts.append(i_expert)
                        experts_in_gpu.append(i_expert)
                        continue
                    
                    # 检查专家是否在预取 placeholder 中（第2类）
                    expert_in_placeholder = False
                    for placeholder_name in ['expert_placeholder', 'expert_placeholder2', 
                                        'expert_placeholder3', 'expert_placeholder4']:
                        stored_expert = self.placeholder_to_expert[placeholder_name]
                        if stored_expert and stored_expert == (i_layer, i_expert):
                            expert_in_placeholder = True
                            break
                    if i_layer not in self.prefetch_list:
                        self.prefetch_list[i_layer] = []
                    if i_layer not in self.prefetching_list:
                        self.prefetching_list[i_layer] = []                            
                    # 分类决策：
                    # 第2类（placeholder 已预取）+ 第3类（预取中）+ 第4类（ondemand）→ GPU 侧
                    # 第5类（其他）→ CPU 侧
                    if expert_in_placeholder or i_expert in ondemand_experts or i_expert in self.prefetch_list[i_layer] or i_expert in self.prefetching_list[i_layer]:
                       
                        if expert_in_placeholder or i_expert in self.prefetch_list[i_layer]:
                            experts_in_placeholder.append(i_expert)  # 第2类：已预取完成，在 placeholder 中
                            
                        elif i_expert in self.prefetching_list[i_layer]:
                            experts_loading.append(i_expert)  # 第3类：正在预取中
                        else:
                            experts_remaining.append(i_expert)  # 第4类：ondemand 待加载
                    else:
                        cpu_experts.append(i_expert)  # 第5类：在 CPU 上执行


                if self.is_decode:
                    laymid = time.time() - layer_tick
                    laypid = time.time()
                
                
                # ===== 【PDScope核心】三线程并行执行（Fiddler 是串行的）=====
                # GPU 线程：内部再分 4 个子线程，分别处理 4 类 GPU 侧专家
                # CPU 线程：处理第5类（纯 CPU 执行的专家）
                # 预取线程：为下一层预取热门专家（可选，大 batch 时启用）
                
                gpu_results = []  
                cpu_results = []
                gpu_time = 0.0
                cpu_time = 0.0

                def process_gpu_experts():
                    """GPU 线程：并行处理 4 类 GPU 侧专家。
                    
                    【对比差异】Fiddler 串行执行 GPU 组的专家，每个专家用 load_state_dict
                    加载到唯一 placeholder。PDScope 用 4 个子线程并行处理：
                    - process_experts_in_gpu: GPU 常驻专家，直接执行
                    - process_experts_in_placeholder: 已预取到 placeholder 的专家
                    - process_experts_loading: 正在预取中的专家（等待或使用 placeholder）
                    - process_experts_remaining: ondemand 专家，用 placeholder5/6 实时加载
                    """
                    nonlocal gpu_time
                    start_time=time.time()
                    def process_experts_in_gpu():
                        """子线程1：处理 GPU 常驻专家——直接在 GPU 上执行。"""
                        results = []
                        for i_expert in experts_in_gpu:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                                
                            expert_input = inps[mask]
                            tick = time.time()
                            expert_output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                            self.perf_stats['expert_compute'].append(time.time() - tick)
                            print("111")
                            weights = routing_weights[mask].gather(
                                1, (selected_experts[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results                   
                    def process_experts_in_placeholder():
                        """子线程2：处理已预取到 placeholder（1-4）的专家。
                        NOTE【注意】当前实现硬编码 run_expert_at_gpu(11, 2, ...)，
                        疑似调试代码——应该根据实际 placeholder 映射来执行。
                        """
                        results = []

                        for i_expert in experts_in_placeholder:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            print("222")
                            expert_input = inps[mask]
                            tick = time.time()
                            expert_output = self.run_expert_at_gpu(11, 2, expert_input)
                            self.perf_stats['expert_compute'].append(time.time() - tick)
 
                            weights = routing_weights[mask].gather(
                                1, (selected_experts[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results                
                    def process_experts_loading():
                        """子线程3：处理正在预取中的专家（prefetching_list 中的）。
                        NOTE:【注意】同样硬编码 run_expert_at_gpu(11, 2, ...)，疑似调试代码。
                        """
                        results = []

                        for i_expert in experts_loading:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            print("333")                                

                            expert_input = inps[mask]
                            tick = time.time()
                            expert_output = self.run_expert_at_gpu(11, 2, expert_input)
                            self.perf_stats['expert_compute'].append(time.time() - tick)
                                
                            weights = routing_weights[mask].gather(
                                1, (selected_experts[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results
                            
                    def process_experts_remaining():
                        """子线程4：处理 ondemand 专家——使用 placeholder5/6 实时加载并执行。
                        这是 TG/TC 贪心调度中被判定为"值得加载到 GPU"的专家。
                        """
                        results = []
                        
                        for i_expert in experts_remaining:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            print("444")    
                            expert_input = inps[mask]
                            tick = time.time()
                            # 获取可用的 ondemand placeholder（5 或 6）
                            if not self.expert_placeholder5_inused:
                                target_placeholder = self.expert_placeholder5
                                self.expert_placeholder5_inused = True
                            elif not self.expert_placeholder6_inused:
                                target_placeholder = self.expert_placeholder6
                                self.expert_placeholder6_inused = True
                            else:
                                while self.expert_placeholder5_inused and self.expert_placeholder6_inused:
                                    time.sleep(0.0001)
                                continue
                                
                            # 同步加载专家权重到 ondemand placeholder
                            # 【对比差异】Fiddler 用 load_state_dict，PDScope 用 _async_ondemand
                            # （名字含 async 但当前是同步调用，实际做 pin_memory + copy_）
                            self._async_ondemand(i_layer, i_expert, target_placeholder)
                            expert_output = target_placeholder(expert_input)
                            print("ondemand",i_expert)
                            weights = routing_weights[mask].gather(
                                1, (selected_experts[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                            
                            
                            if target_placeholder == self.expert_placeholder5:
                                self.expert_placeholder5_inused = False
                            else:
                                self.expert_placeholder6_inused = False
                        return results
                    # 【PDScope核心】GPU 线程内部启动 4 个子线程并行执行
                    # 每个子线程处理一类 GPU 侧专家，结果收集到 results 列表中
                    # Fiddler: 串行遍历 gpu_experts，逐个执行
                    # PDScope: 4 个子线程并行，最大限度利用 GPU
                    
                    def run_and_collect(func):
                        res = func()
                        results.extend(res)
                        
                    threads.append(threading.Thread(target=run_and_collect, args=(process_experts_in_gpu,)))
                    threads.append(threading.Thread(target=run_and_collect, args=(process_experts_in_placeholder,)))
                    threads.append(threading.Thread(target=run_and_collect, args=(process_experts_loading,)))
                    threads.append(threading.Thread(target=run_and_collect, args=(process_experts_remaining,)))
                    
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    
                    
                    for mask_index, expert_output in results:
                        inps_after_experts.index_add_(
                            0,
                            mask_index,
                            expert_output.to(inps_after_experts.dtype)
                        )
                    
                    gpu_time = time.time() - start_time


                def process_cpu_experts():
                    """CPU 线程：在 CPU 上执行第5类专家。
                    【对比差异】Fiddler: 在主线程串行执行 CPU 组
                    PDScope: CPU 线程与 GPU 线程并行，通过 index_add_ 在最后合并结果
                    这是 PDScope 三线程并行的核心——CPU 和 GPU 计算同时进行。
                    """
                    nonlocal cpu_time
                    start_time = time.time()
                    for i_expert in cpu_experts:
                        mask = (selected_experts == i_expert).any(dim=1)
                        if not mask.any():
                            continue
                            
                        expert_input = inps[mask].to("cpu")
                        tick = time.time()
                        # 【对比差异】Fiddler 的 run_expert_at_cpu 接收 routing_weights 参数
                        # PDScope 的 run_expert_at_cpu 不接收 routing_weights
                        # 权重乘法在这里手动完成
                        expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input )
                        self.perf_stats['expert_compute-cpu'].append(time.time() - tick)
                        
                        weights = routing_weights[mask].gather(
                            1, (selected_experts[mask] == i_expert).long().argmax(dim=1, keepdim=True)
                        ).to("cpu")
                        expert_output = expert_output * weights
                        mask_index = mask.nonzero().squeeze(1)
                        cpu_results.append((mask_index, expert_output))
                    cpu_time = time.time() - start_time

                def prefetch_experts():
                    """【PDScope新增】预取线程：异步为下一层加载热门专家。
                    
                    工作流程：
                    1. 确定预取目标层（当前层 + 1）
                    2. 从 self.hot_experts 获取目标层的热门专家列表
                    3. 对每个热门专家，检查是否已在 GPU 或 placeholder 中
                    4. 若不在，则模拟加载（time.sleep(0.027) 模拟权重搬运延迟）
                       然后标记为已预取（加入 prefetch_list）
                    
                    NOTE【注意】当前实现使用 time.sleep 模拟加载，实际的 _async_load_expert
                    调用被替换为 sleep + prefetch_list 更新。这可能是一个简化版本。
                    
                    Fiddler 完全没有预取机制。
                    """
                    
                    
                    hot_experts = self.hot_experts
                    
                    
                    # 根据 batch_size 决定预取多少个专家
                    # batch 越大，预取越多（大 batch 时 GPU 侧负载重，需要更多预取来隐藏延迟）
                    if self.batch_size == 4 or self.batch_size  == 8 or self.batch_size  == 16:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1
                    elif self.batch_size  == 32:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1
                    else:  # 大 batch (64+)
                        self.prefetch_layers = i_layer + 1
                        expert_count = 2
                        
                    
                    if  self.prefetch_layers >= self.n_layer:
                        self.prefetch_layers =  self.prefetch_layers % self.n_layer

                    
                    layer_hot_experts = hot_experts.get( self.prefetch_layers , [])
                    layer_hot_experts_later = hot_experts.get(( self.prefetch_layers +1)%self.n_layer, [])
                    
                    if not layer_hot_experts:
                        
                        
                        return
                    self.prefetch_list[self.prefetch_layers]=[]
                    if self.prefetch_layers not in self.prefetching_list:
                        self.prefetching_list[self.prefetch_layers]=[]                        
                    
                    experts_loaded = 0
                    for i in range(min(expert_count, len(layer_hot_experts))):
                        expert_id = layer_hot_experts[i]
                        expert_not_in_placeholder = True
                        for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                                            'expert_placeholder3', 'expert_placeholder4']:
                            stored_expert = self.placeholder_to_expert[placeholder_name]
                            if stored_expert and stored_expert == (self.prefetch_layers, expert_id):
                                expert_not_in_placeholder = False
                                break
                        if not self.is_expert_in_gpu_now(self.prefetch_layers, expert_id) and expert_not_in_placeholder:
                            tick=time.time()
                            # 标记为正在预取
                            self.prefetching_list[self.prefetch_layers].append(expert_id)
                            # 模拟权重搬运延迟（实际应调用 _async_load_expert）
                            time.sleep(0.027)  # ≈27ms，与 e=27.0 对应
                            # 标记为预取完成
                            self.prefetch_list[self.prefetch_layers].append(expert_id)
                            self.prefetching_list[self.prefetch_layers]=[]
                            
                            experts_loaded += 1
                            if experts_loaded >= expert_count:
                                break


                
                
                # ===== 【PDScope核心】启动三线程并行 =====
                parallel_start = time.time()
                prefetch_thread = threading.Thread(target=prefetch_experts)
                gpu_thread = threading.Thread(target=process_gpu_experts)
                cpu_thread = threading.Thread(target=process_cpu_experts)
                
                # 预取线程仅在大 batch (batch_size > 8) 且首次预取时启动
                # 小 batch 时预取收益不明显，不值得额外开销
                if self.is_decode and self.batch_size>8:
                    if self._prefetch_thread_started==False:
                        prefetch_thread.start()
                        self._prefetch_thread_started = True
                # 启动 GPU 线程和 CPU 线程（两者并行执行）
                gpu_thread.start()
                cpu_thread.start()
                
                # 等待 GPU 和 CPU 线程完成
                # 【对比差异】Fiddler: 先串行执行 GPU 组，再串行执行 CPU 组
                # PDScope: GPU 和 CPU 线程并行，通过 join 等待两者都完成
                #NOTE 注意：未 join prefetch_thread，预取在后台异步进行
                gpu_thread.join()
                cpu_thread.join()
                parallel_time = time.time() - parallel_start

                # 【PDScope新增】计算并行度和线程利用率统计
                max_thread_time = max(gpu_time, cpu_time)
                parallel_degree = (gpu_time + cpu_time) / parallel_time if parallel_time > 0 else 1.0

                
                if self.is_decode:
                    stats_text = f"\nLayer {i_layer} Thread Time Stats:\n"
                    stats_text += f"GPU Thread Time: {gpu_time*1000:.2f}ms\n"
                    stats_text += f"CPU Thread Time: {cpu_time*1000:.2f}ms\n"
                    stats_text += f"Parallel Time: {parallel_time*1000:.2f}ms\n"
                    stats_text += f"Parallel Degree: {parallel_degree:.2f}x\n"
                    print(stats_text)
                    
                    
                    os.makedirs('./log', exist_ok=True)
                    with open('./log/linshi.txt', 'a') as f:
                        f.write(stats_text)
                # 合并 CPU 线程的结果到 inps_after_experts
                # CPU 线程的结果在 CPU 上，需要 .to(self.dev) 传回 GPU
                # 【对比差异】Fiddler: CPU 组的结果也是通过 .to(self.dev) 传回，逻辑相同
                # 但 Fiddler 是串行的，PDScope 在 CPU 线程完成后统一合并
                for mask_index, expert_output in cpu_results:
                    inps_after_experts.index_add_(
                        0,
                        mask_index.to(self.dev),
                        expert_output.to(self.dev).to(inps_after_experts.dtype)
                    )
                
                   
            
            # MoE 残差连接：将专家输出加回残差
            # 【对比差异】Fiddler: inps = inps_residual + inps_after_experts.reshape(original_inps_shape)
            # PDScope: 使用 batch_size, seq_len, hidden_dim 而非 original_inps_shape
            inps = inps_residual + inps_after_experts.reshape(batch_size, seq_len, hidden_dim)  
            # 【PDScope新增】decode 阶段记录每层总耗时，累加到 layer_time_accumulator
            # 注意：此处有两段重复的计时逻辑（第一个 if 只累加 accumulator，第二个 if 做完整统计）
            if self.is_decode:
                layer_time = time.time() - layer_tick
                layer_time_final = time.time() - laypid
                self.layer_time_accumulator[i_layer] += layer_time

            if self.is_decode:
                layer_time = time.time() - layer_tick
                # layer_times: 每层总耗时；layer_times_mid: 中间阶段耗时；layer_times_final: 最终耗时
                layer_times[i_layer] += layer_time  
                layer_times_mid[i_layer] += laymid
                layer_times_final[i_layer] += layer_time_final
                self.layer_time_accumulator[i_layer] += layer_time

                # 【PDScope新增】将每次的层耗时明细追加到 layer_time_stats 列表
                # 用于后续按层/场景统计平均耗时（见 generate() 末尾 L799-808）
                self.layer_time_stats.append({
                    'layer_id': i_layer,
                    'time': layer_time,
                    'token_step': self.past_key_values_length
                })

        # ===== 所有层循环结束 =====
        # 【PDScope新增】调用 get_hot_expert() 收集本轮各层专家热度排名
        # 将 current_iter_expert_stats 转移到 last_iter_expert_stats，供下一轮 prefetch 使用
        hot_experts = self.get_hot_expert()
        
        # 【PDScope新增】decode 阶段输出逐层耗时日志
        # 计算每层耗时(ms)、CPU 专家耗时占比等统计指标，写入日志文件
        if self.is_decode:
            
            for i in range(self.n_layer):
                
                layer_time = layer_times[i] * 1000
                layer_times_mids = layer_times_mid[i] * 1000
                layer_times_finals = layer_times_final[i] * 1000

            avg_layer_time = sum(layer_times.values()) / self.n_layer
            total_cpu_time = sum(self.cpu_expert_time_per_layer.values())
            # avg_cpu_ratio: CPU 专家耗时占平均层耗时的百分比，反映 CPU offload 开销
            avg_cpu_ratio = (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0
            
            os.makedirs('./log', exist_ok=True)
            with open('./log/expert_stats.txt', 'a') as f:
                # 【PDScope新增】逐层记录 CPU 专家耗时、层总耗时、中间/最终阶段耗时
                for i in range(self.n_layer):
                    cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                    layer_time = layer_times[i] * 1000
                    layer_time_mids = layer_times_mid[i] * 1000
                    layer_time_finals = layer_times_final[i] * 1000

        # 所有层处理完毕后，经过最终的 LayerNorm 和 lm_head 得到 logits
        # 【与 Fiddler 相同】
        other_ops_start = time.time()

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        # 【PDScope新增】decode 阶段统计总耗时
        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start

        self.present_key_value = present_key_value
        return lm_logis
        
        
        
    def run_expert_at_gpu(self, i_layer, i_expert, inps ):
        """【PDScope新增】在 GPU 上执行指定专家。
        
        【对比差异】Fiddler 没有 run_expert_at_gpu 方法——GPU 专家通过
        experts[i_expert](current_state, routing_weights) 直接调用。
        PDScope 将其封装为独立方法，增加了：
        1. torch.cuda.synchronize() 确保计时准确
        2. 专家热度分类（normal/hot/veryhot）
        3. 性能统计记录
        """
        start_time = time.time()
        result = self.model.layers[i_layer].block_sparse_moe.experts[i_expert](inps)
        torch.cuda.synchronize()  
        elapsed = time.time() - start_time
        
        # 【PDScope新增】GPU 专家热度分类逻辑：
        # normal: token_count <= 1（冷门专家）
        # hot: 1 < token_count <= 2（中等热度）
        # veryhot: token_count > 2（热门专家，可能需要常驻 GPU）
        token_count = inps.shape[0]
        expert_status = 'normal'
        if token_count > 2:
            expert_status = 'veryhot'
        elif token_count > 1:
            expert_status = 'hot'
        
        # 【PDScope新增】记录到当前迭代统计，供 get_hot_expert() 排序使用
        self.current_iter_expert_stats[i_layer]['expert_ids'].append(i_expert)
        self.current_iter_expert_stats[i_layer]['token_counts'].append(token_count)      

        # 【PDScope新增】追加到 expert_time_stats，供 get_expert_stats() 汇总分析
        self.expert_time_stats.append({
            'layer_id': i_layer,
            'expert_id': i_expert,
            'time': elapsed,
            'device': 'gpu',
            'token_count': token_count,
            'status': expert_status
        })
        return result

    def run_expert_at_cpu(self, i_layer, i_expert, inps ):
        """在 CPU 上执行指定专家。
        
        【对比差异】Fiddler 的 run_expert_at_cpu 签名为:
            run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights)
        它在内部将 routing_weights 传入专家 forward:
            experts[i_expert](inps, routing_weights)
        PDScope 的签名不接受 routing_weights，权重乘法在调用处手动完成。
        这是因为 PDScope 的专家可能在 GPU placeholder 或 CPU 上执行，
        权重处理方式不同，统一在外部乘更灵活。
        """
        start_time = time.time()
        result = self.model.layers[i_layer].block_sparse_moe.experts[i_expert](inps)
        
        elapsed = time.time() - start_time
        # 【PDScope新增】累加每层 CPU 专家耗时，用于计算 avg_cpu_ratio
        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed 
        # 【PDScope新增】与 run_expert_at_gpu 相同的热度分类逻辑
        token_count = inps.shape[0]
        expert_status = 'normal'
        if token_count > 2:
            expert_status = 'veryhot'
        elif token_count > 1:
            expert_status = 'hot'
        
        # 【PDScope新增】记录到当前迭代统计
        self.current_iter_expert_stats[i_layer]['expert_ids'].append(i_expert)
        self.current_iter_expert_stats[i_layer]['token_counts'].append(token_count)

        # 【PDScope新增】追加到 expert_time_stats
        self.expert_time_stats.append({
            'layer_id': i_layer,
            'expert_id': i_expert,
            'time': elapsed,
            'device': 'cpu',
            'token_count': token_count,
            'status': expert_status
        })
        return result
    
    def get_expert_stats(self):
        """【PDScope新增】汇总专家运行统计信息。
        返回各层的热门专家数量分布、token 分布等。Fiddler 无此方法。
        
        返回结构:
        {
            'hot_experts': {layer_id: {'count': 总调用数, 'hot': 热门数, 'veryhot': 超热门数}},
            'hot_counts': {2: 有2个hot的层数, 3: ..., 4: ..., 5: ...},
            'token_distribution': {token_count: 出现次数}
        }
        """
        
        stats = {
            'hot_experts': {i: {'count': 0, 'hot': 0, 'veryhot': 0} for i in range(self.n_layer)},
            'hot_counts': {2: 0, 3: 0, 4: 0, 5: 0},
            'token_distribution': {}  
        }
        
        # 第一遍遍历：统计每层的专家调用次数和热度分布
        for record in self.expert_time_stats:
            layer = record['layer_id']
            token_count = record['token_count']
            expert_status = record['status']
        
            stats['hot_experts'][layer]['count'] += 1
            if expert_status == 'hot':
                stats['hot_experts'][layer]['hot'] += 1
            elif expert_status == 'veryhot':
                stats['hot_experts'][layer]['veryhot'] += 1
        
        # 第二遍遍历：统计每层的 hot+veryhot 总数，用于 hot_counts 分布
        layer_hot_counts = {i: 0 for i in range(self.n_layer)}
        for record in self.expert_time_stats:
            if record['status'] in ['hot', 'veryhot']:
                layer_hot_counts[record['layer_id']] += 1
                
        # 统计有 2~5 个热门专家的层数分布
        for count in layer_hot_counts.values():
            if count >= 2 and count <= 5:
                stats['hot_counts'][count] += 1
                
        # 统计 token 数的频率分布（各专家处理的 token 数出现的次数）
        stats['token_distribution'] = {}
        token_counts = [r['token_count'] for r in self.expert_time_stats]
        unique_counts = set(token_counts)
        
        for count in unique_counts:
            stats['token_distribution'][count] = token_counts.count(count)
                
        return stats
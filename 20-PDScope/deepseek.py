import copy
import threading
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
from transformers.masking_utils import create_causal_mask
from log_utils import LOG, init_log

class FiddlerDeepSeekV2:
    """
    DeepSeek-V2 MoE GPU 卸载推理引擎。
    
    【与 Mixtral 版本的核心差异概览】
    1. 模型架构: DeepSeek 有 26 层(1-26)、40 个路由专家 + 2 个共享专家；Mixtral 有 32 层、8 个专家、无共享专家
    2. 专家路径: layer.mlp.experts / layer.mlp.gate / layer.mlp.shared_experts
                → Mixtral: layer.block_sparse_moe.experts / .gate
    3. 权重名: gate_proj/up_proj/down_proj → Mixtral: w1/w2/w3
    4. Gate 返回值: 直接返回 (selected_experts, routing_weights, _) 三元组
                   → Mixtral: 返回 router_logits，需手动 softmax + topk
    5. TG/TC 参数: e=1.39, tg=0.95 → Mixtral: e=27.0, tg=4.22
    6. CPU 延迟表: microdeepseek.txt → Mixtral: micromixtral.txt
    7. Placeholder 分配: 按顺序 5→6→3→4→1→2 → Mixtral: 按奇偶层分离(1/2 偶数层, 3/4 奇数层)
    8. 缺少 expert_loading_status 和 placeholder_lock（Mixtral 有）
    9. 预取触发: batch_size > 16 → Mixtral: batch_size > 8
    10. expert_input 提取: 需要 batch_mask.view(batch_size, seq_len) → Mixtral: 直接 inps[mask]
    """

    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        # 【与 Mixtral 相同】每层预测的热门专家列表，用于异步预取
        self.hot_experts = {}

        # 【差异】DeepSeek 用 AutoModelForCausalLM + trust_remote_code（DeepSeek 模型需要）
        # Mixtral 用 MixtralForCausalLM（标准 transformers 模型类）
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            use_cache=True,
        )
        self.batch_size = args.batch_size
        # 【差异】cache 策略：DeepSeek batch=8 时 cache=10, else=15
        # Mixtral batch=4 时 cache=5, else=7
        if self.batch_size==4:
           self.cache=5
        elif self.batch_size==8:
           self.cache=10
        else:
           self.cache=15
        self.lm_head = self.model.lm_head
        self.prefil_pre = False
        # 【差异】DeepSeek 模型结构: model.model（两层嵌套）
        # Mixtral: model.model（相同嵌套方式）
        self.model = self.model.model
        # 【差异】DeepSeek 用 layers[1].mlp（第 1 层是 MoE 层，第 0 层是全连接层）
        # Mixtral 用 layers[0].block_sparse_moe（所有层都有 MoE）
        first_layer_mlp = self.model.layers[1].mlp

        # ===== 6 个 GPU 占位专家 =====
        # 【差异】DeepSeek 从 layers[1].mlp.experts[0-5] 拷贝
        # Mixtral 从 layers[0].block_sparse_moe.experts[0,1,2,3,5,6] 拷贝
        # 功能相同：4 个主 placeholder（预取用）+ 2 个 ondemand placeholder（实时加载用）
        self.expert_placeholder = copy.deepcopy(
            first_layer_mlp.experts[0]
        ).to(self.dev)
        self.expert_placeholder2 = copy.deepcopy(
            first_layer_mlp.experts[1]
        ).to(self.dev)
        self.expert_placeholder3 = copy.deepcopy(
            first_layer_mlp.experts[2]
        ).to(self.dev)
        self.expert_placeholder4 = copy.deepcopy(
            first_layer_mlp.experts[3]
        ).to(self.dev)
        # 2 个 ondemand 专用 placeholder
        self.expert_placeholder5 = copy.deepcopy(
            first_layer_mlp.experts[4]
        ).to(self.dev)
        self.expert_placeholder6 = copy.deepcopy(
            first_layer_mlp.experts[5]
        ).to(self.dev)
        # 【差异】反向映射 (layer, expert) → placeholder 对象
        self.expert_to_placeholder = {}
        self.expert_placeholder_inused=False
        self.expert_placeholder2_inused=False
        self.expert_placeholder3_inused=False
        self.expert_placeholder4_inused=False
        self.expert_placeholder5_inused=False
        self.expert_placeholder6_inused=False
        self.expert_loading_status = {
            'expert_placeholder': True,
            'expert_placeholder2': True,
            'expert_placeholder3': True,
            'expert_placeholder4': True
        }
        self.placeholder_lock = threading.Lock()
        self.prefetch_layers=0
        self.is_decode = False
        self.prefetch_list = {}
        self.prefetching_list = {}
        # 【差异】DeepSeek 缺少 expert_loading_status（Mixtral 有完整加载状态管理）
        # DeepSeek 缺少 placeholder_lock（Mixtral 有 threading.Lock()）
        self.placeholder_to_expert = {
            'expert_placeholder': None,
            'expert_placeholder2': None,
            'expert_placeholder3': None,
            'expert_placeholder4': None
        }
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        # 添加 RotaryEmbedding 和 config 用于创建 causal mask
        self.rotary_emb = DeepseekV2RotaryEmbedding(config=self.model.config, device=self.dev)
        self.config = self.model.config
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)

        # 【差异】DeepSeek 有 40 个路由专家 + 2 个共享专家；Mixtral 有 8 个专家、无共享专家
        self.n_expert = self.model.config.n_routed_experts
        self.n_shared_experts = 2
        self.expert_selection_stats = []
        self.expert_time_stats = []

        # 【差异】DeepSeek 层范围 range(1, self.n_layer)；Mixtral range(self.n_layer) 即 0-31
        self.expert_selection_history = {}
        self.hit_stats = {}
        for i in range(1, self.n_layer):
            if i>0:
                self.expert_selection_history[i] = []
                self.hit_stats[i] = {'hits': 0, 'total': 0}

        # 【差异】DeepSeek 权重累积器维度 6（top-6 专家）；Mixtral 维度 8（全部 8 个专家）
        self.expert_weight_accumulator = {}
        for i in range(1, self.n_layer):
            if i>0:
                self.expert_weight_accumulator[i] = torch.zeros(6, device=self.dev)
        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(1, self.n_layer)}

        # 【差异】这些固定值在两个版本中均未被实际使用，实际调度用 e/tg 参数
        self.latency_cpu = 5
        self.latency_gpu = 45

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        # expert_loc: 静态位置表，1=GPU常驻, 0=CPU常驻
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        LOG(
            f"Number of experts on GPU: {n_expert_on_gpu}/{(self.n_layer-1) * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)

        # ===== 分层耗时统计 =====
        self.layer_time_stats = []
        self.layer_time_accumulator = {}
        for i in range(1, self.n_layer):
            self.layer_time_accumulator[i] = 0.0
        self.layer_time_details = {
            'all_gpu': [],
            'all_cpu': [],
            'mixed': []
        }

        self.layer_time_accumulator_details = {
            'all_gpu': {i: 0.0 for i in range(1, self.n_layer)},
            'all_cpu': {i: 0.0 for i in range(1, self.n_layer)},
            'mixed': {i: 0.0 for i in range(1, self.n_layer)}
        }
        self.last_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(1, self.n_layer)
        }
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(1, self.n_layer)
        }

        self.layer_data = {}

        tick = time.time()
        self.bring_expert_to_gpu()

        init_log("./log/linshi.txt")
        LOG("Model is ready.")



    def bring_non_expert_to_gpu(self):
        """将非专家层加载到 GPU。
        
        【差异】DeepSeek 特有:
        - layers[0] 整层加载到 GPU（第 0 层是全连接 MLP，非 MoE）
        - 第 1-26 层只加载 self_attn/input_layernorm/gate/post_attention_layernorm
        - 额外加载 shared_experts（DeepSeek 有共享专家，Mixtral 没有）
        
        Mixtral: 所有 32 层结构相同，每层加载 self_attn/input_layernorm/gate/post_attention_layernorm
                 无 shared_experts，无第 0 层特殊处理
        """
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        # 【差异】DeepSeek 第 0 层整层加载（非 MoE 层）
        self.model.layers[0].to(self.dev)
        for i in range(len(self.model.layers)):
            if i!=0:
                self.model.layers[i].self_attn.to(self.dev)
                self.model.layers[i].input_layernorm.to(self.dev)
                # 【差异】专家路径: layer.mlp.gate → Mixtral: layer.block_sparse_moe.gate
                self.model.layers[i].mlp.gate.to(self.dev)
                self.model.layers[i].post_attention_layernorm.to(self.dev)
        # 【差异】DeepSeek 独有：加载共享专家到 GPU（Mixtral 无共享专家）
        for i in range(1, self.n_layer):
            if i>0:
                self.model.layers[i].mlp.shared_experts.to(self.dev)
    def get_hot_expert(self):
        """收集当前迭代的专家热度排名，按 token 数降序。
        
        【差异】DeepSeek: 遍历 range(self.n_layer)，但跳过 layer_id==0（第 0 层无 MoE）
                Mixtral: 遍历 range(self.n_layer)，所有层都处理
        逻辑与 Mixtral 版本基本相同。
        """
        if not hasattr(self, 'is_decode') or not self.is_decode:
            return {}
        
        hot_experts = {}
        
        for layer_id in range(self.n_layer):
            # 【差异】DeepSeek 跳过第 0 层（全连接层，非 MoE）
            if layer_id>0:
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
        
        self.hot_experts = hot_experts
        return hot_experts
    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """根据热度排序将前 n_expert_on_gpu 个专家标记为 GPU 常驻。
        
        【差异】
        - DeepSeek 热文件: './hot/deep.txt' → Mixtral: './hot/mix.txt'
        - DeepSeek fallback: 枚举所有 26×40=1040 个专家
          Mixtral fallback: 硬编码 82 个热门专家列表
        - DeepSeek 有 min(n_expert_on_gpu, len(popular_experts)) 保护
          Mixtral 直接遍历 range(n_expert_on_gpu)，无此保护
        """
        if popular_experts is None:
            hot_experts_file = './hot/deep.txt'
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, 'r') as f:
                        popular_experts = [tuple(map(int, line.strip().split(',')))
                                            for line in f if line.strip()]
                    LOG(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    LOG(f"Error loading hot experts: {e}")
            else:
                # 【差异】fallback: 枚举所有层和专家（26层×40专家=1040个）
                popular_experts = []
                for layer in range(1, self.n_layer):
                    for expert in range(40):
                        popular_experts.append((layer, expert))
        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def _async_ondemand(self, layer_idx, expert_id, target_placeholder):
        """ondemand 加载：将 CPU 专家权重复制到指定的 GPU placeholder。
        
        【差异】
        - DeepSeek 权重名: ['gate_proj', 'up_proj', 'down_proj']
          Mixtral 权重名: ['w1', 'w2', 'w3']
        - DeepSeek 专家路径: layer.mlp.experts[expert_id]
          Mixtral: layer.block_sparse_moe.experts[expert_id]
        - 两者都使用 pin_memory + copy_ 进行 CPU→GPU 搬运
        """
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]
    
        if next(expert.parameters()).is_cuda:
            return
        
        # pin_memory 固定 CPU 权重，加速后续 copy_
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned

        tick = time.time()
        # 将权重从 CPU 复制到 GPU placeholder
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
            dst.copy_(src)
            
        copytime = time.time() - tick
        


    def _async_load_expert(self, layer_idx, expert_id):
        """异步加载专家到预取 placeholder（1-4）。
        
        【差异】
        - DeepSeek: 按顺序分配 placeholder (1→2→3→4)，无奇偶层分离
          Mixtral: 按奇偶层分离分配（奇数层→3/4，偶数层→1/2）
        - DeepSeek: 缺少 expert_loading_status 更新（Mixtral 加载前设 True，完成后设 False）
        - DeepSeek: 缺少加载完成后的状态清理（Mixtral 在 copy_ 完成后更新 loading_status=False）
        """
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]
    
        if next(expert.parameters()).is_cuda:
            return
        
        # pin_memory 固定权重
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned
        # 【差异】DeepSeek 按顺序分配；Mixtral 按奇偶层分离
        target_placeholder = None
        if not self.expert_placeholder_inused:
            target_placeholder = self.expert_placeholder
            self.expert_placeholder_inused = True
            self.expert_loading_status['expert_placeholder'] = True
        elif not self.expert_placeholder2_inused:
            target_placeholder = self.expert_placeholder2
            self.expert_placeholder2_inused = True
            self.expert_loading_status['expert_placeholder2'] = True
        elif not self.expert_placeholder3_inused:
            target_placeholder = self.expert_placeholder3
            self.expert_placeholder3_inused = True
            self.expert_loading_status['expert_placeholder3'] = True
        elif not self.expert_placeholder4_inused:
            target_placeholder = self.expert_placeholder4
            self.expert_placeholder4_inused = True
            self.expert_loading_status['expert_placeholder4'] = True
        else:
            raise RuntimeError("No available expert placeholder")

        # 更新正向映射: placeholder 名 → (layer, expert)
        if target_placeholder == self.expert_placeholder:
            self.placeholder_to_expert['expert_placeholder'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder2:
            self.placeholder_to_expert['expert_placeholder2'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder3:
            self.placeholder_to_expert['expert_placeholder3'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder4:
            self.placeholder_to_expert['expert_placeholder4'] = (layer_idx, expert_id)

        # copy_ 权重搬运
        tick = time.time()
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
            dst.copy_(src)
        copytime=time.time() - tick

        if target_placeholder == self.expert_placeholder:
            self.expert_loading_status['expert_placeholder'] = False
        elif target_placeholder == self.expert_placeholder2:
            self.expert_loading_status['expert_placeholder2'] = False
        elif target_placeholder == self.expert_placeholder3:
            self.expert_loading_status['expert_placeholder3'] = False
        elif target_placeholder == self.expert_placeholder4:
            self.expert_loading_status['expert_placeholder4'] = False
        
        self.expert_to_placeholder[(layer_idx, expert_id)] = target_placeholder
    def release_placeholder(self, layer_idx, expert_id):
        """释放已用完的 placeholder。
        
        【差异】
        - DeepSeek: 释放后缺少对 expert_loading_status 的重置（因为 DeepSeek 初始化时
          就没有 expert_loading_status）
        - Mixtral: 同样只清理 inused 和 placeholder_to_expert，不重置 loading_status
        - 释放条件两者相同：stored_expert[0] < layer_idx 或循环回第 0 层时释放最后一层的
        """
        placeholder_inused_map = {
            'expert_placeholder': 'expert_placeholder_inused',
            'expert_placeholder2': 'expert_placeholder2_inused',
            'expert_placeholder3': 'expert_placeholder3_inused',
            'expert_placeholder4': 'expert_placeholder4_inused',
        }
        for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                              'expert_placeholder3', 'expert_placeholder4']:
            stored_expert = self.placeholder_to_expert[placeholder_name]
            if stored_expert and (stored_expert[0] < layer_idx or 
                    (stored_expert[0] == self.n_layer - 1 and layer_idx <= 1)):
                setattr(self, placeholder_inused_map[placeholder_name], False)
                self.placeholder_to_expert[placeholder_name] = None
        
    def is_expert_loading(self, placeholder_name):
        """检查指定 placeholder 是否正在加载中。
        
        【差异】DeepSeek 缺少 expert_loading_status 初始化，此方法会抛出 AttributeError。
        Mixtral 有完整的 expert_loading_status 字典（初始化时全部设为 True）。
        """
        return self.expert_loading_status.get(placeholder_name, False)
    
    def is_expert_loaded(self, layer_id, expert_id):
        """检查专家是否已完成预取加载（不在 prefetching_list 中）。
        逻辑与 Mixtral 相同。"""
        return (layer_id not in self.prefetching_list or 
                expert_id not in self.prefetching_list[layer_id])   
    def bring_expert_to_gpu(self):
        """将 expert_loc 中标记为 GPU 的专家实际加载到 GPU 显存。
        
        【差异】
        - DeepSeek 专家路径: layer.mlp.experts[j]
          Mixtral: layer.block_sparse_moe.experts[j]
        - DeepSeek 无 try/except OOM 保护
          Mixtral 有 try/except RuntimeError('out of memory')
        """
        expert_count = 0
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].mlp.experts[j].to(self.dev)
                    expert_count += 1
    def is_expert_in_gpu(self, i_layer, i_expert):
        """查 expert_loc 静态表判断专家是否标记为 GPU 常驻。与 Mixtral 相同。"""
        return self.expert_loc[i_layer, i_expert] == 1
    
    def is_expert_in_gpu_now(self, i_layer, i_expert):
        """检查专家是否实际在 GPU 上（通过检查参数 device）。
        
        【差异】DeepSeek 专家路径: layer.mlp.experts
          Mixtral: layer.block_sparse_moe.experts
        功能与 Mixtral 版本相同——检查参数实际的 device 反映动态变化。
        """
        expert = self.model.layers[i_layer].mlp.experts[i_expert]
        return next(expert.parameters()).is_cuda


    def calc_n_expert_on_gpu(self):
        """根据 GPU 剩余显存计算能容纳多少个专家。
        
        【差异】
        - DeepSeek: batch=64→893, batch=32→993, batch=16→693, else→动态计算
          Mixtral: batch=64→62, batch=32→70, else→74
        - DeepSeek 的 GPU 专家数远大于 Mixtral（因为 DeepSeek 专家参数量更小）
        - DeepSeek else 分支使用动态计算 free_mem // (n_param * 2) - 930
          Mixtral else 分支返回固定值 74
        - DeepSeek 从 layers[1] 取专家参数 → Mixtral 从 layers[0] 取
        """
        fine_expert = self.model.layers[1].mlp.experts[0]
        n_param = sum(p.numel() for p in fine_expert.parameters())
        LOG(f"Number of parameters in a single expert: {n_param}")
        
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)

        if self.batch_size==64:
            return 893
        elif self.batch_size==32:
            return 993
        elif self.batch_size==16:
            return 693
        else:
            return int((free_mem) // (n_param * 2)-930)


    def initial_beam_tensor(self, input_tensor):
        
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        """文本生成主循环。
        
        【与 Mixtral 版本的差异】
        1. 返回值: DeepSeek 不含 layer_time_avg / layer_time_avg_details / layer_time_details
           Mixtral 返回更详细的分层统计
        2. Attention mask 增量更新方式不同（见下方 L462-483 注释）
        3. 缺少 decode_strings 输出打印（Mixtral 有 print Input/Output）
        """
        torch.set_num_threads(16)
        self.past_key_value = transformers.cache_utils.DynamicCache()
        
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
            # 修复：2D attention_mask 使用2D切片
            attention_mask = attention_mask[:, :input_token]
        
        tick = time.time()
        self.is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        self.token_decode_times = []
        self.perf_stats = {
            'token_embedding': [],
            'self_attention': [],
            'moe_gating': [],
            'expert_compute': [],
            'expert_compute-cpu': []
        }

        for i_token in range(output_token):
            token_start_time = time.time()
            
            if self.is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])
            
            # ===== Attention mask 增量更新 =====
            # Decode 阶段：更新 attention_mask 为 2D (batch, seq_len)
            if self.is_decode:
                past_seq_len = self.past_key_values_length
                attention_mask = torch.ones(
                    input_ids.shape[0], past_seq_len + 1,
                    dtype=torch.long, device=self.dev
                )
            
            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

            # 构建 cache_position
            cache_position = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            )

            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask, cache_position, is_prefill=not self.is_decode)

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

        print("--------------------")
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
                'expert_hot_stats': self.get_expert_stats(),
            }
        )
    
    def tokenize(self, text, input_token=None):
        """将输入文本 tokenize。

        返回2D attention_mask，让 mixtral_forward 中创建 causal mask。
        参考: 10-fiddler-main/src/fiddler/deepseek.py
        """
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")

        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[:self.batch_size]

        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.to(self.dev)

        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # 返回 2D attention_mask，让 mixtral_forward 中创建 causal mask
        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask, cache_position, is_prefill=False):
        """DeepSeek 版核心调度引擎。

        【修复内容】
        1. 添加 RoPE position_embeddings 计算（DeepseekV2RotaryEmbedding）
        2. 使用 create_causal_mask 替代错误的 4D mask 扩展
        3. Layer 0 修复：完整的 attention + dense MLP（非 MoE）
        4. 添加 cache_position 参数用于 causal mask 创建
        5. 添加 is_prefill 参数用于 force_gpu 控制
        参考: 10-fiddler-main/src/fiddler/deepseek.py
        """
        hidden_dim = self.model.config.hidden_size
        force_gpu = is_prefill
        tick = time.time()

        inps = self.model.embed_tokens(input_ids)
        self.perf_stats['token_embedding'].append(time.time() - tick)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 计算 position_embeddings (cos, sin tuple)
        position_embeddings = self.rotary_emb(inps, position_ids)

        # 使用 transformers 官方的 create_causal_mask 创建正确的 causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inps,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=self.past_key_value,
            position_ids=position_ids,
        )

        if self.is_decode:
            total_decode_start = time.time()
            # 【差异】层范围 range(1, self.n_layer) → Mixtral: range(self.n_layer) 即 0-31
            layer_times = {i: 0.0 for i in range(1, self.n_layer)}
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # 【差异】DeepSeek 无 RoPE 预计算
        # Mixtral: position_embeddings = self.model.rotary_emb(inps, position_ids)
        
        layer_start_time = time.time()
        layer_total_time = 0.0
        isprefetch=False

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()

            # ===== Layer 0: 完整的 attention + dense FFN（非 MoE）=====
            # 参考: 10-fiddler-main/src/fiddler/deepseek.py
            if i_layer == 0:
                inps_residual = inps
                inps = layer.input_layernorm(inps)
                inps = inps.view(batch_size, seq_len, hidden_dim)

                attn_output = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                if isinstance(attn_output, tuple):
                    if len(attn_output) == 2:
                        inps, present_key_value = attn_output
                    else:
                        inps, _, present_key_value = attn_output
                else:
                    present_key_value = None

                inps = inps_residual + inps
                inps_residual = inps
                inps = layer.post_attention_layernorm(inps)
                inps = inps.view(batch_size, seq_len, hidden_dim)
                # 第一层使用 dense MLP（非 MoE）
                inps = layer.mlp(inps)
                inps = inps_residual + inps
                continue

            # ===== Layer 1-26: MoE 层 =====
            self.release_placeholder(i_layer, 0)

            original_inps_shape = inps.shape
            self.cpu_expert_time_per_layer[i_layer] =0
            inps_residual = inps
            inps = layer.input_layernorm(inps)

            inps = inps.view(batch_size, seq_len, hidden_dim)
            tick = time.time()

            attn_output = layer.self_attn(
                hidden_states=inps,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            torch.cuda.synchronize()
            self.perf_stats['self_attention'].append(time.time() - tick)

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
                
                
            layer_idx=i_layer
            if layer_idx not in self.layer_data:
                self.layer_data[layer_idx] = {
                    "hidden_states": [],
                    "expert_indices": []
                }
            inps = inps.view(batch_size, seq_len, hidden_dim)

            pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)
            tick = time.time()
            selected_experts,routing_weights = layer.mlp.gate(inps)
            torch.cuda.synchronize()
            self.perf_stats['moe_gating'].append(time.time() - tick)

            layer_expert_stats = {
                'layer_id': i_layer,
                'expert_ids': selected_experts.tolist()
            }
            self.expert_selection_stats.append(layer_expert_stats)

            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.mlp.experts

            shared_output = torch.zeros_like(inps)
            expert_out = self.model.layers[i_layer].mlp.shared_experts(inps)
            shared_output += expert_out
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

            filtered_expert_ids = []
            filtered_token_counts = []
            gpu_onloaded= []
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

            sorted_experts = list(zip(filtered_expert_ids, filtered_token_counts))

            e = 1.39
            tg = 0.95
            n = len(sorted_experts)
            ondemand_experts = []

            cpu_time_table = [float(line.strip())  for line in open('microdeepseek.txt')]
            tic=time.time()
            TA = sum(cpu_time_table[min(tokens, 1498)] for expert_id, tokens in sorted_experts[0:n])
            TC=TA
            experts_in_placeholder = []
            for i in range(n-1):
                expert_id, token_count = sorted_experts[i]

                TG = (1 + i) * e + tg
                TC = TC-cpu_time_table[min(token_count, 1498)]
                LOG(f"e,t,n,i {expert_id} {token_count} {n} {i}", level="DEBUG")
                LOG(f"tg {TG}", level="DEBUG")
                LOG(f"cpu_time_totl[tokens] {TC}", level="DEBUG")

                if self.is_decode:
                    if TG < TC:
                        if token_count>1:
                            ondemand_experts.append(expert_id)
                        else:
                            experts_in_placeholder.append(expert_id)
                        if i==n-2:
                            if TC-TG>e :
                                expert_id2, token_count2 = sorted_experts[i+1]
                                ondemand_experts.append(expert_id2)
                            elif TC-TG>e/2:
                                self.prefil_pre=True
                    else:
                        break
                else:
                    if TG < TC+cpu_time_table[min(token_count, 1498)]:
                        ondemand_experts.append(expert_id)
                    else:
                        break
                    
            LOG(f"time: {(time.time() - tic)*1000:.2f}ms")

            if i_layer < self.n_layer - 1:
                next_layer = self.model.layers[i_layer + 1]

                with torch.no_grad():
                    next_predicted_experts, next_routing_weights = next_layer.mlp.gate(inps)

                expert_token_counts = {}
                for batch_idx in range(batch_size * seq_len):
                    for expert in next_predicted_experts[batch_idx]:
                        expert_token_counts[expert.item()] = expert_token_counts.get(expert.item(), 0) + 1

                sorted_experts = sorted(expert_token_counts.items(), key=lambda x: x[1], reverse=True)

                top3_experts = [expert[0] for expert in sorted_experts[:self.cache]]

                self.hot_experts[i_layer + 1] = []
                for expert_id in top3_experts:
                    token_count = expert_token_counts[expert_id]
                    if self.batch_size==4:
                        if token_count >= 3 and not self.is_expert_in_gpu_now(i_layer + 1, expert_id) and i_layer + 1<len(self.model.layers):
                            self.hot_experts[i_layer + 1].append(expert_id)
                        elif len([e for e in top3_experts if expert_token_counts.get(e, 0) >= 2 and
                                not self.is_expert_in_gpu_now(i_layer + 1, e)]) >= 3:
                            self.hot_experts[i_layer + 1].append(expert_id)
                    elif self.batch_size==8 or self.batch_size==16:
                        if token_count >= 4 and not self.is_expert_in_gpu_now(i_layer + 1, expert_id) and i_layer + 1<len(self.model.layers):
                            self.hot_experts[i_layer + 1].append(expert_id)
                        elif len([e for e in top3_experts if expert_token_counts.get(e, 0) >= 3 and
                                not self.is_expert_in_gpu_now(i_layer + 1, e)]) >= 3:
                            self.hot_experts[i_layer + 1].append(expert_id)
                    else:
                        if token_count >= 4 and not self.is_expert_in_gpu_now(i_layer + 1, expert_id) and i_layer + 1<len(self.model.layers):
                            self.hot_experts[i_layer + 1].append(expert_id)
                        elif len([e for e in top3_experts if expert_token_counts.get(e, 0) >= 3 and
                                not self.is_expert_in_gpu_now(i_layer + 1, e)]) >= 3:
                            self.hot_experts[i_layer + 1].append(expert_id)

            if self.cpu_offload == 0:
                LOG("oo", level="DEBUG")

            else:
                inps_after_experts = torch.zeros_like(inps, device=self.dev)
                expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_expert).permute(2, 1, 0)

                cpu_experts = []
                gpu_experts = []
                experts_in_gpu = []

                experts_loading = []
                experts_remaining = []
                experts_remaining2 = []

                selected_expert_ids = selected_experts.unique().tolist()

                gpu_results = []
                cpu_results = []
                gpu_time = 0.0
                cpu_time = 0.0
                self._prefetch_thread_started=False
                for i_expert in selected_expert_ids:

                    if self.is_expert_in_gpu_now(i_layer, i_expert):
                        gpu_experts.append(i_expert)
                        experts_in_gpu.append(i_expert)
                        continue

                    if i_layer not in self.prefetch_list:
                        self.prefetch_list[i_layer] = []
                    if i_layer not in self.prefetching_list:
                        self.prefetching_list[i_layer] = []
                    if  i_expert in ondemand_experts or i_expert in self.prefetch_list[i_layer] or i_expert in self.prefetching_list[i_layer]:

                        if  i_expert in self.prefetch_list[i_layer]:
                            experts_in_placeholder.append(i_expert)
                        elif i_expert in self.prefetching_list[i_layer]:
                            experts_loading.append(i_expert)
                        else:
                            experts_remaining.append(i_expert)

                    else:
                        cpu_experts.append(i_expert)

                def process_gpu_experts():
                    inps_after_experts = torch.zeros_like(inps, device=self.dev)
                    nonlocal gpu_time
                    start_time = time.time()

                    def process_experts_in_gpu():
                        results = []
                        for i_expert in experts_in_gpu:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)
                            tick = time.time()
                            expert_output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                            self.perf_stats['expert_compute'].append(time.time() - tick)

                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
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
                            placeholder = self.expert_to_placeholder.get((i_layer, i_expert))
                            if placeholder is not None:
                                expert_output = placeholder(expert_input)
                            else:
                                expert_output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                            self.perf_stats['expert_compute'].append(time.time() - tick)
                            
                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
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
                            while i_expert in self.prefetching_list.get(i_layer, []):
                                time.sleep(0.0001)
                            tick = time.time()
                            placeholder = self.expert_to_placeholder.get((i_layer, i_expert))
                            if placeholder is not None:
                                expert_output = placeholder(expert_input)
                            else:
                                expert_output = self.run_expert_at_gpu(i_layer, i_expert, expert_input)
                            self.perf_stats['expert_compute'].append(time.time() - tick)
                            
                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            results.append((mask_index, expert_output))
                        return results

                    def process_experts_remaining():
                        results = []
                        threads = []
                        expert_results = {}

                        def process_single_expert(i_expert, placeholder):
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                return
                            batch_mask = mask.view(batch_size, seq_len)
                            expert_input = inps[batch_mask].view(-1, hidden_dim)
                            self._async_ondemand(i_layer, i_expert, placeholder)
                            expert_output = placeholder(expert_input)

                            flat_mask = mask.view(-1)
                            weights = routing_weights[flat_mask].gather(
                                1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
                            )
                            expert_output = expert_output * weights
                            mask_index = mask.nonzero().squeeze(1)
                            with threading.Lock():
                                results.append((mask_index, expert_output))

                        for i_expert in experts_remaining:
                            if not self.expert_placeholder5_inused:
                                placeholder = self.expert_placeholder5
                                self.expert_placeholder5_inused = True
                            elif not self.expert_placeholder6_inused:
                                placeholder = self.expert_placeholder6
                                self.expert_placeholder6_inused = True
                            elif not self.expert_placeholder3_inused:
                                placeholder = self.expert_placeholder3
                                self.expert_placeholder3_inused = True
                            elif not self.expert_placeholder4_inused:
                                placeholder = self.expert_placeholder4
                                self.expert_placeholder4_inused = True
                            elif not self.expert_placeholder_inused:
                                placeholder = self.expert_placeholder
                                self.expert_placeholder_inused = True
                            elif not self.expert_placeholder2_inused:
                                placeholder = self.expert_placeholder2
                                self.expert_placeholder2_inused = True
                            else:
                                experts_remaining2.append(i_expert)
                                continue
                            t = threading.Thread(
                                target=process_single_expert,
                                args=(i_expert, placeholder)
                            )
                            threads.append(t)
                            t.start()
                            for t in threads:
                                t.join()
                            threads = []
                            self.expert_placeholder5_inused = False
                            self.expert_placeholder6_inused = False
                            self.expert_placeholder3_inused = False
                            self.expert_placeholder4_inused = False
                            self.expert_placeholder_inused = False
                            self.expert_placeholder2_inused = False
                        for i_expert in experts_remaining2:
                            if not self.expert_placeholder5_inused:
                                placeholder = self.expert_placeholder5
                                self.expert_placeholder5_inused = True
                            elif not self.expert_placeholder6_inused:
                                placeholder = self.expert_placeholder6
                                self.expert_placeholder6_inused = True
                            elif not self.expert_placeholder3_inused:
                                placeholder = self.expert_placeholder3
                                self.expert_placeholder3_inused = True
                            elif not self.expert_placeholder4_inused:
                                placeholder = self.expert_placeholder4
                                self.expert_placeholder4_inused = True
                            elif not self.expert_placeholder_inused:
                                placeholder = self.expert_placeholder
                                self.expert_placeholder_inused = True
                            elif not self.expert_placeholder2_inused:
                                placeholder = self.expert_placeholder2
                                self.expert_placeholder2_inused = True
                            else:
                                continue
                            t = threading.Thread(
                                target=process_single_expert,
                                args=(i_expert, placeholder)
                            )
                            threads.append(t)
                            t.start()
                            self.expert_placeholder5_inused = False
                            self.expert_placeholder6_inused = False
                            self.expert_placeholder3_inused = False
                            self.expert_placeholder4_inused = False
                            self.expert_placeholder_inused = False
                            self.expert_placeholder2_inused = False

                        return results

                    threads = []
                    results = []

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
                        expert_output = expert_output.view(-1, hidden_dim)
                        inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                        inps_after_experts.index_add_(
                            0,
                            mask_index,
                            expert_output.to(inps_after_experts.dtype)
                        )
                    inps_after_experts = inps_after_experts.view(batch_size, seq_len, hidden_dim)
                    gpu_time = time.time() - start_time

                def process_cpu_experts():
                    nonlocal cpu_time
                    start_time = time.time()
                    for i_expert in cpu_experts:
                        mask = (selected_experts == i_expert).any(dim=1)
                        if not mask.any():
                            continue

                        batch_mask = mask.view(batch_size, seq_len)

                        expert_input = inps[batch_mask].view(-1, hidden_dim).to("cpu")

                        tick = time.time()
                        expert_output = self.run_expert_at_cpu(i_layer, i_expert, expert_input )
                        self.perf_stats['expert_compute-cpu'].append(time.time() - tick)
                        flat_mask = mask.view(-1)
                        weights = routing_weights[flat_mask].gather(
                            1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
                        ).to("cpu")
                        expert_output = expert_output * weights
                        mask_index = mask.nonzero().squeeze(1)
                        cpu_results.append((mask_index, expert_output))
                    cpu_time = time.time() - start_time

                def prefetch_experts():
                    hot_experts = self.hot_experts

                    if self.batch_size == 4:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1
                    elif self.batch_size == 8:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1
                    else:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1

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
                            self.prefetching_list[self.prefetch_layers].append(expert_id)
                            time.sleep(0.0014)
                            self.prefetch_list[self.prefetch_layers].append(expert_id)
                            self.prefetching_list[self.prefetch_layers]=[]

                            experts_loaded += 1
                            if experts_loaded >= expert_count:
                                break

                parallel_start = time.time()
                prefetch_thread = threading.Thread(target=prefetch_experts)
                gpu_thread = threading.Thread(target=process_gpu_experts)
                cpu_thread = threading.Thread(target=process_cpu_experts)

                if self.is_decode and self.batch_size>16:
                    if self._prefetch_thread_started==False:
                        prefetch_thread.start()
                        self._prefetch_thread_started = True
                gpu_thread.start()
                cpu_thread.start()

                gpu_thread.join()
                cpu_thread.join()

                parallel_time = time.time() - parallel_start

                max_thread_time = max(gpu_time, cpu_time)
                parallel_degree = (gpu_time + cpu_time) / parallel_time if parallel_time > 0 else 1.0

                if self.is_decode:
                    LOG(f"\nLayer {i_layer} Thread Time Stats:")
                    LOG(f"GPU Thread Time: {gpu_time*1000:.2f}ms")
                    LOG(f"CPU Thread Time: {cpu_time*1000:.2f}ms")
                    LOG(f"Parallel Time: {parallel_time*1000:.2f}ms")
                    LOG(f"Parallel Degree: {parallel_degree:.2f}x")

                for mask_index, expert_output in cpu_results:
                    expert_output = expert_output.view(-1, hidden_dim)
                    inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                    inps_after_experts.index_add_(
                        0,
                        mask_index.to(self.dev),
                        expert_output.to(self.dev).to(inps_after_experts.dtype)
                    )
                    inps_after_experts = inps_after_experts.view(batch_size, seq_len, hidden_dim)

            total_expert_output = shared_output + inps_after_experts

            inps = inps_residual + total_expert_output.reshape(batch_size, seq_len, hidden_dim)

            if self.is_decode:
                layer_time = time.time() - layer_tick
                self.layer_time_accumulator[i_layer] += layer_time

                self.layer_time_stats.append({
                    'layer_id': i_layer,
                    'time': layer_time,
                    'token_step': self.past_key_values_length
                })

                layer_times[i_layer] += layer_time

                self.layer_time_stats.append({
                    'layer_id': i_layer,
                    'time': layer_time,
                    'token_step': self.past_key_values_length
                })

                layer_times[i_layer] += layer_time

        # ===== 所有层循环结束 =====
        hot_experts = self.get_hot_expert()
        
        # decode 阶段输出逐层耗时日志
        # 【差异】层范围 range(1, self.n_layer) → Mixtral: range(self.n_layer)
        if self.is_decode:
            for i in range(1, self.n_layer):
                layer_time = layer_times[i] * 1000

            avg_layer_time = sum(layer_times.values()) / self.n_layer
            total_cpu_time = sum(self.cpu_expert_time_per_layer.values())
            avg_cpu_ratio = (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0
            
            os.makedirs('./log', exist_ok=True)
            with open('./log/expert_stats.txt', 'a') as f:
                for i in range(1, self.n_layer):
                    cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                    layer_time = layer_times[i] * 1000

        # 最终 LayerNorm + lm_head
        other_ops_start = time.time()
        LOG("ok", level="DEBUG")
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start
            
        self.present_key_value = present_key_value
        return lm_logis

    def run_expert_at_gpu(self, i_layer, i_expert, inps ):
        """在 GPU 上执行指定专家。
        【差异】专家路径: layer.mlp.experts → Mixtral: layer.block_sparse_moe.experts
        """
        start_time = time.time()
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        token_count = inps.shape[0]
        expert_status = 'normal'
        if token_count > 2:
            expert_status = 'veryhot'
        elif token_count > 1:
            expert_status = 'hot'
        
        self.current_iter_expert_stats[i_layer]['expert_ids'].append(i_expert)
        self.current_iter_expert_stats[i_layer]['token_counts'].append(token_count)

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
        【差异】专家路径: layer.mlp.experts → Mixtral: layer.block_sparse_moe.experts
        """
        start_time = time.time()
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed
        token_count = inps.shape[0]
        expert_status = 'normal'
        if token_count > 2:
            expert_status = 'veryhot'
        elif token_count > 1:
            expert_status = 'hot'
        
        self.current_iter_expert_stats[i_layer]['expert_ids'].append(i_expert)
        self.current_iter_expert_stats[i_layer]['token_counts'].append(token_count)

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
        """汇总专家运行统计信息。
        【差异】层范围 range(1, self.n_layer) → Mixtral: range(self.n_layer)
        """
        stats = {
            'hot_experts': {i: {'count': 0, 'hot': 0, 'veryhot': 0} for i in range(1, self.n_layer)},
            'hot_counts': {2: 0, 3: 0, 4: 0, 5: 0},
            'token_distribution': {}
        }
        
        for record in self.expert_time_stats:
            layer = record['layer_id']
            token_count = record['token_count']
            expert_status = record['status']
            
            stats['hot_experts'][layer]['count'] += 1
            if expert_status == 'hot':
                stats['hot_experts'][layer]['hot'] += 1
            elif expert_status == 'veryhot':
                stats['hot_experts'][layer]['veryhot'] += 1
        
        layer_hot_counts = {i: 0 for i in range(1, self.n_layer)}
        for record in self.expert_time_stats:
            if record['status'] in ['hot', 'veryhot']:
                layer_hot_counts[record['layer_id']] += 1
                
        for count in layer_hot_counts.values():
            if count >= 2 and count <= 5:
                stats['hot_counts'][count] += 1
                
        stats['token_distribution'] = {}
        token_counts = [r['token_count'] for r in self.expert_time_stats]
        unique_counts = set(token_counts)
        
        for count in unique_counts:
            stats['token_distribution'][count] = token_counts.count(count)
                
        return stats
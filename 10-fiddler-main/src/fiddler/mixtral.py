import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class FiddlerMixtral:
    """
    Fiddler 推理引擎核心类。
    
    核心思想：不同于传统方法将 CPU 内存中的专家权重搬运到 GPU 执行，
    Fiddler 将激活值传到 CPU，直接在 CPU 上执行专家计算，再将输出传回 GPU。
    激活值远小于权重（batch_size×4096 vs 3×4096×14336），大幅降低通信开销。
    """

    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")

        # 加载 Mixtral-8x7B 模型（模型初始全部在 CPU/磁盘上）
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            # device_map='cpu',
            use_cache=True,
        )
        # 分离 lm_head（语言模型头）和 model（Transformer 主体）
        # 因为后续需要分别管理它们的设备放置
        self.lm_head = self.model.lm_head
        self.model = self.model.model

        # 在 GPU 上预分配一个专家权重的占位符，用于临时加载 CPU 上的专家到 GPU 执行
        # 避免每次动态分配显存的开销
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # KV Cache 初始化（每次 generate() 时会重置）
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cpu_offload = args.cpu_offload  # 0: 全部GPU执行(基线), 1: CPU-GPU协同执行
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)      # Mixtral-8x7B 有32层
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)  # 每层8个专家

        # CPU/GPU 上执行单个专家的预估延迟（单位未指定，用于负载均衡计算的相对值）
        # TODO: find this value based on device config
        self.latency_cpu = 7   # CPU 上每 token 的延迟
        self.latency_gpu = 70  # 将专家权重从 CPU 拷贝到 GPU 的固定延迟

        # 专家命中统计（用于计算 cache hit rate）
        self.cnt_expert_hit = 0  # 命中 GPU 常驻专家的 token 数
        self.cnt_expert_all = 0  # 总 token 数

        # 第一步：将所有非专家层（attention、layer norm、gate、embedding、lm_head）放到 GPU
        self.bring_non_expert_to_gpu()

        # 第二步：根据 GPU 剩余显存，计算能容纳多少个专家，并按"热度"优先级选择常驻 GPU 的专家
        # expert_loc[i][j] = 0 表示第i层第j个专家在 CPU, = 1 表示在 GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        # 根据预分析的专家热度排序，将最热门的 n_expert_on_gpu 个专家标记为 GPU 常驻
        self.set_expert_loc(n_expert_on_gpu)

        # 第三步：将标记为 GPU 的专家实际加载到 GPU
        self.bring_expert_to_gpu()

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """将所有非专家层加载到 GPU。
        
        包括：lm_head、embed_tokens、norm、每层的 self_attn、
        input_layernorm、gate（路由器）、post_attention_layernorm。
        专家层（block_sparse_moe.experts）保留在 CPU，后续按需选择部分加载到 GPU。
        """
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)  # MoE路由器放GPU
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # 专家层（block_sparse_moe.experts）保留在 CPU

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """根据热度排序，将前 n_expert_on_gpu 个最热门专家标记为 GPU 常驻。
        
        Args:
            n_expert_on_gpu: GPU 显存能容纳的专家数量
            popular_experts: 按热度降序排列的 (层号, 专家号) 列表，
                            默认使用预分析的固定排序结果
        """
        if popular_experts is None:
            # 基于离线 profile 得到的专家热度排序（最热门的排在最前面）
            # 热度越高 = 在推理中被路由选中的频率越高，放在 GPU 上命中概率更大
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
                (24, 6),
                (25, 2),
                (28, 4),
                (4, 6),
                (7, 2),
                (20, 3),
                (23, 2),
                (8, 4),
                (30, 0),
                (3, 4),
                (12, 5),
                (23, 7),
                (1, 7),
                (22, 5),
                (18, 4),
                (31, 0),
                (17, 0),
                (0, 5),
                (14, 6),
                (0, 3),
                (15, 7),
                (5, 6),
                (4, 4),
                (24, 7),
                (31, 1),
                (27, 6),
                (22, 2),
                (14, 1),
                (1, 0),
                (29, 1),
                (21, 3),
                (25, 7),
                (22, 3),
                (7, 3),
                (2, 6),
                (29, 5),
                (28, 3),
                (6, 6),
                (7, 5),
                (5, 7),
                (8, 5),
                (20, 4),
                (21, 5),
                (18, 7),
                (27, 0),
                (16, 0),
                (24, 5),
                (12, 2),
                (2, 2),
                (24, 3),
                (4, 1),
                (29, 0),
                (3, 1),
                (21, 6),
                (10, 2),
                (20, 7),
                (19, 0),
                (26, 7),
                (20, 6),
                (23, 3),
                (4, 3),
                (30, 1),
                (1, 6),
                (29, 2),
                (30, 3),
                (0, 6),
                (8, 1),
                (25, 6),
                (29, 4),
                (16, 2),
                (23, 1),
                (26, 1),
                (26, 6),
                (16, 4),
                (2, 5),
                (0, 4),
                (7, 6),
                (14, 4),
                (3, 6),
                (20, 0),
                (18, 3),
                (4, 5),
                (17, 4),
                (0, 1),
                (16, 5),
                (19, 3),
                (23, 0),
                (30, 4),
                (20, 2),
                (13, 6),
                (18, 6),
                (15, 2),
                (3, 5),
                (22, 0),
                (10, 1),
                (9, 6),
                (10, 5),
                (25, 4),
                (9, 2),
                (18, 1),
                (6, 4),
                (4, 2),
                (23, 5),
                (6, 5),
                (21, 2),
                (5, 5),
                (6, 1),
                (26, 5),
                (12, 0),
                (25, 0),
                (4, 0),
                (14, 0),
                (16, 6),
                (31, 2),
                (8, 0),
                (21, 7),
                (14, 3),
                (31, 6),
                (28, 1),
                (5, 3),
                (23, 6),
                (6, 3),
                (18, 5),
                (25, 5),
                (27, 1),
                (11, 7),
                (11, 4),
                (24, 1),
                (0, 7),
                (8, 7),
                (13, 3),
                (21, 4),
                (27, 4),
                (13, 7),
                (3, 2),
                (9, 1),
                (2, 7),
                (7, 0),
                (2, 3),
                (28, 5),
                (27, 3),
                (15, 0),
                (24, 4),
                (5, 2),
                (22, 6),
                (3, 3),
                (28, 6),
                (14, 7),
                (13, 4),
                (28, 7),
                (22, 7),
                (13, 5),
                (19, 1),
                (26, 4),
                (1, 1),
                (17, 1),
                (16, 3),
                (10, 7),
                (29, 6),
                (19, 7),
                (31, 3),
                (7, 7),
                (1, 3),
                (8, 2),
                (9, 4),
                (17, 3),
                (30, 5),
                (15, 3),
                (9, 3),
                (10, 6),
                (12, 7),
                (11, 3),
            ]

        # 从热度列表头部开始，将前 n_expert_on_gpu 个专家标记为 GPU 常驻
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        """将 expert_loc 中标记为 GPU 的专家实际加载到 GPU 显存。"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        """判断指定层指定专家是否常驻在 GPU 上"""
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """根据 GPU 剩余显存计算能容纳多少个专家。
        
        计算方式：剩余显存 / 单个专家参数占用（bfloat16 每参数2字节）
        """
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        # 0.95 是经验值，预留一部分显存给激活值等临时数据
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev)
        return int((free_mem) // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        """在 beam search 第一步中，将 topk 结果从 (beam_width, seq_len, beam_width) 
        正确转换为 (beam_width, 1)。
        
        topk 在最后一个维度取 beam_width 个候选，产生 (beam_width, seq_len, beam_width)。
        该函数取最后一个时间步 [:, -1]，然后按 [0, beam_width, 2*beam_width, ...] 间隔采样，
        将每个 beam 的第 i 个候选配对，确保 beam_width 个分支各自获得不同的候选 token。
        """
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        """文本生成主循环，支持 beam search 解码。
        
        流程：
        1. 初始化空的 KV Cache
        2. Prefill 阶段：输入完整 prompt，填充 KV Cache
        3. Decode 阶段：每次输入1个 token，从 KV Cache 读取历史，逐步生成
        4. 如果 beam_width > 1，第一步展开为 beam_width 条路径，
           后续每条路径独立贪心，最后选概率最高的路径
        
        注意：这里没有使用 transformers 的 model.generate()，因为 Fiddler 需要在
        mixtral_forward 中手动控制 MoE 层的 CPU-GPU 专家调度，无法走标准 forward 路径。
        因此 beam search 和 KV Cache 管理都必须独立实现。
        """
        torch.set_num_threads(16)
        # 重置 KV Cache（步骤一：初始化）
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        
        # tokenize 输入文本，复制 beam_width 份作为 batch
        input_ids, position_ids = self.tokenize(text)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]

        tick = time.time()
        is_decode = False  # 第一个迭代是 prefill，之后都是 decode
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]  # 每个 beam 的解码结果
        search_start = False  # 标记 beam search 是否已展开
        probs = torch.full((input_ids.shape[0], 1), 1.0)  # 每个 beam 的累积概率

        for i_token in range(output_token):
            if self.beam_width == 1:
                print(self.tokenizer.decode(input_ids[0]))
            if is_decode:
                # decode 阶段记录每个 beam 的解码字符串
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            # 核心前向传播（包含 CPU-GPU 专家调度）
            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # 将 logits 转为概率分布
            logits = F.softmax(logits, dim=-1)

            # 步骤三：更新已缓存的序列长度
            self.past_key_values_length += logits.shape[1]

            # 简化版 Beam Search：
            # - 第一步：topk 取 beam_width 个候选，展开为 beam_width 条路径
            # - 后续步：每条路径各自取 top-1，独立贪心，不做全局排序剪枝
            # （非标准 full beam search，缺少每步优胜劣汰机制）
            if search_start:
                # 后续步：每个 beam 只取 top-1
                new_probs, output = torch.topk(logits, 1, dim=-1)
                # new_probs.shape: (batch_size, 1), 这个1是 top-1 的结果-概率
                # output.shape: (batch_size, 1)， 1是 top-1 的 token id，位置与 new_probs 对应
                # 如果是 top-k，则 new_probs.shape: (batch_size, k)，output.shape: (batch_size, k)

                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                # 第一步：展开为 beam_width 条路径
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            probs = probs * new_probs  # 累积路径概率

            # 取最后一个 token 作为下一步的输入（decode 阶段每次只有1个 token）
            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            #input_ids.shape: (batch_size, 1)
            #如果上面是top-k，则input_ids.shape: (batch_size* k, 1)


            # 步骤四：根据已缓存长度计算新的 position_ids
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

            # 计时：prefill 阶段和 decode 阶段分别计时
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        decode_time = time.time() - tick

        # 从所有 beam 中选择累积概率最高的路径
        probs = probs.view(-1, self.beam_width) # probs.shape: (beam_width, 1)-> (beam_width,)
        max_ids = torch.argmax(probs, dim=-1)

        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
        )

    def tokenize(self, text):
        """将输入文本 tokenize，并复制 beam_width 份以支持 beam search 的 batch 推理。"""
        input_ids = []
        encodings = self.tokenizer(text, return_tensors="pt")
        input_id = encodings.input_ids.to(self.dev)
        for i in range(self.beam_width):
            input_ids.append(input_id[0])
        
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        """自定义前向传播，手动逐层执行 Mixtral 模型。
        
        为什么不使用 model.forward()？因为需要在 MoE 层插入自定义的 CPU-GPU 专家调度逻辑，
        标准 model.forward() 会让所有专家在 GPU 上执行，无法实现 Fiddler 的核心创新。
        
        每层的执行流程：
        1. Input LayerNorm → Self-Attention（使用 KV Cache）→ 残差连接
        2. Post-Attention LayerNorm → MoE 路由（gate 选择 top-2 专家）
        3. 专家计算（核心：CPU-GPU 协同调度）→ 残差连接
        """
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        # Token Embedding
        inps = self.model.embed_tokens(inps)

        for i_layer, layer in enumerate(self.model.layers):
            original_inps_shape = inps.shape

            # ===== Self-Attention 部分 =====
            inps_residual = inps
            inps = layer.input_layernorm(inps)
            # Self-Attention：传入 past_key_value（KV Cache），
            # prefill 时填充 cache，decode 时读取+追加 cache
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            # inps.shape: (batch_size, seq_len/token_num, embed_dim)
            inps = inps_residual + inps  # 残差连接

            # ===== MoE (Mixture of Experts) 部分 =====
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            # 展平为 (batch_size*seq_len, hidden_dim)，每个 token 独立路由
            inps = inps.view(-1, hidden_dim)

            # Gate（路由器）：为每个 token 计算 8 个专家的路由权重
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            # 每个 token 选择 top-2 专家（Mixtral 的标准策略）
            routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
            # 归一化路由权重，使两个专家的权重之和为1
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # 累积所有专家的输出
            # 创建一个与输入 tensor 形状、数据类型、设备完全相同的全零 tensor
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            if self.cpu_offload == 0:
                # ===== 模式一：基线模式（cpu_offload=0）=====
                # 所有专家都在 GPU 上执行。
                # 如果专家不在 GPU 常驻，则临时将其权重加载到 GPU 的 placeholder 中执行。
                # selected_experts shape: (batch*seq_len, 2)  — 每个token选了哪2个专家
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)  # (num_experts, top2_or_idx, batch*seq_len)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    # expert_mask[i_expert] shape: (2, batch*seq_len)
                    # 这里top2和idx的含义相反，idx表示行坐标（0或1），top2反而是哪个token被路由到该专家了
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # 该专家没有被任何 token 选中，跳过
                        continue

                    # 转为 Python list 用于索引（tolist 比 tensor 索引在重复使用时更快）
                    top_2_list = top_2.tolist()  # list[int], 长度=被路由到该专家的token数
                    idx_list = idx.tolist()       # list[int], 同上

                    # inps shape: (batch*seq_len, hidden_dim)
                    # inps[None, top_2_list] shape: (1, n_tokens, hidden_dim)
                    #   None 在第0维插入一个维度，top_2_list 沿第1维索引
                    # .reshape(-1, hidden_dim) → (n_tokens, hidden_dim)
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)

                    # ！！如果专家不在 GPU 常驻，则先将其权重加载到 GPU 的 placeholder 中执行
                    if not is_cuda:
                        # 专家在 CPU：将权重加载到 GPU 的 placeholder，在 GPU 上执行
                        # expert_placeholder 是预先在 GPU 上分配的专家模块（见 __init__ 第24-26行）
                        # load_state_dict 只复制权重参数，复用同一块 GPU 显存，避免动态分配
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        # routing_weights[top_2_list, idx_list, None] shape: (n_tokens, 1)
                        #   top_2_list 选 token，idx_list 选第1/第2专家对应的权重
                        #   None 在末尾插入一个维度，变为 (n_tokens, 1)，用于广播乘法
                        # expert 的输入是 (current_state * routing_weights)，即加权后的激活值
                        # expert 输出 shape: (n_tokens, hidden_dim)
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        # 专家已在 GPU：直接执行
                        # 维度同上: (n_tokens, hidden_dim) → (n_tokens, hidden_dim)
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )

                    # index_add_ 原地累加：将当前专家的输出加到 inps_after_experts 对应位置
                    # inps_after_experts shape: (batch*seq_len, hidden_dim)
                    # 0 表示沿第0维（token维度）操作
                    # top_2 是目标位置的索引 tensor，形状 (n_tokens,) — 哪些位置需要累加
                    # current_state shape: (n_tokens, hidden_dim)— 当前专家对这些 token 的输出
                    
                    # 效果：inps_after_experts[top_2[j]] += current_state[j]
                    # 每个 token 会被 2 个专家处理，两次 index_add_ 完成加权求和
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        # 防御性写法：确保 experts[i_expert] 仍在 CPU 上
                        # load_state_dict 只读取参数值，不会移动源模型到 GPU
                        # 但 GPU 显存极其紧张（24GB 跑 90GB 模型），一旦泄漏就 OOM
                        # 所以显式确保该专家在 CPU 上，防止意外占用 GPU 显存
                        experts[i_expert] = experts[i_expert].to("cpu")

            else:
                # ===== 模式二：Fiddler 核心模式（cpu_offload=1）=====
                # CPU-GPU 协同执行专家计算，流程：
                # 1. 统计每个专家被分配的 token 数量
                # 2. 估算每个专家在 CPU/GPU 上执行的延迟
                # 3. 遍历所有 2^8=256 种 CPU/GPU 分配方案，选总延迟最小的
                # 4. GPU 上的专家在 GPU 执行，CPU 上的专家在 CPU 执行（传激活值而非权重）
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # 第一步：统计每个专家的 token 数，计算 CPU/GPU 延迟
                idxs, top_2s = [], []
                cost_per_expert = np.zeros(
                    (len(experts), 2), dtype=float
                )  # [专家编号, 0:CPU延迟 1:GPU延迟]
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    # CPU 延迟与 token 数成正比（逐 token 计算）
                    cost_per_expert[i_expert, 0] = top_2.shape[0] * self.latency_cpu
                    # GPU 延迟是固定的（主要是权重搬运开销）
                    cost_per_expert[i_expert, 1] = self.latency_gpu
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # 专家已在 GPU 常驻，GPU 延迟约等于0（无需搬运权重）
                        cost_per_expert[i_expert, 1] = 0
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                
                # 第二步：暴力搜索最优 CPU/GPU 分配方案
                # 目标：最小化 max(CPU总延迟, GPU总延迟)，即 CPU 和 GPU 尽量并行完成
                # 由于只有 8 个专家，2^8=256 种方案可以暴力枚举
                best_config = -1
                best_cost = float("inf")
                for config in range(1 << len(experts)):
                    sum_cost = 0
                    for i_expert in range(len(experts)):
                        if (config >> i_expert) & 1:
                            sum_cost += cost_per_expert[i_expert, 0]  # 该专家放CPU的延迟
                        else:
                            sum_cost += cost_per_expert[i_expert, 1]  # 该专家放GPU的延迟
                    if sum_cost < best_cost:
                        best_cost = sum_cost
                        best_config = config

                # 第三步：根据最优配置，将专家分为 CPU 组和 GPU 组
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(8):
                    if (best_config >> i_expert) & 1:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)

                # 第四步：先执行 GPU 组的专家
                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # 专家常驻 GPU，直接执行
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        # 专家不在 GPU：加载权重到 placeholder 执行
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

                # 第五步：执行 CPU 组的专家（核心创新）
                # 将激活值从 GPU 传到 CPU，在 CPU 上执行专家计算，再将结果传回 GPU
                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    # 将激活值传到 CPU 执行（而非将权重传到 GPU）
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                        routing_weights[top_2_list, idx_list, None].to("cpu"),
                    )
                    # 将 CPU 计算结果传回 GPU
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

            # MoE 残差连接：将专家输出加回残差
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

        # 所有层处理完毕后，经过最终的 LayerNorm 和 lm_head 得到 logits
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """在 CPU 上执行指定的专家层。
        
        这是 Fiddler 的核心创新点：
        传统方法是将专家权重从 CPU 拷贝到 GPU 再执行；
        Fiddler 是将激活值从 GPU 拷贝到 CPU，在 CPU 上执行专家，再将结果拷回 GPU。
        由于激活值远小于权重，通信开销大幅降低。
        """
        return self.model.layers[i_layer].block_sparse_moe.experts[i_expert](
            inps, routing_weights
        )

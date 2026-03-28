import copy
import threading
import time
import os  # 添加这行导入
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
import threading

class FiddlerMoon:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.hot_experts = {}
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            use_cache=True,
            trust_remote_code=True
        )
        self.batch_size = args.batch_size  # 添加batch_size参数
        self.batch_size = args.batch_size  # 添加batch_size参数   
        if self.batch_size==4:
           self.cache=5    
        else:
           self.cache=10        
        self.prefetch_list = {}  # 记录每层专家选择历史
        self.prefetching_list = {}  # 记录每层专家选择历史   
        self.prefil_pre=False                           
        self.lm_head = self.model.lm_head
        self.model = self.model.model

        first_layer_mlp = self.model.layers[1].mlp
        # mlp_layer = first_layer.mlp
        # print(f"MLP 层结构: {self.model}")        
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
        self.expert_placeholder5 = copy.deepcopy(
            first_layer_mlp.experts[4]
        ).to(self.dev)
        self.expert_placeholder6 = copy.deepcopy(
            first_layer_mlp.experts[5]
        ).to(self.dev)
        self.expert_to_placeholder = {}  # 添加这行初始化字典
        self.expert_placeholder_inused=False 
        self.expert_placeholder2_inused=False 
        self.expert_placeholder3_inused=False   
        self.expert_placeholder4_inused=False    
        self.expert_placeholder5_inused=False   
        self.expert_placeholder6_inused=False              
        self.prefetch_layers=0   
        self.is_decode = False
        self.placeholder_to_expert = {
            'expert_placeholder': None,
            'expert_placeholder2': None,
            'expert_placeholder3': None,
            'expert_placeholder4': None
        }         
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.n_layer = len(self.model.layers)
        # self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
        self.n_expert = self.model.config.n_routed_experts  # 64
        self.n_shared_experts = 2  # 2
        self.expert_selection_stats = []  # 记录专家选择情况
        self.expert_time_stats = []       # 记录专家处理时间

        self.expert_selection_history = {}  # 记录每层专家选择历史
        self.hit_stats = {}  # 记录命中统计
        for i in range(1, 27):
            if i>0:
                self.expert_selection_history[i] = []
                self.hit_stats[i] = {'hits': 0, 'total': 0}
            #yed
        self.expert_weight_accumulator = {}  # 记录每层专家的累计权重
        for i in range(1, 27):
            if i>0:
                self.expert_weight_accumulator[i] = torch.zeros(6, device=self.dev)  
        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(1, 27)}
        # TODO: find this value based on device config
        self.latency_cpu = 5
        self.latency_gpu = 45

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{(self.n_layer-1) * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)
        # print(self.expert_loc)
        self.layer_time_stats = []  # 记录每层处理时间
        self.layer_time_accumulator = {}  # 记录每层累计时间
        for i in range(1, 27):
            self.layer_time_accumulator[i] = 0.0
        self.layer_time_details = {
            'all_gpu': [],    # 两个专家都在GPU
            'all_cpu': [],    # 两个专家都在CPU
            'mixed': []       # 一个GPU一个CPU
        }
        #基于历史预测
        self.layer_time_accumulator_details = {
            'all_gpu': {i: 0.0 for i in range(1, 27)},
            'all_cpu': {i: 0.0 for i in range(1, 27)},
            'mixed': {i: 0.0 for i in range(1, 27)}
        }
        self.last_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []} 
            for i in range(1, 27)
        }
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(1, 27)
        }

        self.layer_data = {}  # 用于存储收集的数据
        # self._register_data_collection_hooks()  # 注册数据收集钩子

        tick = time.time()        
        self.bring_expert_to_gpu()
        print(f"专家 移动总耗时: {(time.time() - tick)*1000:.2f}ms")
        print("Model is ready.")



    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        self.model.layers[0].to(self.dev)            
        for i in range(len(self.model.layers)):
            if i!=0:
                self.model.layers[i].self_attn.to(self.dev)
                self.model.layers[i].input_layernorm.to(self.dev)
                self.model.layers[i].mlp.gate.to(self.dev)  # 注意：这里是mlp.gate
                self.model.layers[i].post_attention_layernorm.to(self.dev)
        for i in range(1, 27):
            if i>0:
                self.model.layers[i].mlp.shared_experts.to(self.dev)
    def get_hot_expert(self):
        """获取每层热点专家(按处理token数量排序)"""
        if not hasattr(self, 'is_decode') or not self.is_decode:
            return {}
        
        hot_experts = {}
        
        for layer_id in range(self.n_layer):
            # 获取当前层的专家统计
            if layer_id>0:
                expert_ids = self.current_iter_expert_stats[layer_id]['expert_ids']
                token_counts = self.current_iter_expert_stats[layer_id]['token_counts']
                
                # 合并专家ID和token数量
                expert_data = list(zip(expert_ids, token_counts))
                
                # 按token数量降序排序
                sorted_experts = sorted(expert_data, key=lambda x: x[1], reverse=True)
                
                # 提取排序后的专家ID
                hot_experts[layer_id] = [expert[0] for expert in sorted_experts]
                
                # 更新last_iter记录
                self.last_iter_expert_stats[layer_id] = {
                    'expert_ids': expert_ids.copy(),
                    'token_counts': token_counts.copy()
                }
                
                # 清空当前迭代记录
                self.current_iter_expert_stats[layer_id]['expert_ids'].clear()
                self.current_iter_expert_stats[layer_id]['token_counts'].clear()
        
        # 保存到成员变量
        self.hot_experts = hot_experts
        return hot_experts
    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts"""

        if popular_experts is None:
            hot_experts_file = './hot/moon.txt'
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, 'r') as f:
                        popular_experts = [tuple(map(int, line.strip().split(','))) 
                                         for line in f if line.strip()]
                    print(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    print(f"Error loading hot experts: {e}")
            else:
                popular_experts = []
                for layer in range(1, 27):  # 1-27层
                    for expert in range(40):  # 每层前60个专家
                        popular_experts.append((layer, expert))
        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            if i_layer < self.n_layer and i_layer!=0:
                self.expert_loc[i_layer, i_expert] = 1



    def _async_ondemand(self, layer_idx, expert_id, target_placeholder):
        expert = self.model.layers[layer_idx].mlp.experts[expert_id]
    
        # 如果专家已经在GPU上，直接返回
        if next(expert.parameters()).is_cuda:
            return 
        
        # 创建CUDA流用于并行传输
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data 
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned

        tick = time.time()
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
            dst.copy_(src)
            
        copytime = time.time() - tick
        print(f"ondemand专家: 层 {layer_idx} 专家 {expert_id} -> {copytime*1000:.2f}ms")

    def _async_load_expert(self, layer_idx, expert_id):

        expert = self.model.layers[layer_idx].mlp.experts[expert_id]
    
        # 如果专家已经在GPU上，直接返回
        if next(expert.parameters()).is_cuda:
            return 
        # 创建CUDA流用于并行传输
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name)
            src_weight_data_tensor = w.weight.data 
            pinned = src_weight_data_tensor.pin_memory()
            w.weight.data = pinned
        target_placeholder = None
        if not self.expert_placeholder_inused:
            target_placeholder = self.expert_placeholder
            self.expert_placeholder_inused = True
        elif not self.expert_placeholder2_inused:
            target_placeholder = self.expert_placeholder2
            self.expert_placeholder2_inused = True
        elif not self.expert_placeholder3_inused:
            target_placeholder = self.expert_placeholder3
            self.expert_placeholder3_inused = True
        elif not self.expert_placeholder4_inused:
            target_placeholder = self.expert_placeholder4
            self.expert_placeholder4_inused = True
        else:
            raise RuntimeError("No available expert placeholder")

        if target_placeholder == self.expert_placeholder:
            self.placeholder_to_expert['expert_placeholder'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder2:
            self.placeholder_to_expert['expert_placeholder2'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder3:
            self.placeholder_to_expert['expert_placeholder3'] = (layer_idx, expert_id)
        elif target_placeholder == self.expert_placeholder4:
            self.placeholder_to_expert['expert_placeholder4'] = (layer_idx, expert_id)         

        # torch.cuda.synchronize()
        tick = time.time()
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            dst = getattr(target_placeholder, name).weight.data
            src = getattr(self.model.layers[layer_idx].mlp.experts[expert_id], name).weight.data
            dst.copy_(src)
        copytime=time.time() - tick
        print(f"预取专家: 层 {layer_idx} 专家 {expert_id} -> {copytime*1000:.2f}ms")
        # 记录专家到占位专家的映射，以便后续使用
        self.expert_to_placeholder[(layer_idx, expert_id)] = target_placeholder        
    def release_placeholder(self, layer_idx, expert_id):
        for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                              'expert_placeholder3', 'expert_placeholder4']:
            stored_expert = self.placeholder_to_expert[placeholder_name]
            if stored_expert and (stored_expert[0] < layer_idx or 
                    (stored_expert[0] == self.n_layer - 1 and layer_idx <= 1)):
                # 释放占位专家
                setattr(self, f"{placeholder_name}_inused", False)
                self.placeholder_to_expert[placeholder_name] = None
                # self.placeholder_to_expert[placeholder_name] = None
                # self.expert_loc_now[stored_expert[0],  expert_id] = 0
        # if (layer_idx, expert_id) in self.expert_to_placeholder:
        #     placeholder = self.expert_to_placeholder[(layer_idx, expert_id)]
        #     if placeholder == self.expert_placeholder:
        #         self.expert_placeholder_inused = False
        #     elif placeholder == self.expert_placeholder2:
        #         self.expert_placeholder2_inused = False
        #     elif placeholder == self.expert_placeholder3:
        #         self.expert_placeholder3_inused = False
        #     elif placeholder == self.expert_placeholder4:
        #         self.expert_placeholder4_inused = False
        #     del self.expert_to_placeholder[(layer_idx, expert_id)]

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        expert_count = 0
        try:
            for i in range(self.n_layer):
                for j in range(self.n_expert):
                    if self.is_expert_in_gpu(i, j):
                        self.model.layers[i].mlp.experts[j].to(self.dev)
                        expert_count += 1
                        
            # 记录成功加载的专家数量
            with open('test.txt', 'a') as f:
                f.write(f"模型: moom, batch_size: {self.batch_size}, 成功加载专家数量: {expert_count}\n")
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # 记录显存溢出时的专家数量
                with open('test.txt', 'a') as f:
                    f.write(f"模型: moom, batch_size: {self.batch_size}, 显存溢出时专家数量: {expert_count}\n")
                raise  # 重新抛出异常
            else:
                raise  # 其他异常直接抛出

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1
    def is_expert_in_gpu_now(self, i_layer, i_expert):
        """检查专家当前是否实际在GPU上（包括占位专家）"""
        # 检查专家是否在占位专家中
        # for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
        #                       'expert_placeholder3', 'expert_placeholder4']:
        #     stored_expert = self.placeholder_to_expert[placeholder_name]
        #     if stored_expert and stored_expert == (i_layer, i_expert):
        #         return True
                
        # 直接检查专家参数是否在GPU上
        expert = self.model.layers[i_layer].mlp.experts[i_expert]
        return next(expert.parameters()).is_cuda
    def is_expert_loading(self, placeholder_name):
        """检查指定占位专家是否正在加载"""
        return self.expert_loading_status.get(placeholder_name, False)
    def is_expert_loaded(self, layer_id, expert_id):
        """检查指定层的专家是否已完成加载(不在prefetching_list中)"""
        # with self.placeholder_lock:  # 使用锁保证线程安全
            # 如果该层不在prefetching_list中，或者专家不在该层的加载列表中，则表示已加载完成
        return (layer_id not in self.prefetching_list or 
                expert_id not in self.prefetching_list[layer_id])  

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        fine_expert = self.model.layers[1].mlp.experts[0]
        n_param = sum(p.numel() for p in fine_expert.parameters())
        print(f"Number of parameters in a single expert: {n_param}")        # 计算一个粗粒度专家的参数量
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev) # TODO: magic number

        if self.batch_size==64:
            return int((free_mem) // (n_param * 2)-350)
        elif self.batch_size==32:
            return int((free_mem) // (n_param * 2)-300)
        else:
            return int((free_mem) // (n_param * 2)-830)


    def initial_beam_tensor(self, input_tensor):
        # transpose tensor of shape (beam_width, seq_len, beam_width) to (beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, text=None, output_token=20, input_token=None):
        torch.set_num_threads(16) # TODO: set appropriately
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        # print(f"[DEBUG] self类型: {type(self)}")  # 应该是 <class '__main__.FiddlerDeepSeek'>
        # print(f"[DEBUG] self属性: {dir(self)}")  # 应该包含 past_key_values 和 past_key_values_length        
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 1
        
        self.expert_selection_stats = []
        self.expert_time_stats = []
        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size
        # if input_token is not None:
        #     text = [t.split()[:input_token] for t in text]  # 按单词截断
        #     text = [' '.join(t) for t in text]  # 重新组合为字符串             
        # print("txet:", text)
        input_ids, position_ids, attention_mask = self.tokenize(text,input_token)
        # print("inputids:", input_ids)
        if input_token is not None:
            # 对每个样本独立截取前input_token个token
            input_ids = torch.stack([
                ids[:input_token] if len(ids) > input_token else ids 
                for ids in input_ids
            ])
            position_ids = torch.stack([
                pos[:input_token] if len(pos) > input_token else pos
                for pos in position_ids
            ])
            # attention_mask需要保持与input_ids相同的形状
            attention_mask = attention_mask[:, :, :, :input_token]
        # print("inputids2:", input_ids)
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

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,  # 跳过前1次迭代
                warmup=3,  # 预热1次迭代
                active=1,  # 记录3次迭代
                repeat=1  # 只执行1轮
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        # prof.start()

        for i_token in range(output_token):
            # prof.step()
            token_start_time = time.time()  # 记录单个token开始时间            
            # if self.beam_width == 1:
                # print(self.tokenizer.decode(input_ids[0]))
                # TODO: streaming output for beam search

            if self.is_decode:
                for i in range(input_ids.shape[0]):
                    # 将 Tensor 转换为列表后再解码
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :].tolist())                    
            # new_mask = torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=self.dev)
            # attention_mask = torch.cat([attention_mask, new_mask], dim=1)     
            #        
            if self.is_decode:
                # decode阶段处理
                if attention_mask.dim() == 4 and attention_mask.shape[-1] == attention_mask.shape[-2]:
                    # 从方阵mask转换为序列mask [1,1,seq_len,seq_len] -> [1,1,1,seq_len]
                    seq_len = attention_mask.shape[-1]
                    attention_mask = attention_mask[..., :1, :]  # 取第一行
                
                # 扩展序列长度维度
                seq_len = attention_mask.shape[-1]
                new_attention_mask = torch.ones(
                    (attention_mask.shape[0], 1, 1, seq_len + 1),
                    dtype=torch.bool,
                    device=self.dev
                )
                new_attention_mask[..., :seq_len] = attention_mask[..., :seq_len]
                attention_mask = new_attention_mask
            else:
                # prefill阶段保持原有方阵mask [1,1,seq_len,seq_len]
                pass
            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev
            ).unsqueeze(0).expand(input_ids.shape[0], -1)         
            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask )

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
            # print(f"Token {i_token} decode time: {token_time*1000:.2f}ms")
            
            # position_ids.shape: (1, 1)
            if not self.is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            self.is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        # print("\nToken decode time summary:")
        # for i, t in enumerate(self.token_decode_times):
        #     print(f"Token {i}: {t*1000:.2f}ms")
        # print(f"Total decode time: {decode_time*1000:.2f}ms")
        # print(f"Average per token: {decode_time*1000/len(self.token_decode_times):.2f}ms")
        # print("--------------------")
       
        print(f"Input: {text}")
        print(f"Output: {decode_strings[max_ids[0]]}")
        # prof.stop()
        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
            {
                'perf_stats': self.perf_stats,
                'expert_selection': self.expert_selection_stats,
                'expert_time': self.expert_time_stats,
                'layer_time': self.layer_time_stats,
                'outputs': decode_strings,  # 添加输出文本
                # 'layer_time_details': self.layer_time_details,  # 添加细粒度时间统计
                'expert_hot_stats': self.get_expert_stats(), 
                # 'layer_time_avg': {
                #     i: self.layer_time_accumulator[i] / max(1, len([x for x in self.layer_time_stats if x['layer_id'] == i]))
                #     for i in range(1, 27)
                # },

            }
        )
    def tokenize(self, text, input_token):
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")
        print(f"输出input_token: {input_token}...\n")    
        # 确保文本数量与batch_size匹配
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
        attention_mask = encodings.attention_mask.bool().to(self.dev)
        # print(f"输出id: {input_ids}...\n")            
        # 扩展为 (batch_size * beam_width, seq_len)
        # input_ids = input_ids.repeat_interleave(self.beam_width, dim=0)
        # attention_mask = attention_mask.repeat_interleave(self.beam_width, dim=0)
        
        # 生成 position_ids
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # 修正attention_mask形状为4D (batch, num_heads, seq_len, seq_len)
        # 注意：这里需要根据模型的实际头数(32)来扩展
        if attention_mask.dim() == 2:
            # 从(batch, seq_len)扩展到(batch, 1, 1, seq_len)
            # attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # # 然后扩展到(batch, num_heads, seq_len, seq_len)
            # attention_mask = attention_mask.expand(-1, 1, -1, -1)
            attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(-1)  # (batch, 1, seq_len, 1)
            attention_mask = attention_mask.expand(-1, -1, -1, seq_length)  # (batch, 1, seq_len, seq_len)        
        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask ):
        hidden_dim = self.model.config.hidden_size
        tick = time.time()
        # print(f"inpsid: {input_ids}ms")        
        inps = self.model.embed_tokens(input_ids)
        self.perf_stats['token_embedding'].append(time.time() - tick)

        if self.is_decode:
            total_decode_start = time.time()
            layer_times = {i: 0.0 for i in range(1, 27)}
            layer_times_fwd = {i: 0.0 for i in range(1, 27)}        
            layer_times_mid = {i: 0.0 for i in range(1, 27)}       
            layer_times_final = {i: 0.0 for i in range(1, 27)}                  
        # position_embeddings = self.model.embed_positions(position_ids)
        # position_embeddings = self.model.rotary_emb(inps, position_ids)
        #待注销
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # 确保attention_mask形状正确
        # if attention_mask.dim() == 2:
        #     attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
        layer_start_time = time.time()        
        layer_total_time = 0.0
        isprefetch=False

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()            
            if i_layer == 0:
                inps = layer.mlp(inps)  # 直接调用整个MLP
            else:

                self.release_placeholder(i_layer, 0)
                # print(f"is: {self.is_decode},layer: {i_layer}...pre: {self.prefetch_layers}\n")   
                laymid = time.time() - layer_tick             #          


                original_inps_shape = inps.shape
                self.cpu_expert_time_per_layer[i_layer] =0
                inps_residual = inps
                inps = layer.input_layernorm(inps)
                # position_embeddings = self.model.embed_positions(position_ids)
                
                inps = inps.view(batch_size, seq_len, hidden_dim)            
                tick = time.time()
                # print("attention_mask.shape", attention_mask.shape)
                attn_output = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,  # 关键修改：传入位置ID
                    past_key_value=self.past_key_value,
                    use_cache=True,
                )
                #      data collection

                torch.cuda.synchronize()  # 确保当前操作完成
                self.perf_stats['self_attention'].append(time.time() - tick)
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
                # inps = inps.view(-1, hidden_dim)
                # inps.shape: (batch_size*seq_len*embed_dim/hidden_dim, hidden_dim)
                layer_idx=i_layer
                if layer_idx not in self.layer_data:
                    self.layer_data[layer_idx] = {
                        "hidden_states": [],
                        "expert_indices": []
                    }
                inps = inps.view(batch_size, seq_len, hidden_dim)  # 恢复3D形状                
                # 获取专家处理前的hidden states
                pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)
                #data collection
                tick = time.time()
                selected_experts,routing_weights = layer.mlp.gate(inps)                         
                torch.cuda.synchronize()  # 确保当前操作完成
                self.perf_stats['moe_gating'].append(time.time() - tick)





                # intermediate variable to store the output of experts
                inps_after_experts = torch.zeros_like(inps, device=self.dev)
                experts = layer.mlp.experts
                #共享专家
                shared_output = torch.zeros_like(inps)


                expert_out = self.model.layers[i_layer].mlp.shared_experts(inps)
                shared_output += expert_out
                expert_token_counts = {}
                for expert_id in selected_experts.unique():
                    mask = (selected_experts == expert_id).any(dim=1)
                    expert_token_counts[expert_id.item()] = mask.sum().item()
                
                # 按token数量降序排序专家
                sorted_experts = sorted(
                    expert_token_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )            
                # 保存到当前层的专家统计中
                self.current_iter_expert_stats[i_layer] = {
                    'expert_ids': [e[0] for e in sorted_experts],  # 专家ID
                    'token_counts': [e[1] for e in sorted_experts]  # 对应token数量
                }
                layer_i_stats = self.current_iter_expert_stats[i_layer]
                for expert_id, token_count in zip(layer_i_stats['expert_ids'], layer_i_stats['token_counts']):
                    print(f"专家 {expert_id} 处理了 {token_count} 个token")
                filtered_expert_ids = []
                filtered_token_counts = []
                for expert_id, token_count in zip(layer_i_stats['expert_ids'], layer_i_stats['token_counts']):
                    # 检查专家是否已经在GPU上
                    expert_in_gpu = False
                    if self.is_expert_in_gpu_now(i_layer, expert_id):
                        expert_in_gpu = True
                    elif (i_layer in self.prefetch_list and expert_id in self.prefetch_list[i_layer]) or \
                        (i_layer in self.prefetching_list and expert_id in self.prefetching_list[i_layer]):
                        expert_in_gpu = True
                    else:
                        # 检查是否在占位专家中
                        for placeholder_name in ['expert_placeholder', 'expert_placeholder2', 
                                            'expert_placeholder3', 'expert_placeholder4']:
                            stored_expert = self.placeholder_to_expert[placeholder_name]
                            if stored_expert and stored_expert == (i_layer, expert_id):
                                expert_in_gpu = True
                                break
                    
                    if not expert_in_gpu:
                        filtered_expert_ids.append(expert_id)
                        filtered_token_counts.append(token_count)
                
                # 直接使用过滤后的结果，不需要再次排序
                sorted_experts = list(zip(filtered_expert_ids, filtered_token_counts))
                    
                # 初始化变量
                e = 1.39   # 搬运开销(m秒)
                tg = 0.95   # GPU计算开销(m秒)
                n = len(sorted_experts)
                ondemand_experts = []
                # if self.is_decode:
                # 计算CPU时间表(假设linshi.txt中的时间是毫秒)
                cpu_time_table = [float(line.strip())  for line in open('micromoon.txt')]
                tic=time.time()
                TA = sum(cpu_time_table[min(tokens, 1498)] for expert_id, tokens in sorted_experts[0:n])
                TC=TA   
                experts_in_placeholder = []            
                for i in range(n-1):
                    expert_id, token_count = sorted_experts[i]                
                    # 计算TG和TC
                    print("e,t,n,i",expert_id, token_count,n,i)
                    TG = (1 + i) * e + tg
                    print("tg",TG)
                    TC = TC-cpu_time_table[min(token_count, 1498)]                    
                    print("cpu_time_totl[tokens]",TC)                
                    # 判断是否需要ondemand
                    if self.is_decode:
                        if TG < TC:
                            if token_count>1:
                                ondemand_experts.append(expert_id)
                                print("执行了")
                        else:
                            print("跳出了")
                            break 
                    else:
                        if TG < TC+cpu_time_table[min(token_count, 1498)]:
                            ondemand_experts.append(expert_id)
                            print("执行了")
                            if i==n-2:
                                if TC-TG>e :
                                    expert_id2, token_count2 = sorted_experts[i+1]
                                    ondemand_experts.append(expert_id2)                 
                                elif TC-TG>e/2:
                                    self.prefil_pre=True
                        else:
                            # 不满足条件，停止检查后续专家
                                                        
                            print("跳出了")
                            break        
                print(f"time: {(time.time() - tic)*1000:.2f}ms")
                # 处理需要ondemand的专家
                for expert_id in ondemand_experts:
                    # 这里添加实际的ondemand处理逻辑
                    print(f"专家 {expert_id} 被标记为ondemand处理")                


                if i_layer < self.n_layer - 1:  # 如果不是最后一层
                    next_layer = self.model.layers[i_layer + 1]
                    # 使用当前层输出预测下一层专家
                    with torch.no_grad():
                        next_predicted_experts,next_routing_weights = next_layer.mlp.gate(inps)  # Moon模型使用mlp.gate

                    expert_token_counts = {}
                    for batch_idx in range(batch_size * seq_len):
                        for expert in next_predicted_experts[batch_idx]:
                            expert_token_counts[expert.item()] = expert_token_counts.get(expert.item(), 0) + 1        
                    # 按token处理量排序，获取前三热点专家
                    sorted_experts = sorted(expert_token_counts.items(), key=lambda x: x[1], reverse=True)

                    top3_experts = [expert[0] for expert in sorted_experts[:self.cache]]   

                if self.cpu_offload == 0:
                    print("oo")
                        # end of one expert

                else:
                    # prefill stage with offloading
                    
                    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_expert + self.n_shared_experts).permute(2, 1, 0)

                    # first, calculate the number of tokens for each expert

                    if self.is_decode:
                        laymid = time.time() - layer_tick
                        laypid = time.time()
                    # 定义收集处理结果的容器
                    # 定义收集处理结果的容器
                    cpu_experts = []
                    gpu_experts = []
                    experts_in_gpu = []

                    experts_loading = []
                    experts_remaining = []                
                                #for prefil
                    selected_expert_ids = selected_experts.unique().tolist()

                    if self.is_decode:
                        laymid = time.time() - layer_tick
                        laypid = time.time()
                    # 定义收集处理结果的容器
                    # 定义收集处理结果的容器
                    gpu_results = []  # 元素为 (mask_index, expert_output)
                    cpu_results = []
                    gpu_time = 0.0
                    cpu_time = 0.0
                    self._prefetch_thread_started=False
                    for i_expert in selected_expert_ids:
                        # 检查专家是否在GPU上
                        if self.is_expert_in_gpu_now(i_layer, i_expert):
                            gpu_experts.append(i_expert)
                            experts_in_gpu.append(i_expert)
                            continue
                            
                        # 检查占位专家是否存储了当前层的专家
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
                        if expert_in_placeholder or i_expert in ondemand_experts or i_expert in self.prefetch_list[i_layer] or i_expert in self.prefetching_list[i_layer]:
                        
                            if expert_in_placeholder or i_expert in self.prefetch_list[i_layer]:
                                experts_in_placeholder.append(i_expert)
                            elif i_expert in self.prefetching_list[i_layer]:
                                experts_loading.append(i_expert)
                            else:
                                experts_remaining.append(i_expert) 
                        else:
                            cpu_experts.append(i_expert)

                    gpu_time = 0.0
                    cpu_time = 0.0

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
                                # print("111")
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
                                print("222")
                                batch_mask = mask.view(batch_size, seq_len)                            
                                expert_input = inps[batch_mask].view(-1, hidden_dim)
                                tick = time.time()
                                expert_output = self.run_expert_at_gpu(12, 4, expert_input)
                                self.perf_stats['expert_compute'].append(time.time() - tick)
                                print(f"专家占位2(使用代理专家): {(time.time() - tick)*1000:.2f}ms")
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
                                print("333")                                
                                # 等待加载完成
                                # while self.is_expert_loaded(i_layer, expert_id):
                                #     time.sleep(0.001)
                                batch_mask = mask.view(batch_size, seq_len)                            
                                expert_input = inps[batch_mask].view(-1, hidden_dim)
                                tick = time.time()
                                expert_output = self.run_expert_at_gpu(1, 10, expert_input)
                                self.perf_stats['expert_compute'].append(time.time() - tick)
                                print(f"专家占位3(使用代理专家): {(time.time() - tick)*1000:.2f}ms")
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
                            
                            for i_expert in experts_remaining:
                                mask = (selected_experts == i_expert).any(dim=1)
                                if not mask.any():
                                    continue
                                print("444")    
                                batch_mask = mask.view(batch_size, seq_len)                            
                                expert_input = inps[batch_mask].view(-1, hidden_dim)
                                tick = time.time()
                                
                                # 使用双缓冲区机制
                                if not self.expert_placeholder5_inused:
                                    target_placeholder = self.expert_placeholder5
                                    self.expert_placeholder5_inused = True
                                elif not self.expert_placeholder6_inused:
                                    target_placeholder = self.expert_placeholder6
                                    self.expert_placeholder6_inused = True
                                else:
                                    while self.expert_placeholder5_inused and self.expert_placeholder6_inused:
                                        time.sleep(0.001)
                                    continue
                                    
                                self._async_ondemand(i_layer, i_expert, target_placeholder)
                                expert_output = target_placeholder(expert_input)
                                print("ondemand",i_expert)
                                flat_mask = mask.view(-1)
                                weights = routing_weights[flat_mask].gather(
                                    1, (selected_experts[flat_mask] == i_expert).long().argmax(dim=1, keepdim=True)
                                )
                                expert_output = expert_output * weights
                                mask_index = mask.nonzero().squeeze(1)
                                results.append((mask_index, expert_output))
                                
                                # 释放占位专家
                                if target_placeholder == self.expert_placeholder5:
                                    self.expert_placeholder5_inused = False
                                else:
                                    self.expert_placeholder6_inused = False
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
                        # 合并结果
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
                        if cpu_experts:
                            dummy_input = torch.randn(1, hidden_dim, device="cpu").to(self.dtype)  # 使用正确的hidden_dim和模型dtype
                            _ = self.model.layers[i_layer].mlp.experts[cpu_experts[0]](dummy_input)                           
                        for i_expert in cpu_experts:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            batch_mask = mask.view(batch_size, seq_len)                            
                            # 获取专家输入 [num_selected_tokens, hidden_dim]
                            expert_input = inps[batch_mask].view(-1, hidden_dim).to("cpu")                         
                            # expert_input = inps[mask].to("cpu")
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


                    # 启动并行的GPU和CPU处理线程
                    gpu_thread = threading.Thread(target=process_gpu_experts)
                    cpu_thread = threading.Thread(target=process_cpu_experts)
                    # if not self.is_decode and self.prefil_pre==True:
                    #     if self._prefetch_thread_started==False:
                    #         prefetch_thread.start()
                    #         self._prefetch_thread_started = True
                    #         self.prefil_pre=False

                    gpu_thread.start()
                    cpu_thread.start()
                    
                    # 只等待GPU和CPU线程完成
                    gpu_thread.join()
                    cpu_thread.join()
                    parallel_time = time.time() - parallel_start

                    # 计算并行度
                    max_thread_time = max(gpu_time, cpu_time)
                    parallel_degree = (gpu_time + cpu_time) / parallel_time if parallel_time > 0 else 1.0

                    # 打印时间统计
                    if self.is_decode:
                        stats_text = f"\nLayer {i_layer} Thread Time Stats:\n"
                        stats_text += f"GPU Thread Time: {gpu_time*1000:.2f}ms\n"
                        stats_text += f"CPU Thread Time: {cpu_time*1000:.2f}ms\n"
                        stats_text += f"Parallel Time: {parallel_time*1000:.2f}ms\n"
                        stats_text += f"Parallel Degree: {parallel_degree:.2f}x\n"
                        print(stats_text)
                        
                        # 写入到临时日志文件
                        os.makedirs('./log', exist_ok=True)
                        with open('./log/linshi.txt', 'a') as f:
                            f.write(stats_text)
                    # 合并GPU处理结果
                    # 合并CPU处理结果（需要移动到GPU）
                    for mask_index, expert_output in cpu_results:
                        expert_output = expert_output.view(-1, hidden_dim)
                        inps_after_experts = inps_after_experts.view(-1, hidden_dim)
                        inps_after_experts.index_add_(
                            0,
                            mask_index.to(self.dev),
                            expert_output.to(self.dev).to(inps_after_experts.dtype)
                        )
                        inps_after_experts = inps_after_experts.view(batch_size, seq_len, hidden_dim)

                # addition because there's residual connection over moe layer
                total_expert_output = shared_output + inps_after_experts

                # 残差连接（恢复原始形状）
                inps = inps_residual + total_expert_output.reshape(batch_size, seq_len, hidden_dim)           
                # inps = inps_residual + inps_after_experts.reshape(batch_size, seq_len, hidden_dim)  
                if self.is_decode:
                    layer_time = time.time() - layer_tick
                    layer_time_final = time.time() - laypid
                    self.layer_time_accumulator[i_layer] += layer_time
                    # self.layer_time_accumulator_details[expert_case][i_layer] += layer_time
                    
                    self.layer_time_stats.append({
                        'layer_id': i_layer,
                        'time': layer_time,
                        'token_step': self.past_key_values_length,
                        # 'expert_case': expert_case  # 添加专家位置情况
                    })
                    
                    # 按专家位置分类记录

                # end of one layer

                if self.is_decode:
                    layer_time = time.time() - layer_tick
                    layer_times[i_layer] += layer_time  # 累计层时间
                    layer_times_mid[i_layer] += laymid
                    layer_times_final[i_layer] += layer_time_final
                    self.layer_time_accumulator[i_layer] += layer_time
                    
                    # 记录到统计信息
        hot_experts = self.get_hot_expert()
        # for layer_id in hot_experts:
            # print(f"层 {layer_id} 热点专家: {hot_experts[layer_id][0]}")    
        if self.is_decode:
            # 打印层时间统计
            print("\n各层处理时间统计(ms):")
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
            avg_cpu_ratio = (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0
            # print(f"平均每层时间: {avg_layer_time*1000:.2f}ms (CPU专家平均占比: {avg_cpu_ratio:.1f}%)")
            os.makedirs('./log', exist_ok=True)
            with open('./log/expert_stats.txt', 'a') as f:
                f.write("\n各层处理时间统计(ms):\n")
                for i in range(1, 27):
                    cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                    layer_time = layer_times[i] * 1000
                    layer_time_mids = layer_times_mid[i] * 1000
                    layer_time_finals = layer_times_final[i] * 1000
                    # cpu_ratio = (cpu_time / layer_time * 100) if layer_time > 0 else 0
                    f.write(f"层 {i}: {layer_time:.2f}ms (, 中间层: {layer_time_mids:.2f}ms, 最终层: {layer_time_finals:.2f}ms, %)\n")
                    # f.write(f"层 {i}: {layer_time:.2f}ms (CPU专家: {cpu_time:.2f}ms, 占比: {cpu_ratio:.1f}%)\n")
                
                # f.write(f"平均每层时间: {avg_layer_time*1000:.2f}ms (CPU专家平均占比: {avg_cpu_ratio:.1f}%)\n")
        other_ops_start = time.time()
        print("ok")
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start
            print(f"\nToken处理时间统计:")
            print(f"总时长: {total_decode_time*1000:.2f}ms")
            print(f"  - {self.n_layer}层处理: {layer_total_time*1000:.2f}ms (平均每层: {layer_total_time*1000/self.n_layer:.2f}ms)")
            print(f"  - 其他操作(norm+lm_head): {other_ops_time*1000:.2f}ms")
        self.present_key_value = present_key_value
        return lm_logis

    # def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        # return self.model.layers[i_layer].mlp.experts[i_expert](
        #     inps, routing_weights
        # )
    def run_expert_at_gpu(self, i_layer, i_expert, inps ):
        """Run the expert at GPU"""
        start_time = time.time()
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        torch.cuda.synchronize()  # 确保当前操作完成
        elapsed = time.time() - start_time
        
        token_count = inps.shape[0]
        expert_status = 'normal'
        if token_count > 2:
            expert_status = 'veryhot'
        elif token_count > 1:
            expert_status = 'hot'
        # print(f"GPU专家处理 - 层 {i_layer} 专家 {i_expert}: "
        #       f"处理 {token_count} tokens, 耗时 {elapsed*1000:.2f}ms, 状态 {expert_status}")
        self.current_iter_expert_stats[i_layer]['expert_ids'].append(i_expert)
        self.current_iter_expert_stats[i_layer]['token_counts'].append(token_count)      

        # 记录专家处理时间
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
        """Run the expert at CPU"""
        start_time = time.time()
        result = self.model.layers[i_layer].mlp.experts[i_expert](inps)
        torch.cuda.synchronize()  # 确保当前操作完成
        elapsed = time.time() - start_time
        if self.is_decode:
            self.cpu_expert_time_per_layer[i_layer] += elapsed 
        token_count = inps.shape[0]
        expert_status = 'normal'
        if token_count > 2:
            expert_status = 'veryhot'
        elif token_count > 1:
            expert_status = 'hot'
        # print(f"CPU专家处理 - 层 {i_layer} 专家 {i_expert}: "
        #       f"处理 {token_count} tokens, 耗时 {elapsed*1000:.2f}ms, 状态 {expert_status}")
        self.current_iter_expert_stats[i_layer]['expert_ids'].append(i_expert)
        self.current_iter_expert_stats[i_layer]['token_counts'].append(token_count)

        # 记录专家处理时间
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
        """获取专家热度统计信息"""
        stats = {
            'hot_experts': {i: {'count': 0, 'hot': 0, 'veryhot': 0} for i in range(1, 27)},
            'hot_counts': {2: 0, 3: 0, 4: 0, 5: 0},
            'token_distribution': {}  # 新增: token数量分布统计
        }
        
        # 按层统计hot/veryhot专家
        for record in self.expert_time_stats:
            layer = record['layer_id']
            token_count = record['token_count']
            expert_status = record['status']
            
            # 原有hot/veryhot统计
            stats['hot_experts'][layer]['count'] += 1
            if expert_status == 'hot':
                stats['hot_experts'][layer]['hot'] += 1
            elif expert_status == 'veryhot':
                stats['hot_experts'][layer]['veryhot'] += 1
        
        # 统计hot专家数量分布
        layer_hot_counts = {i: 0 for i in range(1, 27)}
        for record in self.expert_time_stats:
            if record['status'] in ['hot', 'veryhot']:
                layer_hot_counts[record['layer_id']] += 1
                
        for count in layer_hot_counts.values():
            if count >= 2 and count <= 5:
                stats['hot_counts'][count] += 1
                
        # 重置token分布统计，只统计当前迭代
        stats['token_distribution'] = {}
        token_counts = [r['token_count'] for r in self.expert_time_stats]
        unique_counts = set(token_counts)
        
        for count in unique_counts:
            stats['token_distribution'][count] = token_counts.count(count)
                
        return stats
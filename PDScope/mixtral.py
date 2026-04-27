import copy
import threading
import time
import os  
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
import threading

class FiddlerMixtral:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.hot_experts = {}
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            
            use_cache=True,
        )
        self.cache =args.cache
        self.batch_size = args.batch_size  
        if self.batch_size==4:
           self.cache=5    
        else:
           self.cache=7    
        self.lm_head = self.model.lm_head
        self.model = self.model.model
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
        self.prefil_pre=False
        self.expert_placeholder5 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[5]
        ).to(self.dev)
        self.expert_placeholder6 = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[6]
        ).to(self.dev)
        self.expert_placeholder5_inused = False
        self.expert_placeholder6_inused = False

        self.expert_to_placeholder = {}  
        self.expert_placeholder_inused=False 
        self.expert_placeholder2_inused=False 
        self.expert_placeholder3_inused=False   
        self.expert_placeholder4_inused=False  
        self.placeholder_lock = threading.Lock()       
        self.prefetch_layers=0   
        self.is_decode = False
        self.placeholder_to_expert = {
            'expert_placeholder': None,
            'expert_placeholder2': None,
            'expert_placeholder3': None,
            'expert_placeholder4': None
        }         
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
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)

        self.expert_selection_stats = []  
        self.expert_time_stats = []       
        self.prefetch_list = {}  
        self.prefetching_list = {}  
        self.expert_selection_history = {}  
        self.hit_stats = {}  
        for i in range(self.n_layer):
            self.expert_selection_history[i] = []
            self.hit_stats[i] = {'hits': 0, 'total': 0}
            
        self.expert_weight_accumulator = {}  
        for i in range(self.n_layer):
            self.expert_weight_accumulator[i] = torch.zeros(8, device=self.dev)  

        self.cpu_expert_time_per_layer = {i: 0.0 for i in range(self.n_layer)}
        
        self.latency_cpu = 5
        self.latency_gpu = 45

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.expert_loc_now = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)
        
        self.layer_time_stats = []  
        self.layer_time_accumulator = {}  
        for i in range(self.n_layer):
            self.layer_time_accumulator[i] = 0.0
        self.layer_time_details = {
            'all_gpu': [],    
            'all_cpu': [],    
            'mixed': []       
        }
        
        self.layer_time_accumulator_details = {
            'all_gpu': {i: 0.0 for i in range(self.n_layer)},
            'all_cpu': {i: 0.0 for i in range(self.n_layer)},
            'mixed': {i: 0.0 for i in range(self.n_layer)}
        }
        self.last_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []} 
            for i in range(self.n_layer)
        }
        self.current_iter_expert_stats = {
            i: {'expert_ids': [], 'token_counts': []}
            for i in range(self.n_layer)
        }

        self.layer_data = {}  
        
        self.expert_selection_count = np.zeros((self.n_layer, self.n_expert), dtype=int)
        tick = time.time()        
        self.bring_expert_to_gpu()
        
        print("Model is ready.")



    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            

    def get_hot_expert(self):
        
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
        """Set the location of experts"""
        if popular_experts is None:
            
            
            
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
        
        return self.expert_loading_status.get(placeholder_name, False)
    def is_expert_loaded(self, layer_id, expert_id):
        
        
            
        return (layer_id not in self.prefetching_list or 
                expert_id not in self.prefetching_list[layer_id])     
    def release_placeholder(self, layer_idx, expert_id):
        for placeholder_name in ['expert_placeholder', 'expert_placeholder2',
                              'expert_placeholder3', 'expert_placeholder4']:
            stored_expert = self.placeholder_to_expert[placeholder_name]
            if stored_expert and (stored_expert[0] < layer_idx or 
                    (stored_expert[0] == self.n_layer - 1 and layer_idx <= 1)):
                
                setattr(self, f"{placeholder_name}_inused", False)
                self.placeholder_to_expert[placeholder_name] = None

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        expert_count = 0
        try:
            for i in range(self.n_layer):
                for j in range(self.n_expert):
                    if self.is_expert_in_gpu(i, j):
                        self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)
                        expert_count += 1
                        
            
            
                
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                
                
                    
                
            
                raise  
    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1
    def is_expert_in_gpu_now(self, i_layer, i_expert):
        
        
        
        
        
        
        
                
        
        expert = self.model.layers[i_layer].block_sparse_moe.experts[i_expert]
        return next(expert.parameters()).is_cuda


    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        
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
        
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor
    def _process_single_cpu_expert(self, i_layer, i_expert, combined_inps, combined_weights, combined_selected, expert_mask, combined_mask):
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
        attention_mask = encodings.attention_mask.bool().to(self.dev)
        
        
        
        
        
        
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        
        
        if attention_mask.dim() == 2:
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
            attention_mask = attention_mask.expand(-1, 32, -1, -1)
        
        return input_ids, position_ids, attention_mask

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask ):
        hidden_dim = self.model.config.hidden_size
        tick = time.time()
        
        inps = self.model.embed_tokens(input_ids)
        self.perf_stats['token_embedding'].append(time.time() - tick)
        
        if self.is_decode:
            total_decode_start = time.time()
            layer_times = {i: 0.0 for i in range(self.n_layer)}
            layer_times_fwd = {i: 0.0 for i in range(self.n_layer)}        
            layer_times_mid = {i: 0.0 for i in range(self.n_layer)}       
            layer_times_final = {i: 0.0 for i in range(self.n_layer)}                  
        
        position_embeddings = self.model.rotary_emb(inps, position_ids)
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        
        
        layer_start_time = time.time()        
        layer_total_time = 0.0
        isprefetch=False

        for i_layer, layer in enumerate(self.model.layers):
            layer_tick = time.time()            

            self.release_placeholder(i_layer, 0)
            
            laymid = time.time() - layer_tick             
            
            
            
            
            
            
            
            
            
            
            


            original_inps_shape = inps.shape
            self.cpu_expert_time_per_layer[i_layer] =0
            inps_residual = inps
            inps = layer.input_layernorm(inps)

            inps = inps.view(batch_size, seq_len, hidden_dim)            
            tick = time.time()
            attn_output = layer.self_attn(
                hidden_states=inps,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,  
                past_key_value=self.past_key_value,
                use_cache=True,
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
            inps = inps.view(-1, hidden_dim)
            
            layer_idx=i_layer
            if layer_idx not in self.layer_data:
                self.layer_data[layer_idx] = {
                    "hidden_states": [],
                    "expert_indices": []
                }
            
            
            pre_expert_hidden_states = inps.view(batch_size, seq_len, -1)
            
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
            
                
            filtered_expert_ids = []
            filtered_token_counts = []
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
                
            
            e = 27.0   
            tg = 4.22   
            n = len(sorted_experts)
            ondemand_experts = []
            
            
            cpu_time_table = [float(line.strip())  for line in open('micromixtral.txt')]
            tic=time.time()
            TA = sum(cpu_time_table[min(tokens, 1498)] for expert_id, tokens in sorted_experts[0:n])
            TC=TA   
            experts_in_placeholder = []            
            for i in range(n-1):
                expert_id, token_count = sorted_experts[i]                
                
                print("e,t,n,i",expert_id, token_count,n,i)
                TG = (1 + i) * e + tg
                print("tg",TG)
                TC = TC-cpu_time_table[min(token_count, 1498)]                    
                print("cpu_time_totl[tokens]",TC)                
                
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
            
            
                
                

            if i_layer < self.n_layer - 1:  
                next_next_layer = self.model.layers[i_layer + 1]
                
                with torch.no_grad():
                    next_next_router_logits = next_next_layer.block_sparse_moe.gate(inps)
                    next_next_routing_weights = F.softmax(next_next_router_logits, dim=1)
                    _, next_next_predicted_experts = torch.topk(next_next_routing_weights, 2, dim=-1)
                
                
                expert_token_counts = {}
                for batch_idx in range(batch_size * seq_len):
                    for expert in next_next_predicted_experts[batch_idx]:
                        expert_token_counts[expert.item()] = expert_token_counts.get(expert.item(), 0) + 1        
                
                sorted_experts = sorted(expert_token_counts.items(), key=lambda x: x[1], reverse=True)
                top3_experts = [expert[0] for expert in sorted_experts[:self.cache]]   
                
                
                
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
                
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                cpu_experts = []
                gpu_experts = []
                experts_in_gpu = []
                
                experts_loading = []
                experts_remaining = []                
                            
                selected_expert_ids = selected_experts.unique().tolist()
                                                
                
                
                
                
                
                
                
                

                
                
                

                
                self._prefetch_thread_started=False
                for i_expert in selected_expert_ids:
                    
                    if self.is_expert_in_gpu_now(i_layer, i_expert):
                        gpu_experts.append(i_expert)
                        experts_in_gpu.append(i_expert)
                        continue
                        
                    
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


                if self.is_decode:
                    laymid = time.time() - layer_tick
                    laypid = time.time()
                
                
                gpu_results = []  
                cpu_results = []
                gpu_time = 0.0
                cpu_time = 0.0

                def process_gpu_experts():
                    nonlocal gpu_time
                    start_time=time.time()
                    def process_experts_in_gpu():
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
                        results = []
                        
                        for i_expert in experts_remaining:
                            mask = (selected_experts == i_expert).any(dim=1)
                            if not mask.any():
                                continue
                            print("444")    
                            expert_input = inps[mask]
                            tick = time.time()
                            
                            
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
                        inps_after_experts.index_add_(
                            0,
                            mask_index,
                            expert_output.to(inps_after_experts.dtype)
                        )
                    
                    gpu_time = time.time() - start_time


                def process_cpu_experts():
                    nonlocal cpu_time
                    start_time = time.time()
                    for i_expert in cpu_experts:
                        mask = (selected_experts == i_expert).any(dim=1)
                        if not mask.any():
                            continue
                            
                        expert_input = inps[mask].to("cpu")
                        tick = time.time()
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
                    
                    
                    hot_experts = self.hot_experts
                    
                    
                    if self.batch_size == 4 or self.batch_size  == 8 or self.batch_size  == 16:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1
                    elif self.batch_size  == 32:
                        self.prefetch_layers = i_layer + 1
                        expert_count = 1
                    else:  
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
                            

                            self.prefetching_list[self.prefetch_layers].append(expert_id)
                            time.sleep(0.027)  
                            self.prefetch_list[self.prefetch_layers].append(expert_id)
                            self.prefetching_list[self.prefetch_layers]=[]
                            
                            experts_loaded += 1
                            if experts_loaded >= expert_count:
                                break


                
                
                parallel_start = time.time()
                prefetch_thread = threading.Thread(target=prefetch_experts)
                
                
                
                gpu_thread = threading.Thread(target=process_gpu_experts)
                cpu_thread = threading.Thread(target=process_cpu_experts)
                
                
                
                
                
                if self.is_decode and self.batch_size>8:
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
                    stats_text = f"\nLayer {i_layer} Thread Time Stats:\n"
                    stats_text += f"GPU Thread Time: {gpu_time*1000:.2f}ms\n"
                    stats_text += f"CPU Thread Time: {cpu_time*1000:.2f}ms\n"
                    stats_text += f"Parallel Time: {parallel_time*1000:.2f}ms\n"
                    stats_text += f"Parallel Degree: {parallel_degree:.2f}x\n"
                    print(stats_text)
                    
                    
                    os.makedirs('./log', exist_ok=True)
                    with open('./log/linshi.txt', 'a') as f:
                        f.write(stats_text)
                

                
                for mask_index, expert_output in cpu_results:
                    inps_after_experts.index_add_(
                        0,
                        mask_index.to(self.dev),
                        expert_output.to(self.dev).to(inps_after_experts.dtype)
                    )
                
                   
            
            inps = inps_residual + inps_after_experts.reshape(batch_size, seq_len, hidden_dim)  
            if self.is_decode:
                layer_time = time.time() - layer_tick
                layer_time_final = time.time() - laypid
                self.layer_time_accumulator[i_layer] += layer_time
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            

            if self.is_decode:
                layer_time = time.time() - layer_tick
                layer_times[i_layer] += layer_time  
                layer_times_mid[i_layer] += laymid
                layer_times_final[i_layer] += layer_time_final
                self.layer_time_accumulator[i_layer] += layer_time
                
                
                self.layer_time_stats.append({
                    'layer_id': i_layer,
                    'time': layer_time,
                    'token_step': self.past_key_values_length
                })
        hot_experts = self.get_hot_expert()
        
            
        if self.is_decode:
            
            
            for i in range(self.n_layer):
                
                layer_time = layer_times[i] * 1000
                layer_times_mids = layer_times_mid[i] * 1000
                layer_times_finals = layer_times_final[i] * 1000
                
                
                

            
            
            avg_layer_time = sum(layer_times.values()) / self.n_layer
            total_cpu_time = sum(self.cpu_expert_time_per_layer.values())
            avg_cpu_ratio = (total_cpu_time / avg_layer_time * 100) if avg_layer_time > 0 else 0
            
            os.makedirs('./log', exist_ok=True)
            with open('./log/expert_stats.txt', 'a') as f:
                
                for i in range(self.n_layer):
                    cpu_time = self.cpu_expert_time_per_layer[i] * 1000
                    layer_time = layer_times[i] * 1000
                    layer_time_mids = layer_times_mid[i] * 1000
                    layer_time_finals = layer_times_final[i] * 1000
                    
                    
                    
                
                
        other_ops_start = time.time()

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        if self.is_decode:
            other_ops_time = time.time() - other_ops_start
            total_decode_time = time.time() - total_decode_start
            
            
            
            
        self.present_key_value = present_key_value
        return lm_logis

    
        """Run the expert at CPU"""
        
        
        
    def run_expert_at_gpu(self, i_layer, i_expert, inps ):
        """Run the expert at GPU"""
        start_time = time.time()
        result = self.model.layers[i_layer].block_sparse_moe.experts[i_expert](inps)
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
        """Run the expert at CPU"""
        start_time = time.time()
        result = self.model.layers[i_layer].block_sparse_moe.experts[i_expert](inps)
        
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
        
        stats = {
            'hot_experts': {i: {'count': 0, 'hot': 0, 'veryhot': 0} for i in range(self.n_layer)},
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
        
        
        layer_hot_counts = {i: 0 for i in range(self.n_layer)}
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
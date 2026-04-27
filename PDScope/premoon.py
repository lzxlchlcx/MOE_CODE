import torch
import torch.nn as nn
from transformers import AutoTokenizer, MixtralForCausalLM
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import json
import os
import random
import sys

sys.path.append("./")
from moon import FiddlerMoon

class Config:

    model_name = "/home/share/bz/model/Moonlight-16B-A3B-Instruct"  
    
    
    
    
    
    
    dataset_path = "/home/share/bz/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"  
    dataset_type = 'share'  
    output_dir = "expert_predictors"  
    topk=6
    projection_dim = 256  
    hidden_dim = 128  
    mlp_layers = 3  
    num_experts = 64  
    batch_size = 32  
    lr = 0.001  
    epochs = 20  
    max_sequence_length = 163840  
    max_samples = 1000  
    save_every = 100  
    num_layers = 27  
def parse_args():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=20)
    
    
    parser.add_argument("--model", type=str, default="/home/share/bz/Moonlight-16B-A3B-Instruct")
    parser.add_argument("--cpu-offload", type=int, default=1, choices=[0, 1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[512])
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[32])
    parser.add_argument("--n-sample", type=int, default=5)
    
    return parser.parse_args()
def load_dataset(path, dataset_type='sharegpt'):

    if dataset_type == 'dpo':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                texts.append(data['chosen'][0]['content'])
        return texts
    elif dataset_type == 'llava':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                
                for conv in item.get('conversations', []):
                    if conv['from'] == 'human':  
                        texts.append(conv['value'])
        return texts        
    else:  
        with open(path, "r") as f:
            data = json.load(f)
        texts = []
        for d in data:
            if len(d.get("conversations", [])) > 0:
                texts.append(" ".join(d["conversations"][0]["value"].split()))
        return texts

    


class MoEPredictor(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  
        
        
        
        dynamic_hidden_dim = 256 if 8 <= layer_idx <= 20 else 128  
        input_dim = config.projection_dim + config.num_experts
        output_dim = config.num_experts
        
        
        dynamic_mlp_layers = config.mlp_layers + (2 if 8 <= layer_idx <= 20 else 0)  
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(0.4 if layer_idx > 20 else 0.3)  
        )        

        
        
        
        
        
        
        layers = []
        dim_in = config.projection_dim
        
        for i in range(dynamic_mlp_layers):
            
            if 8 <= layer_idx <= 20 and i > 0:
                layers.append(ResidualBlock(dim_in))  
            else:
                layers.append(nn.Linear(dim_in, dynamic_hidden_dim))
                layers.append(nn.GELU())  
                layers.append(nn.Dropout(0.4 if layer_idx > 20 else 0.3))
                dim_in = dynamic_hidden_dim
        
        
        layers.append(nn.Linear(dim_in, config.num_experts))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input_proj(x)
        return self.mlp(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  

class MoEDataset(Dataset):

    
    def __init__(self, layer_idx, config):
        self.config = config
        self.layer_idx = layer_idx
        current_data = np.load(
            os.path.join(config.output_dir, f"layer_{layer_idx}.npz"), 
            allow_pickle=True
        )
        prev_data = np.load(
            os.path.join(config.output_dir, f"layer_{layer_idx-1}.npz"), 
            allow_pickle=True
        ) if layer_idx > 1 else None
        

        
        proj_path = os.path.join(config.output_dir, f"layer_{layer_idx}_projector.pt")
        if os.path.exists(proj_path):
            self.projection = torch.load(proj_path)
        else:
            from sklearn.decomposition import PCA
            self.projection = PCA(n_components=config.projection_dim)
            self.projection.fit(current_data["hidden_states"])
            torch.save(self.projection, proj_path)
      
        
        
        self.current_hidden = self.projection.transform(current_data["hidden_states"])
        self.current_experts = current_data["expert_indices"]
        if prev_data is not None:
            self.prev_hidden = self.projection.transform(prev_data["hidden_states"])
            self.prev_experts = prev_data["expert_indices"]
        else:
            
            self.prev_hidden = np.zeros_like(self.current_hidden)
            self.prev_experts = np.zeros_like(self.current_experts)
            
    
    def __len__(self):
        return len(self.current_hidden)
    
    def __getitem__(self, idx):
        return (
            torch.cat([
                torch.tensor(self.prev_hidden[idx], dtype=torch.float32),
                torch.tensor(self.prev_experts[idx], dtype=torch.float32)
            ]),  
            torch.tensor(self.current_experts[idx], dtype=torch.float32)  
        )

def train_predictors(config):

    os.makedirs(config.output_dir, exist_ok=True)
    layer_best_val_losses = []
    log_file = os.path.join(config.output_dir, "training_log.txt")    
    
    for layer_idx in range(config.num_layers):
        layer_file = os.path.join(config.output_dir, f"layer_{layer_idx}.npz")
        
        
        if not os.path.exists(layer_file):
            print(f"Skipping layer {layer_idx} - no data")
            continue
            
        print(f"\nTraining predictor for layer {layer_idx}")
        
        
        dataset = MoEDataset(layer_idx, config)
        
        
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size
        )
        
        all_indices = []
        for _, labels in train_dataset:
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            all_indices.append(labels)
        if layer_idx > 0:
            prev_layer_file = os.path.join(config.output_dir, f"layer_{layer_idx-1}.npz")
            if not os.path.exists(prev_layer_file):
                print(f"Skipping layer {layer_idx} - previous layer data not found")
                continue        
        expert_counts = np.zeros(config.num_experts, dtype=int)
        
        if len(all_indices) > 0:
            
            if all(isinstance(idx, (int, float)) for idx in all_indices):
                indices_array = np.array(all_indices, dtype=int)
            else:
                indices_array = np.concatenate(
                    [np.array(idx).flatten() for idx in all_indices if idx.size > 0]
                ).astype(int)
            
            if indices_array.size > 0:
                counts = np.bincount(indices_array, minlength=config.num_experts)
                expert_counts = counts[:config.num_experts]  
        
        
        epsilon = 1e-5
        total = np.sum(config.num_experts / (expert_counts + epsilon))
        expert_weights = torch.tensor(
            config.num_experts / (expert_counts + epsilon) / total,
            dtype=torch.float32
        )


        
        model = MoEPredictor(
            layer_idx=layer_idx,
            config=config
        ).float()
        

        lr = 0.002 if 8 <= layer_idx <= 20 else config.lr
        weight_decay = 1e-3 if layer_idx > 20 else 1e-4
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        
        
        
        class HybridLoss(nn.Module):
            def __init__(self, alpha=expert_weights, gamma=2.0):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss(weight=alpha)
                self.gamma = gamma
                
            def forward(self, outputs, targets):
                bce_loss = self.bce(outputs, targets)
                
                
                probas = torch.sigmoid(outputs)
                focal_loss = -targets * (1 - probas).pow(self.gamma) * torch.log(probas + 1e-8)
                focal_loss = focal_loss.mean()
                
                return bce_loss + 0.3 * focal_loss  
        
        criterion = HybridLoss()        
        
        best_val_loss = float("inf")
        train_losses, val_losses = [], []
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )

        patience = 5
        best_epoch = 0        
        for epoch in range(config.epochs):
            
            model.train()
            epoch_train_loss = 0
            for inputs, labels in train_loader:
                if 8 <= layer_idx <= 20:  
                    inputs += torch.normal(0, 0.03, inputs.shape)  
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
                optimizer.step()
                epoch_train_loss += loss.item() * inputs.size(0)
            scheduler.step()  
            epoch_train_loss /= len(train_dataset)
            train_losses.append(epoch_train_loss)
            
            
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                total_hits = 0
                total_samples = 0
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item() * inputs.size(0)
                    
                    
                    k = config.topk  
                    _, pred_topk = torch.topk(outputs, k=k, dim=1)  
                    _, true_topk = torch.topk(labels, k=k, dim=1)   
                    
                    
                    
                    batch_matches = 0
                    for pred, true in zip(pred_topk, true_topk):
                        pred_set = set(pred.tolist())
                        true_set = set(true.tolist())
                        batch_matches += len(pred_set & true_set)  
                    
                    
                    total_hits += batch_matches
                    
                    total_samples += len(true_topk)*k
                    

                
                hit_rate = total_hits / total_samples if total_samples > 0 else 0



            epoch_val_loss /= len(val_dataset)
            
            val_losses.append(epoch_val_loss)
            log_msg = f"layer {layer_idx},Epoch {epoch+1}/{config.epochs}: " \
                    f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, " \
                    f"Hit Rate: {hit_rate:.2%}" 
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
            
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), 
                          os.path.join(config.output_dir, f"layer_{layer_idx}_mlp.pth"))
            elif epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        layer_best_val_losses.append((layer_idx, best_val_loss))
        
    plt.figure(figsize=(12, 6))
    layers, losses = zip(*sorted(layer_best_val_losses, key=lambda x: x[0]))
    plt.bar(range(len(layers)), losses)
    plt.xticks(range(len(layers)), layers)
    plt.xlabel("Layer Index")
    plt.ylabel("Best Validation Loss")
    plt.title("Best Validation Loss Across Layers")
    for i, v in enumerate(losses):
        plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    
    plt.savefig(os.path.join(config.output_dir, "all_layers_val_loss.png"))
    plt.close()


class ExpertPredictor:

    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = FiddlerMoon(config)
        
        
        self.predictors = {}
        self.projections = {}
        
        for layer_idx in range(config.num_layers):
            proj_path = os.path.join(config.output_dir, f"layer_{layer_idx}_projector.pt")
            mlp_path = os.path.join(config.output_dir, f"layer_{layer_idx}_mlp.pth")
            
            if os.path.exists(proj_path) and os.path.exists(mlp_path):
                
                self.projections[layer_idx] = torch.load(proj_path)
                
                
                mlp = MoEPredictor(
                    input_dim=config.projection_dim,
                    output_dim=config.num_experts,
                    config=config
                ).to(self.device).float()
                mlp.load_state_dict(torch.load(mlp_path))
                mlp.eval()
                self.predictors[layer_idx] = mlp
    
    def predict_for_layer(self, text, layer_idx):

        
        
        
        
        
        
        
        
        
        self.model.generate(
                    text, output_token=32, input_token=512
                )
        hidden_states=self.model.layer_data[layer_idx]["hidden_states"]       
        
        
        
        
        
        
        
        
        
        
        
        
        if not hasattr(self, 'hidden_states'):
            return []
        
        
        results = []
        batch_size, seq_len, _ = self.hidden_states.shape
        
        for b in range(batch_size):
            for s in range(seq_len):
                
                hidden_state = self.hidden_states[b, s]
                projected = self.projections[layer_idx].transform([hidden_state])[0]
                
                
                input_tensor = torch.tensor(projected, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    prediction = self.predictors[layer_idx](input_tensor).sigmoid()
                    prediction = prediction.cpu().numpy()
                
                
                top2_indices = np.argsort(prediction[0])[-2:][::-1]
                
                # results.append({
                #     "token": self.tokenizer.decode(inputs.input_ids[b, s]),
                #     "layer": layer_idx,
                #     "position": s,
                #     "predicted_experts": top2_indices.tolist(),
                #     "expert_probabilities": prediction[0].tolist()
                # })
        
        return results

if __name__ == "__main__":
    config = Config()
    
    
    args = argparse.Namespace(
        model=config.model_name,
        cpu_offload=1,
        batch_size=config.batch_size,
        beam_width=1
    )
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    train_predictors(config)
    
    
    
    
    
    
    
    
    
    
    
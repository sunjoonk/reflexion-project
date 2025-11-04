# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
import pprint

from model import Transformer

def load_config(config_path, cli_args):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if cli_args.learning_rate:
        config['train']['learning_rate'] = cli_args.learning_rate
    
    if cli_args.batch_size:
        config['train']['batch_size'] = cli_args.batch_size

    return config

def get_dummy_batch(batch_size, src_seq_len, tgt_seq_len, vocab_size, pad_idx, device):
    src = torch.randint(1, vocab_size, (batch_size, src_seq_len)).to(device)
    src[:, -5:] = pad_idx 
    
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_seq_len)).to(device)
    tgt[:, -6:] = pad_idx
    
    return src, tgt

def main():
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the config YAML file (e.g., configs/base.yaml)")
    parser.add_argument('--learning_rate', type=float, default=None, 
                        help="Override learning rate in config")
    parser.add_argument('--batch_size', type=int, default=None, 
                        help="Override batch size in config")

    args = parser.parse_args()
    config = load_config(args.config, args)

    print("--- Loaded Configuration ---")
    pprint.pprint(config)
    print("------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config['model']
    data_cfg = config['data']
    train_cfg = config['train']

    model = Transformer(
        src_vocab_size=data_cfg['src_vocab_size'],
        tgt_vocab_size=data_cfg['tgt_vocab_size'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        d_ff=model_cfg['d_ff'],
        num_layers=model_cfg['num_layers'],
        dropout=model_cfg['dropout'],
        max_len=model_cfg['max_len'],
        pad_token_idx=data_cfg['pad_idx']
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=data_cfg['pad_idx'])
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_cfg['learning_rate'], 
        betas=train_cfg['optimizer']['betas'], 
        eps=train_cfg['optimizer']['eps']
    )

    print("--- Start Training ---")
    model.train()

    for epoch in range(train_cfg['epochs']):
        
        src_data, tgt_data = get_dummy_batch(
            batch_size=train_cfg['batch_size'],
            src_seq_len=20,
            tgt_seq_len=25,
            vocab_size=data_cfg['src_vocab_size'],
            pad_idx=data_cfg['pad_idx'],
            device=device
        )
        
        tgt_input = tgt_data[:, :-1]
        tgt_output = tgt_data[:, 1:]
        
        optimizer.zero_grad()
        logits = model(src_data, tgt_input)
        
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            tgt_output.reshape(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch+1:2d}/{train_cfg['epochs']:2d}, Loss: {loss.item():.4f}")

    print("--- Training Finished ---")

if __name__ == "__main__":
    main()
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=int, default=None, help='Resume from epoch')
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
import datetime
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.data.nuswide_loader import NUSWIDEDataset
from src.models.losses import calc_dpsh_loss_standard
from src.utils.metrics import calculate_mAP

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'NUSWIDE')
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-4
BIT = 16
GAMMA = 0.1
MAP_TOPK = 5000
MAP_QUERY_SIZE = 2000

TOP_21_LABELS = [
    'sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants',
    'lake', 'ocean', 'road', 'flowers', 'sunset', 'reflection', 'rocks', 'vehicle', 'tree',
    'snow', 'beach', 'mountain'
]

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'NUSWIDE_Top21', 'DPSH', 'ResNet50', f'{BIT}bits')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class DPSH_ResNet50(nn.Module):
    def __init__(self, bit=16, pretrained=True):
        super(DPSH_ResNet50, self).__init__()
        self.bit = bit
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, bit)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.fc(x)
        return h, torch.sign(h)

    def get_hash(self, x):
        with torch.no_grad():
            h, b = self.forward(x)
        return h, b


def train_dpsh(bit, train_loader, query_loader, db_loader, eval_indices, start_epoch=0, best_map=0.0, resume_path=None):
    model = DPSH_ResNet50(bit).to(DEVICE)
    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path))
        print(f"Loaded resume checkpoint: {resume_path}")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_log = {
        'epochs': [],
        'loss': [],
        'map': [],
        'best_map': best_map,
        'best_epoch': start_epoch
    }
    
    log_file = os.path.join(RESULTS_DIR, 'train_log.json')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            saved_log = json.load(f)
            train_log.update(saved_log)
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            S = (torch.matmul(labels, labels.t()) > 0).float()

            h, b = model(images)
            loss = calc_dpsh_loss_standard(h, h, S, b, gamma=GAMMA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            map_val = evaluate(model, query_loader, db_loader, eval_indices)
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, mAP@5000={map_val:.4f}')
            
            train_log['epochs'].append(epoch + 1)
            train_log['loss'].append(avg_loss)
            train_log['map'].append(map_val)
            
            if map_val > train_log['best_map']:
                train_log['best_map'] = map_val
                train_log['best_epoch'] = epoch + 1
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_best.pth'))
                print(f'Best mAP updated: {map_val:.4f}')
            
            with open(log_file, 'w') as f:
                json.dump(train_log, f, cls=NpEncoder)
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_final.pth'))
    
    return train_log['best_map']


def evaluate(model, query_loader, db_loader, eval_indices):
    model.eval()
    
    query_features = []
    query_labels = []
    with torch.no_grad():
        for images, labels in query_loader:
            images = images.to(DEVICE)
            h, b = model.get_hash(images)
            query_features.append(h.cpu())
            query_labels.append(labels)
    
    db_features = []
    db_labels = []
    with torch.no_grad():
        for images, labels in db_loader:
            images = images.to(DEVICE)
            h, b = model.get_hash(images)
            db_features.append(h.cpu())
            db_labels.append(labels)
    
    query_features = torch.cat(query_features).numpy()
    query_labels = torch.cat(query_labels).numpy()
    db_features = torch.cat(db_features).numpy()
    db_labels = torch.cat(db_labels).numpy()
    
    query_binary = np.sign(query_features)
    db_binary = np.sign(db_features)
    
    query_labels = query_labels[:, eval_indices]
    db_labels = db_labels[:, eval_indices]
    map_val = calculate_mAP(query_binary, db_binary, query_labels, db_labels, topk=MAP_TOPK)
    return map_val


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("Loading NUS-WIDE dataset...")
    train_dataset = NUSWIDEDataset(root=DATA_ROOT, transform=transform, train=True)
    test_dataset = NUSWIDEDataset(root=DATA_ROOT, transform=transform, train=False)
    
    print(f"Train Size: {len(train_dataset)}, Test Size: {len(test_dataset)}")
    
    missing_labels = [l for l in TOP_21_LABELS if l not in test_dataset.label_cols]
    if missing_labels:
        raise ValueError(f"Missing Top-21 labels in dataset: {missing_labels}")
    eval_indices = [test_dataset.label_cols.index(l) for l in TOP_21_LABELS]
    test_top21 = test_dataset.df[TOP_21_LABELS].values.sum(axis=1) > 0
    query_candidates = np.where(test_top21)[0]
    rng = np.random.RandomState(SEED)
    query_size = min(MAP_QUERY_SIZE, len(query_candidates))
    query_indices = rng.choice(query_candidates, query_size, replace=False).tolist()
    query_subset = Subset(test_dataset, query_indices)
    db_subset = train_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    db_loader = DataLoader(db_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Query: {len(query_subset)}, Database: {len(db_subset)}")
    
    start_epoch = 0
    best_map = 0.0
    
    resume_path = None
    if args.resume:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{args.resume}.pth')
        if os.path.exists(checkpoint_path):
            print(f'Resumed from epoch {args.resume}')
            start_epoch = args.resume
            resume_path = checkpoint_path
            
            log_file = os.path.join(RESULTS_DIR, 'train_log.json')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    saved_log = json.load(f)
                    best_map = saved_log.get('best_map', 0.0)
        else:
            final_path = os.path.join(CHECKPOINT_DIR, 'model_final.pth')
            if os.path.exists(final_path):
                print(f'Loading fallback checkpoint: {final_path}')
                resume_path = final_path
                log_file = os.path.join(RESULTS_DIR, 'train_log.json')
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        saved_log = json.load(f)
                        best_map = saved_log.get('best_map', 0.0)
    
    best_map = train_dpsh(BIT, train_loader, query_loader, db_loader, eval_indices, start_epoch, best_map, resume_path)
    print(f'Training completed! Best mAP: {best_map:.4f}')


if __name__ == "__main__":
    main()

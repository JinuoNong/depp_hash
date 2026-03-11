import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
parser.add_argument('--resume', type=int, default=None, help='Resume from epoch')
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import sys
import json
import datetime
import random

import scipy.linalg

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
from src.utils.metrics import calculate_mAP

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_csq_center(bit, num_classes, device=DEVICE):
    if bit < num_classes:
        centers = torch.randint(0, 2, (num_classes, bit), device=device, dtype=torch.float32)
        centers = centers * 2 - 1
        return centers

    H = scipy.linalg.hadamard(bit)
    centers = torch.from_numpy(H).to(device=device, dtype=torch.float32)
    if bit > num_classes:
        idx = torch.randperm(bit, device=device)[:num_classes]
        centers = centers[idx]
    return centers

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'NUSWIDE')
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-4
BIT = 32
GAMMA = 0.1
MAP_TOPK = 5000
MAP_QUERY_SIZE = 2000
NUM_CLASSES = 81
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, 'models', 'pretrained', 'resnet50_imagenet1k_v1.pth')

TOP_21_LABELS = [
    'sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants',
    'lake', 'ocean', 'road', 'flowers', 'sunset', 'reflection', 'rocks', 'vehicle', 'tree',
    'snow', 'beach', 'mountain'
]

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'NUSWIDE_Top21', 'CSQ', 'ResNet50', f'{BIT}bits')
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


class CSQ_ResNet50(nn.Module):
    def __init__(self, bit=32, num_classes=81, pretrained_path=None):
        super(CSQ_ResNet50, self).__init__()
        self.bit = bit
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            resnet.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {pretrained_path}")
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        self.hash_layer = nn.Linear(self.feature_dim, bit)
        self.bn_layer = nn.BatchNorm1d(bit, affine=False)
        self.register_buffer('class_centers', get_csq_center(bit, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.hash_layer(x)
        h = self.bn_layer(h)
        b = torch.tanh(h)
        return h, b

    def get_hash(self, x):
        with torch.no_grad():
            h, _ = self.forward(x)
            b = torch.sign(h)
        return h, b


class CSQLoss_MultiLabel(nn.Module):
    def __init__(self, gamma=0.1):
        super(CSQLoss_MultiLabel, self).__init__()
        self.gamma = gamma

    def forward(self, h, b, labels, class_centers):
        labels_float = labels.float()
        center_targets = torch.matmul(labels_float, class_centers)
        label_sum = labels_float.sum(dim=1, keepdim=True).clamp(min=1)
        center_targets = center_targets / label_sum
        
        h_tanh = torch.tanh(h)
        center_loss = F.mse_loss(h_tanh, center_targets)
        b_binary = torch.sign(h).detach()
        quantization_loss = F.mse_loss(h_tanh, b_binary)
        
        return center_loss + self.gamma * quantization_loss


def train_csq(bit, train_loader, query_loader, db_loader, eval_indices=None, resume_from=None):
    print(f"\n>>> Start Training CSQ-ResNet50 ({bit} bits) for {EPOCHS} epochs on NUS-WIDE <<<")
    print(f">>> Using GPU: {DEVICE}")

    model = CSQ_ResNet50(bit=bit, num_classes=NUM_CLASSES, pretrained_path=PRETRAINED_PATH).to(DEVICE)
    criterion = CSQLoss_MultiLabel(gamma=GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    best_map = 0.0

    if resume_from is not None and resume_from > 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{resume_from}.pth")
        if os.path.exists(checkpoint_path):
            print(f">>> Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = resume_from
            best_map = checkpoint.get('best_map', 0.0)
            print(f">>> Resumed from epoch {resume_from}, best mAP: {best_map:.4f}")
        else:
            print(f">>> Warning: Checkpoint {checkpoint_path} not found, starting from scratch")

    log_data = {
        "config": {
            "model": "CSQ",
            "backbone": "ResNet50",
            "bit": bit,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "gamma": GAMMA,
            "dataset": "NUS-WIDE (Full Train, Top-21 Eval)",
            "map_topk": MAP_TOPK,
            "map_subset_size": {"query": MAP_QUERY_SIZE, "db": "Full Train Dataset"}
        },
        "epochs": [],
        "loss": [],
        "map": [],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            h, b = model(images)
            loss = criterion(h, b, labels, model.class_centers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        log_data["epochs"].append(epoch + 1)
        log_data["loss"].append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            print(f"Calculating mAP@{MAP_TOPK}...")
            model.eval()
            qB, qL = [], []
            rB, rL = [], []

            with torch.no_grad():
                for images, labels in query_loader:
                    images = images.to(DEVICE)
                    _, b = model.get_hash(images)
                    qB.append(b.cpu().numpy())
                    qL.append(labels.numpy())

                for images, labels in db_loader:
                    images = images.to(DEVICE)
                    _, b = model.get_hash(images)
                    rB.append(b.cpu().numpy())
                    rL.append(labels.numpy())

            qB = np.concatenate(qB)
            qL = np.concatenate(qL)
            rB = np.concatenate(rB)
            rL = np.concatenate(rL)

            if eval_indices is not None:
                qL = qL[:, eval_indices]
                rL = rL[:, eval_indices]

            mAP = calculate_mAP(qB, rB, qL, rL, verbose=True, topk=MAP_TOPK)
            log_data["map"].append(mAP)
            print(f"Epoch {epoch+1} mAP@{MAP_TOPK}: {mAP:.4f}")

            if mAP > best_map:
                best_map = mAP
                best_model_path = os.path.join(CHECKPOINT_DIR, "model_best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                    'config': log_data["config"]
                }, best_model_path)
                print(f"[*] New Best mAP! Model saved to {best_model_path}")
        else:
            log_data["map"].append(None)

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

        log_path = os.path.join(RESULTS_DIR, 'train_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4, cls=NpEncoder)

    final_model_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map,
        'config': log_data["config"]
    }, final_model_path)
    print(f"Final Model saved to {final_model_path}")
    print(f"Training Complete. Best mAP: {best_map:.4f}")


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading NUS-WIDE dataset...")
    train_dataset = NUSWIDEDataset(root=DATA_ROOT, transform=transform, train=True, selected_labels=None)
    test_dataset = NUSWIDEDataset(root=DATA_ROOT, transform=transform, train=False, selected_labels=None)

    print(f"Full Train Size: {len(train_dataset)}, Full Test Size: {len(test_dataset)}")

    all_labels = train_dataset.label_cols.tolist()
    missing_labels = [l for l in TOP_21_LABELS if l not in all_labels]
    if missing_labels:
        raise ValueError(f"Missing Top-21 labels in dataset: {missing_labels}")
    top21_indices = [all_labels.index(l) for l in TOP_21_LABELS]

    top21_subset_df = test_dataset.df[TOP_21_LABELS]
    valid_query_mask = top21_subset_df.sum(axis=1) > 0
    valid_query_indices = np.where(valid_query_mask)[0]

    print(f"Found {len(valid_query_indices)} images in Test set belonging to Top-21.")

    if len(valid_query_indices) > MAP_QUERY_SIZE:
        query_indices = np.random.choice(valid_query_indices, MAP_QUERY_SIZE, replace=False)
    else:
        query_indices = valid_query_indices

    query_subset = Subset(test_dataset, query_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    query_loader = DataLoader(query_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    db_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"mAP Validation (Top-21) - Query: {len(query_subset)}, Database: {len(train_dataset)}")

    train_csq(BIT, train_loader, query_loader, db_loader, eval_indices=top21_indices, resume_from=args.resume)


if __name__ == "__main__":
    main()

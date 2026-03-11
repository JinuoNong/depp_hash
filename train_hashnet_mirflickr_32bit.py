import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
parser.add_argument('--resume', type=int, default=None, help='Resume from epoch (e.g., 30 means start from epoch 31)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn as nn
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
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.losses import calc_dpsh_loss_standard
from src.utils.metrics import calculate_mAP

DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'mirflickr25k')
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-4
BIT = 32
GAMMA = 0.1
MAP_TOPK = 5000

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'MIRFlickr25K', 'HashNet', 'ResNet50', f'{BIT}bits')
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


class MIRFlickr25K(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'mirflickr')
        self.anno_dir = os.path.join(root_dir, 'annotations')

        self.classes = ['animals', 'baby', 'bird', 'car', 'clouds', 'dog', 'female', 'flower', 'food', 'indoor', 'lake', 'male', 'night', 'people', 'plant_life', 'portrait', 'river', 'sea', 'sky', 'structures', 'sunset', 'transport', 'tree', 'water']

        self.labels = np.zeros((25000, 24), dtype=np.float32)

        for idx, cls in enumerate(self.classes):
            file_path = os.path.join(self.anno_dir, f'{cls}.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        img_id = int(line.strip()) - 1
                        self.labels[img_id, idx] = 1.0

        self.valid_indices = np.where(self.labels.sum(axis=1) > 0)[0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = os.path.join(self.img_dir, f'im{real_idx + 1}.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[real_idx]
        return image, torch.tensor(label)


class HashNet_ResNet50(nn.Module):
    def __init__(self, bit=32, pretrained=True):
        super(HashNet_ResNet50, self).__init__()
        self.bit = bit
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        self.hash_layer = nn.Linear(self.feature_dim, bit)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.hash_layer(x)
        b = torch.tanh(h)
        return h, b

    def get_hash(self, x):
        with torch.no_grad():
            h, b = self.forward(x)
        return h, b


def train_hashnet(bit, train_loader, query_loader, db_loader, resume_from=None):
    print(f"\n>>> Start Training HashNet-ResNet50 ({bit} bits) for {EPOCHS} epochs on MIRFlickr25K <<<")
    print(f">>> Using GPU: {DEVICE}")

    model = HashNet_ResNet50(bit=bit).to(DEVICE)
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
            "model": "HashNet",
            "backbone": "ResNet50",
            "bit": bit,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "gamma": GAMMA,
            "dataset": "MIRFlickr25K",
            "map_topk": MAP_TOPK
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
            S = (torch.matmul(labels, labels.t()) > 0).float()

            h, b = model(images)
            loss = calc_dpsh_loss_standard(h, h, S, b, gamma=GAMMA)

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
                    _, b = model(images)
                    qB.append(b.cpu().numpy())
                    qL.append(labels.numpy())

                for images, labels in db_loader:
                    images = images.to(DEVICE)
                    _, b = model(images)
                    rB.append(b.cpu().numpy())
                    rL.append(labels.numpy())

            qB = np.concatenate(qB)
            qL = np.concatenate(qL)
            rB = np.concatenate(rB)
            rL = np.concatenate(rL)

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
            'loss': avg_loss,
            'best_map': best_map
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

    print("Loading MIRFlickr25K dataset...")
    full_dataset = MIRFlickr25K(root_dir=DATA_ROOT, transform=transform)

    print(f"Full Dataset Size: {len(full_dataset)}")

    split_path = os.path.join(DATA_ROOT, 'split_mirflickr_25k_standard.json')
    if os.path.exists(split_path):
        print(f"Loading standard split from {split_path}")
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        query_indices = split_data['query']
        db_indices = split_data['database']
        train_indices = db_indices
    else:
        print("Standard split not found, using full-training fallback split")
        rng = np.random.RandomState(42)
        num_query = min(2000, len(full_dataset))
        query_indices = sorted(rng.choice(len(full_dataset), num_query, replace=False).tolist())
        query_index_set = set(query_indices)
        db_indices = [i for i in range(len(full_dataset)) if i not in query_index_set]
        train_indices = db_indices

    train_subset = Subset(full_dataset, train_indices)
    query_subset = Subset(full_dataset, query_indices)
    db_subset = Subset(full_dataset, db_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    db_loader = DataLoader(db_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_subset)}, Query: {len(query_subset)}, Database: {len(db_subset)}")

    train_hashnet(BIT, train_loader, query_loader, db_loader, resume_from=args.resume)


if __name__ == "__main__":
    main()

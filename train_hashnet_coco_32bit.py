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
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'MS COCO2014')
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-4
BIT = 32
GAMMA = 0.1
MAP_TOPK = 5000

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'MSCOCO', 'HashNet', 'ResNet50', f'{BIT}bits')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class MSCOCO2014(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, f'{split}2014')
        self.anno_file = os.path.join(root_dir, 'annotations', f'instances_{split}2014.json')

        self.num_classes = len(COCO_CLASSES)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(COCO_CLASSES)}

        self.image_ids = []
        self.labels = {}

        if os.path.exists(self.anno_file):
            with open(self.anno_file, 'r') as f:
                coco_anno = json.load(f)

            self.images = {img['id']: img for img in coco_anno['images']}
            self.categories = coco_anno['categories']

            cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}

            annotations_by_image = {}
            for ann in coco_anno['annotations']:
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)

            for img_id, anns in annotations_by_image.items():
                label = np.zeros(self.num_classes, dtype=np.float32)
                for ann in anns:
                    cat_idx = cat_id_to_idx.get(ann['category_id'])
                    if cat_idx is not None and cat_idx < self.num_classes:
                        label[cat_idx] = 1.0

                if label.sum() > 0:
                    self.image_ids.append(img_id)
                    self.labels[img_id] = label

            print(f"Loaded {len(self.image_ids)} images with annotations for {split} split")
        else:
            print(f"Annotation file not found: {self.anno_file}")
            self.image_ids = []
            self.labels = {}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_filename = img_info['file_name']

        img_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[img_id]
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
    print(f"\n>>> Start Training HashNet-ResNet50 ({bit} bits) for {EPOCHS} epochs on MSCOCO2014 <<<")
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
            "dataset": "MSCOCO2014",
            "num_classes": len(COCO_CLASSES),
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

    print("Loading MSCOCO2014 train dataset...")
    train_dataset = MSCOCO2014(root_dir=DATA_ROOT, split='train', transform=transform)
    print(f"Train Dataset Size: {len(train_dataset)}")

    print("Loading MSCOCO2014 val dataset...")
    val_dataset = MSCOCO2014(root_dir=DATA_ROOT, split='val', transform=transform)
    print(f"Val Dataset Size: {len(val_dataset)}")

    rng = np.random.RandomState(42)

    num_query = min(5000, len(val_dataset))
    query_indices = sorted(rng.choice(len(val_dataset), num_query, replace=False).tolist())
    query_index_set = set(query_indices)
    db_indices = [i for i in range(len(val_dataset)) if i not in query_index_set]

    query_subset = Subset(val_dataset, query_indices)
    db_subset = Subset(val_dataset, db_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    db_loader = DataLoader(db_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Query: {len(query_subset)}, Database: {len(db_subset)}")

    train_hashnet(BIT, train_loader, query_loader, db_loader, resume_from=args.resume)


if __name__ == "__main__":
    main()

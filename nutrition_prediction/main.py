#!/usr/bin/env python
# coding: utf-8
import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
import warnings

# Setup
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.manual_seed(88)
np.random.seed(88)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("outputs", exist_ok=True)


# Dataset definition
class NutritionDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, img_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size

        df = pd.read_csv(csv_file)

        # Data cleansing
        df = df.dropna(subset=['ID', 'Value']).reset_index(drop=True)

        df['ID'] = df['ID'].astype(str).str.strip().str.split('.').str[0]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dish_id = str(self.df.iloc[idx]['ID']).strip().split('.')[0]
        calories = self.df.iloc[idx]['Value']

        # RGB image
        rgb_path = os.path.join(self.data_dir, 'color', dish_id, 'rgb.png')
        rgb_image = Image.open(rgb_path).convert('RGB').resize(self.img_size)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            rgb_image = transforms.ToTensor()(rgb_image)

        # Depth_raw image
        depth_path = os.path.join(self.data_dir, 'depth_raw', dish_id, 'depth_raw.png')
        depth_image = Image.open(depth_path).convert('L').resize(self.img_size)
        depth_image = np.array(depth_image).astype(np.float32)
        depth_image = np.log1p(depth_image)
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        depth_image = torch.tensor(depth_image).unsqueeze(0)

        return rgb_image, depth_image, torch.tensor(calories, dtype=torch.float32)



# Model definition
class VolumeCompositionNet(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()

        # Two branches: volumn estimator + food type classifier
        # Parameters have been tunning to its optimal performance
        self.volume_branch = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256), nn.ReLU(),

            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.composition_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

        # Alpha fusion parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Project both branches to the same dimension
        self.v_fc = nn.Linear(512, 256)
        self.c_fc = nn.Linear(512, 256)

        # MLP fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, rgb, depth):
        v = self.volume_branch(depth).view(depth.size(0), -1)
        c = self.composition_branch(rgb).view(rgb.size(0), -1)
        v, c = self.v_fc(v), self.c_fc(c)

        # Alpha weighted fusion and nonlinear fusion
        fused = torch.cat([
            self.alpha * v,
            (1 - self.alpha) * c,
            v * c
        ], dim=1)

        out = self.fusion(fused)
        return out, v, c

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for rgb, depth, y in tqdm(dataloader, desc="Training", ncols=100):
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
        optimizer.zero_grad()
        out, _, _ = model(rgb, depth)
        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * rgb.size(0)
    return total_loss / len(dataloader.dataset)

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, preds, tgts = 0, [], []
    with torch.no_grad():
        for rgb, depth, y in tqdm(dataloader, desc="Validation", ncols=100):
            rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
            out, _, _ = model(rgb, depth)
            loss = criterion(out.squeeze(), y)
            total_loss += loss.item() * rgb.size(0)
            preds.extend(out.squeeze().cpu().numpy())
            tgts.extend(y.cpu().numpy())
    return total_loss / len(dataloader.dataset), np.array(preds), np.array(tgts)

def calculate_metrics(pred, y):
    mse = np.mean((pred - y)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - y))
    rel = np.mean(np.abs(pred - y) / (y + 1e-8)) * 100
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Relative_Error(%)": rel}

# Main training workflow
def main():
    print(f"Using device: {device}")
    print("Initializing and starting training...")

    base_dir = "outputs"
    exp_idx = 0
    while os.path.exists(f"{base_dir}_{exp_idx}"):
        exp_idx += 1
    output_dir = f"{base_dir}_{exp_idx}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Current experiment output directory: {output_dir}")

    data_dir = r"E:\LUZewen\University of Melbourne\25s2\COMP90086\3. assignments\Assignment_final\comp-90086-nutrition-5-k\Nutrition5K\Nutrition5K\train"
    csv_file = r"E:\LUZewen\University of Melbourne\25s2\COMP90086\3. assignments\Assignment_final\comp-90086-nutrition-5-k\Nutrition5K\Nutrition5K\nutrition5k_train.csv"

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    dataset = NutritionDataset(data_dir, csv_file, transform=data_transforms['train'])
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    val_set.dataset.transform = data_transforms['val']

    num_workers = min(8, os.cpu_count() // 2)
    print(f"Detected CPU cores: {os.cpu_count()}, using num_workers={num_workers}")
    train_loader = DataLoader(train_set, 32, True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, 32, False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    model = VolumeCompositionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 5)

    best_loss, patience, no_improve = float('inf'), 10, 0
    train_history, val_history, val_metrics_history = [], [], []

    # Training loop: 60 epoch at most, with early stop using patience parameter = 10
    for epoch in range(60):
        print(f"\nEpoch {epoch+1}/60")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, preds, tgts = validate_epoch(model, val_loader, criterion, device)
        metrics = calculate_metrics(preds, tgts)

        # MSE is used for one of the tunning criterions
        print(f"Train={train_loss:.4f} | Val={val_loss:.4f} | MSE={metrics['MSE']:.2f}")
        print(f"Current alpha = {model.alpha.item():.3f}")

        train_history.append(train_loss)
        val_history.append(val_loss)
        val_metrics_history.append(metrics)

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
            print("Saved best model")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

    print("Training finished!")

    # Plot training curves and output it to a targeted folder
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot([m['RMSE'] for m in val_metrics_history], color='orange')
    plt.title('RMSE Curve')
    plt.subplot(1, 3, 3)
    plt.plot([m['MAE'] for m in val_metrics_history], color='green')
    plt.title('MAE Curve')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300)
    plt.close()

    # Load best model
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pth"))

    # Visualize samples with smallest and largest prediction errors (5 for each)
    def visualize_predictions_by_error(model, dataloader, device, criterion, num_samples_each=5):
        model.eval()
        preds, tgts, rgbs, depths = [], [], [], []

        with torch.no_grad():
            for rgb, depth, y in dataloader:
                rgb, depth = rgb.to(device), depth.to(device)
                out, _, _ = model(rgb, depth)
                preds.append(out.squeeze().cpu())
                tgts.append(y)
                rgbs.append(rgb.cpu())
                depths.append(depth.cpu())

        # Merge tensors globally to avoid index overflow
        preds = torch.cat(preds)
        tgts = torch.cat(tgts)
        rgbs = torch.cat(rgbs)
        depths = torch.cat(depths)

        errors = torch.abs(preds - tgts)
        sorted_idx = torch.argsort(errors)
        idxs = torch.cat([sorted_idx[:num_samples_each], sorted_idx[-num_samples_each:]])

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        fig, axes = plt.subplots(len(idxs), 3, figsize=(15, 4 * len(idxs)))
        for i, idx in enumerate(idxs):
            rgb_disp = rgbs[idx] * std + mean
            rgb_disp = torch.clamp(rgb_disp, 0, 1)
            depth_disp = depths[idx] * std + mean
            depth_disp = torch.clamp(depth_disp, 0, 1)

            axes[i, 0].imshow(rgb_disp.permute(1, 2, 0))
            axes[i, 0].set_title("RGB")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(depth_disp.permute(1, 2, 0))
            axes[i, 1].set_title("Depth")
            axes[i, 1].axis("off")

            # Bar comparison
            axes[i, 2].bar(["real", "predict"], [tgts[idx], preds[idx]])
            axes[i, 2].set_title(f"error: {errors[idx]:.2f}")
            axes[i, 2].set_ylabel("calories")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_samples.png", dpi=300)
        plt.close()
        print(f"Error visualization saved -> {output_dir}/error_samples.png")

    visualize_predictions_by_error(model, val_loader, device, criterion)

    val_loss, preds, tgts = validate_epoch(model, val_loader, criterion, device)
    metrics = calculate_metrics(preds, tgts)
    plt.figure(figsize=(8,6))
    plt.scatter(tgts, preds, alpha=0.6)
    plt.plot([tgts.min(), tgts.max()], [tgts.min(), tgts.max()], 'r--')
    plt.title(f"Pred vs Real (R²={r2_score(tgts,preds):.3f})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/val_scatter.png", dpi=300)
    plt.close()

    # Test set prediction
    class TestDataset(Dataset):
        def __init__(self, test_dir, transform=None, img_size=(224, 224)):
            self.color_dir = os.path.join(test_dir, "color")
            self.depth_dir = os.path.join(test_dir, "depth_raw")
            self.ids = sorted(os.listdir(self.color_dir))
            self.transform = transform
            self.img_size = img_size

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            dish = self.ids[idx]

            # RGB image
            rgb_path = os.path.join(self.color_dir, dish, "rgb.png")
            rgb = Image.open(rgb_path).convert("RGB").resize(self.img_size)
            rgb = self.transform(rgb) if self.transform else transforms.ToTensor()(rgb)

            # Depth image
            depth_path = os.path.join(self.depth_dir, dish, "depth_raw.png")
            depth = Image.open(depth_path).convert("L").resize(self.img_size)
            depth = np.array(depth).astype(np.float32)
            depth = depth / (depth.max() + 1e-8)
            depth = torch.tensor(depth).unsqueeze(0)

            return rgb, depth, dish

    test_dir = r"E:\LUZewen\University of Melbourne\25s2\COMP90086\3. assignments\Assignment_final\comp-90086-nutrition-5-k\Nutrition5K\Nutrition5K\test"
    test_loader = DataLoader(TestDataset(test_dir, transform=data_transforms['val']), 16, False)
    model.eval(); preds, ids = [], []
    with torch.no_grad():
        for rgb, depth, did in tqdm(test_loader, desc="Testing"):
            rgb, depth = rgb.to(device), depth.to(device)
            out,_,_ = model(rgb, depth)
            preds += out.squeeze().cpu().tolist()
            ids += list(did)
    pd.DataFrame({"ID":ids,"Value":preds}).to_csv(f"{output_dir}/submission_final.csv", index=False)
    print(f"Test predictions saved -> {output_dir}/submission_final.csv")

    # Feature importance analysis as one of the tunning criterions
    def analyze_feature_importance(model, dataloader):
        model.eval()
        vol, comp, tgt = [], [], []
        with torch.no_grad():
            for rgb, depth, y in tqdm(dataloader, desc="Feature analysis"):
                rgb, depth = rgb.to(device), depth.to(device)
                _, v, c = model(rgb, depth)
                vol += v.norm(dim=1).cpu().tolist()
                comp += c.norm(dim=1).cpu().tolist()
                tgt += y.tolist()
        vol_corr = np.corrcoef(vol, tgt)[0,1]
        comp_corr = np.corrcoef(comp, tgt)[0,1]
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.scatter(vol, tgt, alpha=0.6); plt.title(f"Volume corr={vol_corr:.3f}")
        plt.subplot(1,2,2)
        plt.scatter(comp, tgt, alpha=0.6, color="orange"); plt.title(f"Composition corr={comp_corr:.3f}")
        plt.tight_layout(); plt.savefig(f"{output_dir}/feature_importance.png", dpi=300); plt.close()
        print(f"Volume correlation={vol_corr:.3f}, Composition correlation={comp_corr:.3f}")

    analyze_feature_importance(model, val_loader)
    print(f"Feature importance analysis saved -> {output_dir}/feature_importance.png")

if __name__ == "__main__":
    main()

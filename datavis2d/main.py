import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Visualization Tool")

        # データフォルダの選択ボタン
        self.data_folder = tk.StringVar()
        tk.Label(root, text="データフォルダ:").grid(row=0, column=0, padx=10, pady=10)
        tk.Entry(root, textvariable=self.data_folder, width=50).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(root, text="参照", command=self.browse_folder).grid(row=0, column=2, padx=10, pady=10)

        # モデルの選択
        tk.Label(root, text="モデル:").grid(row=1, column=0, padx=10, pady=10)
        self.model_choice = ttk.Combobox(root, values=["ResNet50", "ResNet18"], state="readonly")
        self.model_choice.grid(row=1, column=1, padx=10, pady=10)
        self.model_choice.current(0)

        # 次元削減手法の選択
        tk.Label(root, text="次元削減手法:").grid(row=2, column=0, padx=10, pady=10)
        self.dim_reduction_choice = ttk.Combobox(root, values=["t-SNE", "UMAP"], state="readonly")
        self.dim_reduction_choice.grid(row=2, column=1, padx=10, pady=10)
        self.dim_reduction_choice.current(0)

        # プロットボタン
        tk.Button(root, text="プロット", command=self.plot_data).grid(row=3, column=1, padx=10, pady=20)

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        self.data_folder.set(folder_selected)

    def plot_data(self):
        data_folder = self.data_folder.get()
        if not data_folder:
            messagebox.showwarning("警告", "データフォルダを選択してください")
            return

        # 画像変換
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # データセットとデータローダー
        dataset = ImageFolderDataset(root_dir=data_folder, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # モデルの選択
        model_name = self.model_choice.get()
        if model_name == "ResNet50":
            model = models.resnet50(pretrained=True)
        elif model_name == "ResNet18":
            model = models.resnet18(pretrained=True)

        model = torch.nn.Sequential(*list(model.children())[:-1])  # 最後の分類層を除去
        model.eval()

        # 特徴量の抽出
        features = []
        labels = []

        with torch.no_grad():
            for images, lbls in dataloader:
                output = model(images)
                features.append(output.squeeze(-1).squeeze(-1).numpy())
                labels.extend(lbls.numpy())

        features = np.concatenate(features, axis=0)

        # 次元削減
        dim_reduction_method = self.dim_reduction_choice.get()
        if dim_reduction_method == "t-SNE":
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)
        elif dim_reduction_method == "UMAP":
            umap_model = umap.UMAP(n_components=2, random_state=42)
            reduced_features = umap_model.fit_transform(features)

        # 可視化
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(dataset.classes):
            indices = np.where(np.array(labels) == i)
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=class_name)

        plt.legend()
        plt.title(f"2D Visualization using {model_name} and {dim_reduction_method}")
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

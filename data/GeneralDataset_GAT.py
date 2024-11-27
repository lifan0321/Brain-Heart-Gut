import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root, fold = 1, transform=None, split=' '):
        self.root_feats = os.path.join(root, fold)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.root_feats):
            feats_class_brain = os.path.join(self.root_feats, class_folder)

            if os.path.isdir(feats_class_brain):
                for feats_brain in os.listdir(feats_class_brain):
                    feats_brain_path = os.path.join(feats_class_brain, feats_brain)
                    sample_list.append((feats_brain_path, int(class_folder)))

        return sample_list
    
    def _get_sample_weights(self):
        class_counts = {}
        total_samples = len(self.samples)

        # 统计每个类别的样本个数
        for _, _, label in self.samples:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # 计算每个类别的权重
        weights = [1.0 / class_counts[label] for _, _, label in self.samples]
        return weights

    def __len__(self):
        return len(self.samples)

    def __normalize__(self, mx):
        rowsum = torch.sum(mx, dim=1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[r_inv == float('inf')] = 0
        r_inv[r_inv == float('-inf')] = 0        
        r_mat_inv = torch.diag(r_inv)
        # print(r_mat_inv)
        result = torch.matmul(r_mat_inv, mx)
        return result

    def __getitem__(self, index):
        feats_brain_path, label = self.samples[index]
        feats_brain = torch.load(feats_brain_path).to('cuda')
        dim_size = feats_brain.size(0)
        corr_brain = torch.ones((dim_size, dim_size)).to('cuda')
        label_tensor = torch.tensor(label).to('cuda')

        return feats_brain, corr_brain, label_tensor
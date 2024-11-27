import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BHGDataset(Dataset):
    def __init__(self, root_feats_brain, root_feats_all, fold = 1, transform=None, split=' '):
        self.root_feats_brain = os.path.join(root_feats_brain, fold)
        self.root_feats_all = os.path.join(root_feats_all, fold)

        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.root_feats_brain):
            feats_class_brain = os.path.join(self.root_feats_brain, class_folder)
            feats_class_all = os.path.join(self.root_feats_all, class_folder)

            if os.path.isdir(feats_class_brain):
                for feats_brain in os.listdir(feats_class_brain):
                    parts = feats_brain.split("_")
                    subject_name = "_".join(parts[:2])
                    feats_brain_path = os.path.join(feats_class_brain, feats_brain)
                    feats_all = subject_name + "_all.pth"
                    feats_all_path = os.path.join(feats_class_all, feats_all)
                    sample_list.append((feats_brain_path, feats_all_path, int(class_folder)))
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
        feats_brain_path, feats_all_path, label = self.samples[index]

        feats_brain = torch.load(feats_brain_path).to('cuda')
        dim_size_brain = feats_brain.size(0)
        corr_brain = torch.ones((dim_size_brain, dim_size_brain)).to('cuda')        

        feats_all = torch.load(feats_all_path).to('cuda')
        dim_size_all = feats_all.size(0)
        corr_all = torch.ones((dim_size_all, dim_size_all)).to('cuda')        
        label_tensor = torch.tensor(label).to('cuda')

        return feats_brain, corr_brain, feats_all, corr_all, label_tensor

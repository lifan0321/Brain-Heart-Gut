import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root, fold_path, transform=None, split=' '):
        self.root_feats = root
        self.root_fold = os.path.join(fold_path, split)

        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.root_fold):
            class_folder_path = os.path.join(self.root_fold, class_folder)  

            for subnamedir in os.listdir(class_folder_path):
                sub_sample_list = []

                subname_feats_brain = os.path.join(self.root_feats, subnamedir)
                organs = os.listdir(subname_feats_brain)
                organs.sort()
                for organ in organs:
                    feats_brain_path = os.path.join(subname_feats_brain, organ, subnamedir + "_" + organ + ".pth")
                    sub_sample_list.append(feats_brain_path)
                sample_list.append((sub_sample_list, int(class_folder)))

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

    def __split_patches__(self, mask):
        patch_size = 8

        region_patch_feats = mask.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        region_patch_feats = region_patch_feats.contiguous().view(-1, patch_size ** 3)
        return region_patch_feats

    def __getitem__(self, index):
        sub_sample_list, _ = self.samples[index]
        region_num = len(sub_sample_list)
        patch_feats = []
        feats = []
        for idx in range(region_num):
            feats_brain_path = sub_sample_list[idx]
            feats_brain = torch.load(feats_brain_path)
            if self.transform:
                feats_brain = self.transform(feats_brain)
            feats.append(feats_brain)
            feats_brain_feats = self.__split_patches__(feats_brain)
            patch_feats.append(feats_brain_feats)
            del feats_brain, feats_brain_feats
        return patch_feats, feats
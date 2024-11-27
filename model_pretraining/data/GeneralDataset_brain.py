import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root, fold_path, transform=None, split=' '):
        self.root_feats_brain = root
        self.root_fold = os.path.join(fold_path, split)
        self.transform = transform
        
        self.brain_91 = ['Accumbens_Area_L', 'Accumbens_Area_R', 'Amygdala_L', 'Amygdala_R', 'CC_Anterior', 'CC_Central', 
                        'CC_Mid_Anterior', 'CC_Mid_Posterior', 'CC_Posterior', 'Caudate_L', 'Caudate_R', 'Choroid_Plexus_L', 
                        'Choroid_Plexus_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 
                        'Cingulum_Post_R', 'Cuneus_L', 'Cuneus_R', 'Entorhinal_L', 'Entorhinal_R', 'Frontal_Mid_Caudal_L', 
                        'Frontal_Mid_Caudal_R', 'Frontal_Mid_Rostral_L', 'Frontal_Mid_Rostral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 
                        'Frontalpole_L', 'Frontalpole_R', 'Fusiform_L', 'Fusiform_R', 'Hippocampus_L', 'Hippocampus_R', 'Insula_L', 
                        'Insula_R', 'Isthmuscingulate_L', 'Isthmuscingulate_R', 'Lingual_L', 'Lingual_R', 'Occipital_Lat_L', 
                        'Occipital_Lat_R', 'Orbitofrontal_Lat_L', 'Orbitofrontal_Lat_R', 'Orbitofrontal_Med_L', 'Orbitofrontal_Med_R', 
                        'Pallidum_L', 'Pallidum_R', 'Paracentral_L', 'Paracentral_R', 'Parahippocampal_L', 'Parahippocampal_R', 
                        'Parietal_Inf_L', 'Parietal_Inf_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parsopercularis_L', 'Parsopercularis_R',
                        'Parsorbitalis_L', 'Parsorbitalis_R', 'Parstriangularis_L', 'Parstriangularis_R', 'Pericalcarine_L', 'Pericalcarine_R',
                        'Postcentral_L', 'Postcentral_R', 'Precentral_L', 'Precentral_R', 'Precuneus_L', 'Precuneus_R', 'Putamen_L', 'Putamen_R', 
                        'Supramarginal_L', 'Supramarginal_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 
                        'Temporal_Sup_Banks_L', 'Temporal_Sup_Banks_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporalpole_L', 'Temporalpole_R', 
                        'Thalamus_L', 'Thalamus_R', 'Transversetemporal_L', 'Transversetemporal_R', 'VentralDC_L', 'VentralDC_R']

        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.root_fold):
            class_folder_path = os.path.join(self.root_fold, class_folder)  

            for subnamedir in os.listdir(class_folder_path):
                sub_sample_list = []

                subname_feats_brain = os.path.join(self.root_feats_brain, subnamedir)
                for organ in self.brain_91:
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
        sub_sample_list, label = self.samples[index]
        region_num = len(sub_sample_list)
        patch_feats = []
        for idx in range(region_num):
            feats_brain_path = sub_sample_list[idx]
            feats_brain = torch.load(feats_brain_path)
            if self.transform:
                feats_brain = self.transform(feats_brain)
            feats_brain_feats = self.__split_patches__(feats_brain)
            patch_feats.append(feats_brain_feats)
        label_tensor = torch.tensor(label)
        return patch_feats, label_tensor
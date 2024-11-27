import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root_feats_brain, root_feats_heart, root_feats_gut, fold_path, fold = 1, transform=None, split=' '):
        self.root_feats_brain = root_feats_brain
        self.root_feats_heart = root_feats_heart
        self.root_feats_gut = root_feats_gut
        self.root_fold = os.path.join(fold_path, fold, split)

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
        self.heart_23 = ['CN_Atrioventricular_ants_resample', 'CN_Sinoatrial_ants_resample',  'Valve_Aortic_ants_resample', 'Valve_Mitral_ants_resample', 'Valve_Pulmonic_ants_resample',
                         'Valve_Tricuspid_ants_resample', 'Ventricle_L_Segment1_ants_resample', 'Ventricle_L_Segment2_ants_resample', 'Ventricle_L_Segment3_ants_resample',
                         'Ventricle_L_Segment4_ants_resample', 'Ventricle_L_Segment5_ants_resample', 'Ventricle_L_Segment6_ants_resample', 'Ventricle_L_Segment7_ants_resample', 
                         'Ventricle_L_Segment8_ants_resample', 'Ventricle_L_Segment9_ants_resample', 'Ventricle_L_Segment10_ants_resample', 'Ventricle_L_Segment11_ants_resample', 
                         'Ventricle_L_Segment12_ants_resample', 'Ventricle_L_Segment13_ants_resample', 'Ventricle_L_Segment14_ants_resample', 'Ventricle_L_Segment15_ants_resample',
                         'Ventricle_L_Segment16_ants_resample', 'Ventricle_L_Segment17_ants_resample']
        self.gut_50 = ['Duodenum_0', 'Duodenum_1', 'Duodenum_2', 'Duodenum_3', 'Duodenum_4', 'Duodenum_5', 'Duodenum_6', 'Duodenum_7', 'Duodenum_8', 'Duodenum_9',
                        'Colon_0', 'Colon_1', 'Colon_2', 'Colon_3', 'Colon_4', 'Colon_5', 'Colon_6', 'Colon_7', 'Colon_8', 'Colon_9',
                        'Colon_10', 'Colon_11', 'Colon_12', 'Colon_13', 'Colon_14', 'Colon_15', 'Colon_16', 'Colon_17', 'Colon_18', 'Colon_19',  
                        'Colon_sigmoid_0', 'Colon_sigmoid_1', 'Colon_sigmoid_2', 'Colon_sigmoid_3', 'Colon_sigmoid_4', 'Colon_sigmoid_5', 'Colon_sigmoid_6', 'Colon_sigmoid_7', 'Colon_sigmoid_8', 'Colon_sigmoid_9', 
                        'Rectum_0', 'Rectum_1', 'Rectum_2', 'Rectum_3', 'Rectum_4', 'Rectum_5', 'Rectum_6', 'Rectum_7', 'Rectum_8', 'Rectum_9']

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

                subname_feats_heart = os.path.join(self.root_feats_heart, subnamedir)
                for organ in self.heart_23:
                    feats_heart_path = os.path.join(subname_feats_heart, organ, subnamedir + "_" + organ + ".pth")
                    sub_sample_list.append(feats_heart_path)

                subname_feats_gut = os.path.join(self.root_feats_gut, subnamedir)
                for organ in self.gut_50:
                    feats_gut_path = os.path.join(subname_feats_gut, organ, subnamedir + "_" + organ + ".pth")
                    sub_sample_list.append(feats_gut_path)

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
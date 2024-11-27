# This is the code for crop 8x8x8 patches
import nibabel as nib
import numpy as np
import os
from cc3d import connected_components
import torch

def img2box(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax

def crop_ROI_mask(mask):
    labels_out, N = connected_components(mask, connectivity=26, return_N=True)
    numPix = []
    for segid in range(1, N + 1):
        numPix.append([segid, (labels_out == segid).astype(np.int8).sum()])
    numPix = np.array(numPix)

    if len(numPix) != 0:
        max_connected_image = np.int8(labels_out == numPix[np.argmax(numPix[:, 1]), 0])
        min_x, max_x, min_y, max_y, min_z, max_z = img2box(max_connected_image)
    else:
        print('coarse stage does not detect the organ, will skip this case')
        return 0, 0, 0, 0, 0, 0

    x_extend, y_extend, z_extend = (0, 0, 0)

    max_x = min(max_x + x_extend, mask.shape[0])
    max_y = min(max_y + y_extend, mask.shape[1])
    max_z = min(max_z + z_extend, mask.shape[2])
    min_x = max(min_x - x_extend, 0)
    min_y = max(min_y - y_extend, 0)
    min_z = max(min_z - z_extend, 0)

    return min_x, max_x, min_y, max_y, min_z, max_z


def calculate_boundary(mask_folder):
    brain_91_region = ['Accumbens_Area_L', 'Accumbens_Area_R', 'Amygdala_L', 'Amygdala_R', 'CC_Anterior', 'CC_Central', 
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
    b_91_list = []
    for region_fd_subdir in brain_91_region:
        x_list, y_list, z_list = [], [], []
        crop_x, crop_y, crop_z = 0, 0, 0
        for subdir in subdirs:
            sub_reg_label_file = os.path.join(mask_folder, subdir, region_fd_subdir + "_rigid_mask.nii.gz")
            pet_file_info = nib.load(sub_reg_label_file)
            pet_file_arr = pet_file_info.get_fdata()
            x_start, x_end, y_start, y_end, z_start, z_end = crop_ROI_mask(pet_file_arr)
            x, y, z = x_end - x_start + 1, y_end - y_start + 1, z_end - z_start + 1
            # print(x, y, z)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        x = max(x_list)
        y = max(y_list)
        z = max(z_list)
        patch_size = 8
        x = x + (patch_size - x % patch_size) % patch_size
        y = y + (patch_size - y % patch_size) % patch_size
        z = z + (patch_size - z % patch_size) % patch_size

        b_91_list.append((region_fd_subdir, (x, y, z)))
        return b_91_list

def crop_patches(main_folder, mask_folder, b_91_list):
    subdirs = os.listdir(main_folder)
    for subdir in subdirs:
        pet_file = os.path.join(main_folder, subdir + "_pet.nii.gz")
        pet_file_info = nib.load(pet_file)
        pet_file_arr = pet_file_info.get_fdata()
        file_arr_affine = pet_file_info.affine
        subsubdir = os.path.join(mask_folder, subdir)
        
        organdirs = os.listdir(subsubdir)
        organdirs.sort()

        Pons_file = os.path.join(subsubdir, 'Pons.nii.gz') # use to calculate SUVR values
        Pons_info = nib.load(Pons_file)
        Pons_arr = Pons_info.get_fdata()
        Pons_arr = np.round(Pons_arr).astype(int)

        suvr_mask = Pons_arr

        pet_file_arr = pet_file_arr.squeeze()
        suvr_arr = pet_file_arr * suvr_mask

        elements = np.sum(suvr_mask)
        elements_value = np.sum(suvr_arr)

        suvr_value = elements_value / elements

        for organdir in organdirs:
            organrfile = os.path.join(subsubdir, organdir)
            parts = organdir.split('.')
            organ = parts[0]
            if organ not in brain_91_region:
                continue

            Duodenum_info = nib.load(organrfile)
            Duodenum_arr = Duodenum_info.get_fdata()

            value = dict(b_91_list)[organ]
            crop_x, crop_y, crop_z = value[0], value[1], value[2]
            x_start, x_end, y_start, y_end, z_start, z_end = crop_ROI_mask(Duodenum_arr)

            x_start = max((x_start - (crop_x - (x_end - x_start)) // 2), 0) 
            x_end = crop_x + x_start
            y_start = max((y_start - (crop_y - (y_end - y_start)) // 2), 0)
            y_end = crop_y + y_start
            z_start = max((z_start - (crop_z - (z_end - z_start)) // 2), 0)
            z_end = crop_z + z_start
            print(x_start, x_end, y_start, y_end, z_start, z_end)
            region = pet_file_arr[x_start:x_end, y_start:y_end, z_start:z_end]
            label = Duodenum_arr[x_start:x_end, y_start:y_end, z_start:z_end]
            mask = region * label
            mask_suvr = mask / suvr_value

            mask_name_suvr = "xxxx" # your own name
            nib.Nifti1Image(mask_suvr, file_arr_affine).to_filename(mask_name_suvr)

            mask_suvr = torch.tensor(mask_suvr)
            region_name = "xxxx" # your own name
            torch.save(mask_suvr, region_name)

            label_name_suvr = "xxxx" # your own name
            nib.Nifti1Image(label, file_arr_affine).to_filename(label_name_suvr)

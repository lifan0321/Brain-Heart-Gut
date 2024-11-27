import os
import ants
import numpy as np
import SimpleITK as sitk

# register PET to MNI 152 space
def register_PET_MNI152(pet_path, MNI152_pet_path):
    MNI152_pet_path = "template/pet.nii"
    m_img_pet = ants.image_read(pet_path)
    f_img_pet = ants.image_read(MNI152_pet_path)

    mytx = ants.registration(fixed=f_img_pet, moving=m_img_pet, type_of_transform='Rigid')
    warped_img_pet = ants.apply_transforms(fixed=f_img_pet, moving=m_img_pet, transformlist=mytx['fwdtransforms'],
                                           interpolator="linear")
    filename_pet = "rigid_pet.nii.gz"
    ants.image_write(warped_img_pet, filename_pet)

# register CT/T1 & mask to PET
def register_ct_mask_PET(ct_path, mask_path, pet_path):
    f_img_pet = ants.image_read(pet_path)
    m_img_ct = ants.image_read(ct_path)
    m_img_mask = ants.image_read(mask_path)

    mytx = ants.registration(fixed=f_img_pet, moving=m_img_ct, type_of_transform='Rigid')
    warped_img_ct = ants.apply_transforms(fixed=f_img_pet, moving=m_img_ct, transformlist=mytx['fwdtransforms'],
                                           interpolator="linear")
    warped_img_mask = ants.apply_transforms(fixed=f_img_pet, moving=m_img_mask, transformlist=mytx['fwdtransforms'],
                                           interpolator="nearestNeighbor")                                           
    filename_ct = "rigid_ct.nii.gz"
    ants.image_write(warped_img_ct, filename_ct)
    filename_mask = "rigid_mask.nii.gz"
    ants.image_write(warped_img_mask, filename_mask)

# resample images
def resample(input_file, dst_file):
    outspacing = [1, 1, 1]
    outsize = [0, 0, 0]
    arr = sitk.Image(sitk.ReadImage(input_file))
    inputsize = arr.GetSize()
    inputspacing = arr.GetSpacing() 
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
    # print(outsize)
    transform = sitk.Transform()
    transform.SetIdentity()
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(arr.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(arr.GetDirection())
    resampler.SetSize(outsize)
    resampler.SetInterpolator(sitk.sitkLinear)
    newvol = resampler.Execute(arr)
    sitk.WriteImage(newvol, dst_file)
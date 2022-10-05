import numpy as np
from pathlib import Path
import DPC_Takam_2020 as Takam
from matplotlib import pyplot as plt
import utils
#Load defective image

# raw_path        = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/scene/HisiRAW_2592x1536_12bits_RGGB_Linear_ISO300_1.raw"
# raw_filename    = Path(raw_path).stem
# GT_path         = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/Raw GT/GT_scene_"+ raw_filename + ".raw"
# mask_path       = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/Yongji/corrected masks/DPC_mask_yongji_imp_2_Defective_100_scene_HisiRAW_2592x1536_12bits_RGGB_Linear_ISO300_1.raw"
# size = (1536, 2592) #2592x1536

# raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)      # Construct an array from data in a text or binary file.
# GT = np.fromfile(GT_path, dtype="uint16").reshape(size)      # Construct an array from data in a text or binary file.
# mask = np.fromfile(mask_path, dtype="uint16").reshape(size)      # Construct an array from data in a text or binary file.

# print(raw_file[0:10,0:10])
# print(GT[0:10,0:10])
# print(mask[0:10,0:10])




# raw_file  = raw_file[10:20,10:20]
# raw_file = np.clip(np.float32(raw_file)-200, 0, 4095).astype("uint16")
# print("before WB", np.amax(raw_file), np.amin(raw_file))
# print(raw_file)
# raw_file = utils.white_balance(raw_file, 320/256, 740/256, 256/256)
# print(np.amax(raw_file), np.amin(raw_file))
# print(raw_file)

# img = utils.demosaic_raw(raw_file, "RGGB")
# img = img/np.amax(img)
# img = ((img**(1/2.2))*255).astype("uint8")

# print(np.amax(raw_file), np.amin(raw_file))

# plt.imsave(str(Path(raw_path).parent)+"Input_img.png", img) 
# dpc = Takam.DPC(raw_file, (100,100))
# corr_img = dpc.execute()
# mask = dpc.mask
# print(np.count_nonzero(mask))

"""defective_img, original_val = DPC_2012.introduce_defect(raw_file)
print(np.count_nonzero(original_val))
save_files = [(defective_img,  str(Path(raw_path).parent) + "/Defective_" + raw_filename + ".raw"), 
                  (original_val,   str(Path(raw_path).parent) + "/GT_DPC_" + raw_filename + ".raw" )]
for img, path in save_files:
    with open(path, "wb") as file:
        img.astype("uint16").tofile(file)

plt.imsave(str(Path(raw_path).parent) + "/Defective_" + raw_filename + ".png", defective_img)
print("Defective Image saved!")"""

# def_img  = np.fromfile(str(Path(raw_path).parent) + "/Defective_" + raw_filename + ".raw", dtype="uint16").reshape((1084,1924)) 

# img_GT   = np.fromfile(str(Path(raw_path).parent) + "/GT_DPC_" + raw_filename + ".raw", dtype="uint16").reshape((1084,1924)) 

# # img_patch = def_img[0:10, 0:10]
# print(def_img.shape)
# # print(def_img)
# dpc = Takam.DPC(def_img, (size))
# corr_img = dpc.execute()
# print(corr_img)
# corr_mask = dpc.mask
# print(np.count_nonzero(corr_mask))
# print(np.count_nonzero(img_GT))
# Takam.Evaluation(img_GT, corr_mask)
# plt.imsave("/home/user3/infinite-isp/out_frames/" + "/corrected_" + raw_filename + ".png", corr_img)


##############################################

# test on synthetic patch of size 100
# def_patch = np.full((10,10), fill_value=1001).astype("uint16")
# def_patch[5,5]=4000
# for i in range(0,def_patch.shape[1],2):
#     for j in range(0, def_patch.shape[0], 2):
#         def_patch[j,:] = 1010

#         def_patch[:,i] = 1005

# print(def_patch.shape)
# print(def_patch)


# GT_patch  = np.zeros((10,10))
# GT_patch[5,5]=4000

# dpc = Takam.DPC(def_patch, (10,10))
# corr_img = dpc.execute()
# # print(corr_img)
# corr_mask = dpc.mask
# print(corr_mask.dtype)
# print(GT_patch)
# Takam.Evaluation(GT_patch, corr_mask)
############################################## 
from utils import gamma, white_balance, demosaic_raw
import DPC_open_isp as openisp

img_path  = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/ISO100 - ISO1000/Raw input/Defective_100_ISO800_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407205358_BNR_OFF.raw" 
mask_path = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/ISO100 - ISO1000/openISP/corrected masks/DPC_mask_openISP_imp_1th_80_Defective_100_ISO800_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407205358_BNR_OFF.raw"
GT_path   = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/ISO100 - ISO1000/Raw GT/GT_ISO800_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407205358_BNR_OFF.raw"
save_path = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/FPs_Defective_100_ISO800_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407205358_BNR_OFF.csv"
size = (1536, 2592) #2592x1536

img_array = np.fromfile(img_path, dtype= "uint16").reshape(size)
GT_array   = np.fromfile(GT_path, dtype="uint16").reshape(size)
mask_array = np.fromfile(mask_path, dtype= "uint16").reshape(size)

# print(np.nonzero(GT_array))
print(np.amax(np.nonzero(GT_array)), np.amin(np.nonzero(GT_array)))


def_img = np.clip(np.float32(img_array)-200, 0, 4095).astype("uint16")
print(np.amax(def_img), np.amin(def_img))

# save_img = gamma(demosaic_raw(white_balance(def_img.copy(), 320/256, 740/256, 256/256), "RGGB"))
# plt.imsave("/home/user3/Desktop/testing_display_input.png", save_img)
img_padded = np.pad(def_img, (2,2), "reflect")
print(img_padded[31,1889], img_padded[33, 1891], img_padded[35,1893])


dpc        = openisp.DPC(def_img, size, 80) 
corr_img   = dpc.execute() 
corr_mask  = dpc.mask
print(sum(corr_img.ravel()==3894))
print(np.amax(corr_img))
img_padded = np.pad(corr_img, (2,2), "reflect")
print(img_padded[31,1889], img_padded[33, 1891], img_padded[35,1893])
exit()

print(np.amax(corr_img), np.amin(corr_img))
save_img = gamma(demosaic_raw(white_balance(def_img.copy(), 320/256, 740/256, 256/256), "RGGB"))
plt.imsave("/home/user3/Desktop/DPC_output.png", save_img)
print("img saved")






#==============================================================================
# tested bug in open ISP
"""test_img = np.array([[1,2,3], [4,5,6], [7,8,9]],dtype=object)
test_img = np.pad(test_img, (3,3))
print(test_img.shape)
test_img[0:2, :] = "side"
test_img[:, 0:2] = "side"
test_img[-2:, :] = "side"
test_img[:, -2:] = "side"
test_img[3,3]  = 100
# print(test_img)
for y in range(0,test_img.shape[0] - 2):
    for x in range(0,test_img.shape[1] - 2):
        p0 = test_img[y + 1, x + 1]
        p1 = test_img[y, x]
        p2 = test_img[y, x + 1]
        p3 = test_img[y, x + 2]
        p4 = test_img[y + 1, x]
        p5 = test_img[y + 1, x + 2]
        p6 = test_img[y + 2, x]
        p7 = test_img[y + 2, x + 1]
        p8 = test_img[y + 2, x + 2]
        
        
        if test_img[y+1,x+1]==100:
            print(p1+p8/2, p1, p8)
            test_img[y + 1, x + 1] = p1+p8/2 
        #     # test_img[y, x] = (y,x)
print(test_img)"""
# print(np.where(test_img=="p1"))
# print(range(test_img.shape[0] - 4))
# print(range(test_img.shape[0] - 4))
#==============================================================================

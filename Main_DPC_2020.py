from ast import Gt
import os
import DPC_open_isp as openisp
import DPD_R_2020 as Takam_R
import Takam_imp 
import DPC_Takam_2020 as Takam
import DPC_Yongji_2020 as yongji
import DPC_yongji_diagonal_info_2020 as yongji_improved
from utils import demosaic_raw, white_balance, introduce_defect, Evaluation, gamma, Results
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# Define variables
main_path       = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/"
folders         = ["Raw input"]#["ISO100", "ISO800", "ISO1600", "ISO2500", "ISO3500", "ISO4000", "ISO5000", "ISO6500", "ISO17000", "scene"] 
paths           = [] 

# get all file paths
for folder in folders:
    path = main_path + folder + "/" 
    paths.extend([path + file for file in os.listdir(path) if ".raw"==file[-4:]])
print("total images: ", len(paths))

# result obj to compile results
result = Results()
for raw_path in paths:    
    raw_filename    = Path(raw_path).stem.split(".")[0]
    out_img_path    = main_path +"Takam/corrected images/DPC_Output_Takam_imp_" + raw_filename +".png"
    out_mask_path   = main_path +"Takam/corrected masks/DPC_mask_Takam_imp_" + raw_filename +".raw"
    GT_path         = main_path + "Raw GT/GT_" + ("_").join(raw_filename.split("_")[2:]) + ".raw"
    org_img_path    = main_path + "Undefected Raw Input/" + ("_").join(raw_filename.split("_")[2:]) + ".raw"
    size = (1536, 2592)                       #(height, width)

    # Flags
    add_defect = False
    run_DPC    = True
    evaluate   = True


    # Read the raw image
    print("Reading raw file {}...".format(paths.index(raw_path)))
    raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)      # Construct an array from data in a text or binary file.
    print(raw_filename)

    # Generate a defective image
    if add_defect:
        defective_img, original_val = introduce_defect(raw_file, 100, padding=False)
        # print(np.nonzero(original_val))
        print("Total defect pixels: ", np.count_nonzero(original_val))
        
        # Save the defective image, mask and ground truth values for the defective pixels as binary files.
        save_files = [(defective_img,  main_path +"Raw input/Defective_100_" + raw_path.split("/")[-2] + "_" + raw_filename + ".raw"), 
                    (original_val,   main_path +"Raw GT/GT_" + raw_path.split("/")[-2] +"_"+ raw_filename + ".raw" )]
        for img, path in save_files:
            with open(path, "wb") as file:
                img.astype("uint16").tofile(file)
        
        # Save the defective image as .png image  (BLC --> WB --> Demsaic --> Gamma correction --> save)
        def_blc = np.clip(np.float32(defective_img)-200, 0, 4095).astype("uint16")
        save_as  = gamma(demosaic_raw(white_balance(def_blc.copy(), 320/256, 740/256, 256/256), "RGGB"))
        plt.imsave(main_path +"Input images/"+ raw_filename + "_" + raw_path.split("/")[-2] +"_20DP_Def_input.png", save_as)
    
        print("Defective Image saved!")

    # Read the defective image
    if run_DPC:
        def_img = np.clip(np.float32(raw_file)-200, 0, 4095).astype("uint16")    # BLC
        
        # convert to 3 channel image before saving
        # save_img = gamma(demosaic_raw(white_balance(def_img.copy(), 320/256, 740/256, 256/256), "RGGB")) 
        # plt.imsave(str(Path(raw_path).parent)+ "/Input_images/" + raw_filename + "_Input_image_.png",save_img )
        print(def_img.shape)
        print(np.amax(def_img), np.amin(def_img))

        dpc        = Takam_imp.DPC(def_img, size, np.amin(def_img), np.amax(def_img),20) 
        corr_img   = dpc.execute() 
        corr_mask  = dpc.mask
        print(np.count_nonzero(corr_mask[0:2,:]))
        print(np.count_nonzero(corr_mask[-2:,:]))
        print(np.count_nonzero(corr_mask[:,0:2]))
        print(np.count_nonzero(corr_mask[:,-2:]))

        print(corr_img.shape, corr_mask.shape)
        print(np.count_nonzero(corr_mask))

        # Save the corrected 3 channel image after white balancing
        save_corr_img = gamma(demosaic_raw(white_balance(corr_img.copy(), 320/256, 740/256, 256/256), "RGGB"))
        plt.imsave(out_img_path, save_corr_img)
        with open(out_mask_path, "wb") as file:
            corr_mask.astype("uint16").tofile(file)
        print("file saved")
    
    # Evaluate DPC
    if evaluate:
        
        img_GT  = np.fromfile(org_img_path, dtype="uint16").reshape(size)
        mask_GT = np.fromfile(GT_path, dtype="uint16").reshape(size)   
        print(raw_filename, "\n being compared with\n", str((Path(org_img_path).stem)))
        print(np.count_nonzero(mask_GT), np.count_nonzero(corr_mask))
        confusion_matrix  = Evaluation(img_GT, corr_img, mask_GT, corr_mask)
        confusion_matrix.insert(0, raw_filename)
        result.add_row(confusion_matrix)

result.save_csv(main_path, "Takam_imp(min,max,20)_results")
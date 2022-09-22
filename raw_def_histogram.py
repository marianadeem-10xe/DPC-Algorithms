import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

main_path       = "/home/user3/Desktop/Maria Nadeem/Infinite-ISP/Defect Pixel Detection and Correction/DPC_dataset/Threshold tuning/"
folders         = ["Raw input"]#["ISO100", "ISO800", "ISO1600", "ISO2500", "ISO3500", "ISO4000", "ISO5000", "ISO6500", "ISO17000", "scene"] 
save_path       = main_path + "Histograms/"    
paths           = [] 
size = (1536, 2592) 

# get all file paths
for folder in folders:
    path = main_path + folder + "/" 
    paths.extend([path + file for file in os.listdir(path) if ".raw"==file[-4:]])
print("total images: ", len(paths))

for raw_path in paths:
    raw_filename    = Path(raw_path).stem.split(".")[0]
    print("Reading raw file {}...".format(paths.index(raw_path)))
    raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)      # Construct an array from data in a text or binary file.
    print(raw_filename)
    
    stucklow = (raw_file.ravel()<16)
    stuckhigh = (raw_file.ravel()>4080)
    
    img_low = raw_file.ravel()[stucklow]
    img_high = raw_file.ravel()[stuckhigh]
    
    fig = plt.figure()
    histogram = plt.hist([img_low, img_high] , histtype= "bar")
    print("Total DPs: ", np.int32(np.sum(histogram[0])))
    print("Stuck low :", histogram[0][0][0])
    print("Stuck high :", histogram[0][-1][-1])
    # plt.show()
    plt.savefig(main_path + "Histograms/DPs only/hist_DPs_"+ raw_filename + ".png")
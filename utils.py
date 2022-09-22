from operator import index
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import colour_demosaicing as cd

def introduce_defect(img, total_defective_pixels, padding):
    
    """Randomly replaces pixels values with extremely high or low pixel values to create dead pixels.
     Note that the defect is never introduced on the periphery of the image. 
     The function returns defective image: image containing 0.5% dead pixels, mask: bool array with 1 
     indicating dead pixels, orig_val: containing original values of the dead pixels introduced."""
    
    if padding:
        padded_img = np.pad(img, ((2,2), (2,2)), "symmetric")
    else:
        padded_img  = img.copy()
   
    orig_val   = np.zeros((padded_img.shape[0], padded_img.shape[1])) 
    
    while total_defective_pixels:
        defect     = [random.randrange(1,15), random.randrange(4081, 4095)]   # stuck low int b/w 1 and 15, stuck high float b/w 4081 and 4095
        defect_val = defect[random.randint(0,1)] 
        random_row, random_col   = random.randint(2, img.shape[0]-3), random.randint(2, img.shape[1]-3)
        left, right  = orig_val[random_row, random_col-2], orig_val[random_row, random_col+2]
        top, bottom  = orig_val[random_row-2, random_col], orig_val[random_row+2, random_col]
        neighbours   = [left, right, top, bottom]
        
        if not any(neighbours) and orig_val[random_row, random_col]==0: # if all neighbouring values in orig_val are 0 and the pixel itself is not defective
            orig_val[random_row, random_col]   = padded_img[random_row, random_col]
            padded_img[random_row, random_col] = defect_val
            total_defective_pixels-=1
    
    return padded_img, orig_val 

#########################################################################################################

def Evaluation(true_img, pred_img, true_mask, pred_mask):
    assert np.size(true_mask)==np.size(pred_mask), "Sizes of the input images must match."
    
    # Compute error
    error = np.round((np.add(true_img.astype("float32"), -pred_img.astype("float32"))**2).sum()/true_mask.size, 4)
    pred_mask[pred_mask>0] = 1
    true_mask[true_mask>0] = 1

    conf_mat     = confusion_matrix(true_mask.ravel(), pred_mask.ravel())
    # pos_pred_val = round(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1]), 4)    # TP/(TP+FP)
    
    print("---------------------")
    print("Error: ", error)
    print("---------------------")
    print("True positives: ",  conf_mat[1,1])
    print("False positives: ", conf_mat[0,1])
    print("True negatives: ",  conf_mat[0,0])
    print("False negatives: ", conf_mat[1,0])
    print("---------------------")
    conf_mat = conf_mat.ravel().tolist()
    conf_mat.append(error)                   #extend([error, pos_pred_val])
    return conf_mat

#######################################################################################################

class Results:
    def __init__(self):
        self.confusion_pd = pd.DataFrame(np.zeros((1,6)), columns=["Filename", "TN", "FP","FN", "TP", "MSE"])
    
    def add_row(self,row):
        self.confusion_pd = pd.concat([self.confusion_pd, pd.DataFrame(np.array(row).reshape(1,6), columns=["Filename", "TN", "FP","FN", "TP", "MSE"])], ignore_index=False)
    
    def save_csv(self, path, filename):
        self.confusion_pd.to_csv(path + "/" +filename + ".csv", index=False)

#########################################################################################################

def demosaic_raw(img, bayer):

        bpp = 12
        img = np.float32(img) / np.power(2, bpp)
        hs_raw = np.uint8(img*255)
        img = np.uint8(cd.demosaicing_CFA_Bayer_bilinear(hs_raw, bayer))
        return img

def get_color(x, y):
    is_even = lambda x: x%2==0

    if is_even(x) and is_even(y):
        return "R"
    elif not(is_even(x)) and not(is_even(y)):
        return "B"
    else:
        return "G"


def white_balance(img, R_gain, B_gain, G_gain):
    img = img.astype("float32")
    for x in range(img.shape[1]): #col
        for y in range(img.shape[0]):   #row
            color = get_color(x, y)
            if color=="R":
                img[y][x] *= R_gain 
            elif color=="G":
                img[y][x] *= G_gain 
            else:
                img[y][x] *= B_gain
    img = np.clip(img, 0, 4095)
    img = ((img/np.amax(img))*(4095)).astype("uint16")
    return img 

def gamma(img):
    img = img/np.max(img)
    img = (img**(1/2.2))*255
    return img.astype("uint8")    
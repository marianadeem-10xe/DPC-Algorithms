import random
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
#########################################################################################################

# Function to introduce dead pixels randomly
def introduce_defect(img):
    
    """Randomly replaces pixels values with extremely high or low pixel values to create dead pixels.
     Note that the defect is never introduced on the periphery of the image. 
     The function returns defective image: image containing 0.5% dead pixels, mask: bool array with 1 
     indicating dead pixels, orig_val: containing original values of the dead pixels introduced."""
    
    total_defective_pixels = 5000 #int(img.ravel().shape[0]*0.5/100)
    
    padded_img = np.pad(img, ((2,2)), "symmetric")
    orig_val   = np.zeros((padded_img.shape[0], padded_img.shape[1])) 
    
    while total_defective_pixels:
        defect     = [random.randrange(1,15), random.randrange(4081, 4095)]   # stuck low int b/w 1 and 15, stuck high float b/w 4081 and 4095
        defect_val = defect[random.randint(0,1)] 
        random_row, random_col   = random.randint(2, img.shape[0]-3), random.randint(2, img.shape[1]-3)
        left, right  = orig_val[random_row, random_col-2], orig_val[random_row, random_col+2]
        top, bottom  = orig_val[random_row-2, random_col], orig_val[random_row+2, random_col]
        neighbours   = [left, right, top, bottom]
        
        if not any(neighbours) and orig_val[random_row, random_col]==0: # if all neighbouring values in orig_val are 0 and the pixel itself is not defetive
            orig_val[random_row, random_col]   = padded_img[random_row, random_col]
            padded_img[random_row, random_col] = defect_val
            total_defective_pixels-=1
    
    return padded_img, orig_val 

#########################################################################################################

def Evaluation(true_mask, pred_mask):
    assert np.size(true_mask)==np.size(pred_mask), "Sizes of the input images must match."
    
    # Compute error
    error = (np.add(true_mask, -pred_mask)**2).sum()/true_mask.size
    pred_mask[pred_mask>0] = 1
    true_mask[true_mask>0] = 1

    conf_mat = confusion_matrix(true_mask.ravel(), pred_mask.ravel())
    print("---------------------")
    print("Error: ", np.round(error, 4))
    print("---------------------")
    print("True positives: ",  conf_mat[1,1])
    print("False positives: ", conf_mat[0,1])
    print("True negatives: ",  conf_mat[0,0])
    print("False negatives: ", conf_mat[1,0])
    print("---------------------")
    confusion_pd = pd.DataFrame(conf_mat.reshape(1,4), columns=["TN", "FP","FN", "TP"])
    confusion_pd.to_csv('/home/user3/infinite-isp/out_frames/Results.csv', index=False)

#########################################################################################################

# Define Class DPC

class DPC:
    def __init__(self, img, size):
        self.img = img
        self.height = size[0]
        self.width = size[1]
        self.threshold = 20 

    def execute(self):
        
        """Replace the dead pixel value with corrected pixel value and returns 
        the corrected image."""
        
        self.mask = self.detect_dead_pixels()
        for i in range(self.mask.shape[0]):        #column
            for j in range(self.mask.shape[1]):    #row 
                if self.mask[i,j]!=0:
                    self.img[i,j] = self.mask[i,j]
        return self.img

    def DPD_M(self, i, j):
        """This function applies a median fileter to check whether the pixel at the given
        coordinates is defective or not. Returns True if defective else False."""

        
        to_be_tested_pv = self.img[i,j]
        left, right, top, bottom = self.img[i,j-2], self.img[i,j+2], self.img[i-2,j], self.img[i+2,j]
        d1, d2, d3, d4           = self.img[i-2,j-2], self.img[i-2,j+2], self.img[i+2,j-2], self.img[i+2,j+2]   
        neighbours               = [left, right, top, bottom, d1, d2, d3, d4]
        neighbours.sort()
        PH, PL, P_med            = neighbours[-1], neighbours[0], np.median(np.array(neighbours))

        if not(PH < to_be_tested_pv < PL):
        
            diff     = abs(to_be_tested_pv-P_med)
            thresh_1 = abs((to_be_tested_pv+P_med)/2)
            thresh_2 = abs(((to_be_tested_pv+P_med)/2) - 4095)

            if diff > thresh_1 or diff > thresh_2:
                return True
        
        return False    

    def detect_dead_pixels(self):
        
        """Generates a masked array with zero for good pixels and corrected value
         (avg of left and right pixel) for the dead pixels."""
        
        mask = np.zeros((self.height, self.width)).astype("uint16")
    
        for i in range(2, self.height-6):        #column
            for j in range(2, self.width-6):     #row
                if mask[i,j]==0:
                    coordinates = [[i,j], [i,j+2], [i,j+4], 
                                   [i+2,j], [i+2,j+2], [i+2,j+4], 
                                   [i+4,j], [i+4,j+2], [i+4,j+4]]
                    
                    block      = [self.img[x][y] for x, y in coordinates]
                    sorted_block  = np.argsort(block)
                    # print("sblock:", sorted_block)
                    block.sort()
                    # print("block:", block)
                    PH, PL, P_median = max(block), min(block), np.median(block)
                    R, IQR = PH-PL, (block[7]+block[6]-block[1]-block[2])/2 
                    if R > self.threshold or R > 2*IQR:
                        q_inf = block[2]-block[0]
                        q_med = block[5]-block[3]
                        q_sup = block[8]-block[6]
                        

                        if q_inf > q_med or q_sup > q_med:
                            if q_inf > q_med:   # min val is hypothetically defective
                                idx, fn =  0, lambda x: x+1 
                            else:               # max val is hypothetically defective
                                idx, fn = -1, lambda x: x-1
                            x, y = coordinates[sorted_block[idx]][0], coordinates[sorted_block[idx]][1]
                            status = self.DPD_M(x,y)
                            while status:
                                # print("entered while")
                                idx = fn(idx)
                                mask[x,y] = P_median
                                adj_coord = coordinates[sorted_block[idx]]
                                E_c = abs(int(self.img[x][y]) - int(self.img[adj_coord[0], adj_coord[1]]))
                                if E_c < (R/(2*3-1)):        # adj pixel is hypothetically defective
                                    x, y      = adj_coord[0], adj_coord[1]
                                    status    = self.DPD_M(x, y)
                                else:
                                    status = False
                    
        return mask        

#########################################################################################################
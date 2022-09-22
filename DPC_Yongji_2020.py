import numpy as np

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

    
    def detect_dead_pixels(self):
        
        """Generates a masked array with zero for good pixels and corrected value
         (avg of left and right pixel) for the dead pixels."""
        
        mask = np.zeros((self.height, self.width)).astype("uint16")
        gh_l, gv_l, diff_l = [],[], []
        for i in range(2, self.height-2):        #column
            for j in range(2, self.width-2):     #row
                to_be_tested_pv = self.img[i,j]
                left, right, top, bottom = self.img[i,j-2], self.img[i,j+2], self.img[i-2,j], self.img[i+2,j]
                d1, d2, d3, d4           = self.img[i-2,j-2], self.img[i-2,j+2], self.img[i+2,j-2], self.img[i+2,j+2]   
                neighbours               = [left, right, top, bottom, d1, d2, d3, d4]
                neighbours.sort()
                PM, PH, PL, PN           = neighbours[-1], neighbours[-2], neighbours[1], neighbours[0]        
                
                diff = PH-PL; diff_l.append(diff) 
                avg  = (sum(neighbours)-(sum([PM,PH,PL,PN])))/4
                # print(left, right, top, bottom, d1, d2, d3, d4)
                if not ((avg-diff) < to_be_tested_pv < (avg+diff)):
                    # print("DP found at p: ", to_be_tested_pv)
                    # print(est)
                    GH = abs(int(d1)-int(top))  + abs(int(d2)-int(top))   + abs(int(d3)-int(bottom)) + abs(int(d4)-int(bottom))
                    GV = abs(int(d1)-int(left)) + abs(int(d2)-int(right)) + abs(int(d3)-int(left))   + abs(int(d4)-int(right))
                    
                    gh_l.append(GH);gv_l.append(GV)
                    # print("gH, gV", GH,GV)
                    if GH < self.threshold and GH < GV - self.threshold:
                        mask[i,j] = (left+right)/2
                    elif GV < self.threshold and GV < GH - self.threshold:
                        mask[i,j] = (top+bottom)/2
                    else:                        
                        mask[i,j] = (left+right+top+bottom)/4       
                        # mask[i,j] = np.median(np.array([left, right, top, bottom])).astype("uint16")       
        # print("GH max, min:", max(gh_l), min(gh_l))
        # print("GV max, min:", max(gv_l), min(gv_l))
        # print("diff max, min:", max(diff_l), min(diff_l))
        return mask        

#########################################################################################################

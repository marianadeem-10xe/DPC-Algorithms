import numpy as np
#########################################################################################################

# Define Class DPC

class DPC:
    def __init__(self, img, size, min_th, max_th, grad_threshold):
        self.img       = img
        self.height    = size[0]
        self.width     = size[1]
        self.stucklow  = min_th
        self.stuckhigh = max_th
        self.grad_th   = grad_threshold

    def padding(self):
        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

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
        # self.img = self.padding()
        mask = np.zeros((self.img.shape[0], self.img.shape[1])).astype("uint16")
        for i in range(2, self.img.shape[0]-2):        #column
            for j in range(2, self.img.shape[1]-2):     #row
                to_be_tested_pv = self.img[i,j]
                left, right, top, bottom = self.img[i,j-2], self.img[i,j+2], self.img[i-2,j], self.img[i+2,j]
                d1, d2, d3, d4           = self.img[i-2,j-2], self.img[i-2,j+2], self.img[i+2,j-2], self.img[i+2,j+2]   
                neighbours               = [int(left), int(right), int(top), int(bottom), int(d1), int(d2), int(d3), int(d4)]
                neighbours.sort()
                PH, PL, P_med            = neighbours[-1], neighbours[0], np.median(np.array(neighbours))

                if not(PH < to_be_tested_pv < PL):
                
                    diff     = abs(int(to_be_tested_pv)-P_med)
                    thresh_1 = abs((int(to_be_tested_pv)+P_med)/2 - self.stucklow)
                    thresh_2 = abs(((int(to_be_tested_pv)+P_med)/2) - self.stuckhigh)

                    if diff > thresh_1 or diff > thresh_2:
                        # Compute gradients
                        GH = abs(int(d1)-int(top))  + abs(int(d2)-int(top))   + abs(int(d3)-int(bottom)) + abs(int(d4)-int(bottom))
                        GV = abs(int(d1)-int(left)) + abs(int(d2)-int(right)) + abs(int(d3)-int(left))   + abs(int(d4)-int(right))
                        Dia_l = abs(int(d1)-int(d4))
                        Dia_r = abs(int(d2)-int(d3))
                        
                        if (GV < self.grad_th) or (GH < self.grad_th) or (Dia_l < self.grad_th) or (Dia_r < self.grad_th):
                            # print("entered")
                            if GV==min([GV, GH, Dia_l, Dia_r]):
                                mask[i,j] = (top+bottom)/2
                            elif GH==min([GV, GH, Dia_l, Dia_r]):
                                mask[i,j] = (left+right)/2
                            elif Dia_l==min([GV, GH, Dia_l, Dia_r]): 
                                mask[i,j] =  (d1+d4)/2
                            elif Dia_r==min([GV, GH, Dia_l, Dia_r]):
                                mask[i,j] = (d2+d3)/2    
                        else:                        
                            mask[i,j] = (left+right+top+bottom)/4
                        # mask[i,j] =  np.median(np.array(neighbours))
        return mask        

#########################################################################################################
             

            
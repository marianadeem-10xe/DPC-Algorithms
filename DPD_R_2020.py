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

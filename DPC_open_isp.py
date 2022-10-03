import numpy as np

#########################################################################################################

# Define Class DPC

class DPC:
    def __init__(self, img, size, threshold):
        self.img = img
        self.height = size[0]
        self.width = size[1]
        self.threshold = threshold
        self.mode      = "gradient" 
    
    def execute(self):
        
        """Replace the dead pixel value with corrected pixel value and returns 
        the corrected image."""
        
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1])).astype("uint16")
        dpc_img = np.empty((self.height, self.width), np.uint16)
        for y in range(self.img.shape[0] - 4):
            for x in range(self.img.shape[1] - 4):
                p0 = self.img[y + 2, x + 2]
                p1 = self.img[y, x]
                p2 = self.img[y, x + 2]
                p3 = self.img[y, x + 4]
                p4 = self.img[y + 2, x]
                p5 = self.img[y + 2, x + 4]
                p6 = self.img[y + 4, x]
                p7 = self.img[y + 4, x + 2]
                p8 = self.img[y + 4, x + 4]
                
                if not(min([p1, p2, p3, p4, p5, p6, p7, p8]) < p0 < max([p1, p2, p3, p4, p5, p6, p7, p8])):
                    if (abs(int(p1) - int(p0)) > self.threshold) and (abs(int(p2) - int(p0)) > self.threshold) and (abs(int(p3) - int(p0)) > self.threshold) \
                            and (abs(int(p4) - int(p0)) > self.threshold) and (abs(int(p5) - int(p0)) > self.threshold) and (abs(int(p6) - int(p0)) > self.threshold) \
                            and (abs(int(p7) - int(p0)) > self.threshold) and (abs(int(p8) - int(p0)) > self.threshold):
                        if self.mode == 'mean':
                            p0 = (p2 + p4 + p5 + p7) / 4
                        elif self.mode == 'gradient':
                            
                            dv = abs(2 * int(p0) - int(p2) - int(p7))
                            dh = abs(2 * int(p0) - int(p4) - int(p5))
                            ddl = abs(2 * int(p0) - int(p1) - int(p8))
                            ddr = abs(2 * int(p0) - int(p3) - int(p6))
                            
                            if (min(dv, dh, ddl, ddr) == dv):
                                p0 = (p2 + p7) / 2
                            elif (min(dv, dh, ddl, ddr) == dh):
                                p0 = (p4 + p5) / 2
                            elif (min(dv, dh, ddl, ddr) == ddl):
                                p0 = (p1 + p8) / 2
                            else:
                                p0 = (p3 + p6) / 2
                
                dpc_img[y, x] = p0
                if self.img[y + 2, x + 2]!=p0:
                    self.mask[y + 2, x + 2] = p0
        self.img = dpc_img
        return self.img       

#########################################################################################################
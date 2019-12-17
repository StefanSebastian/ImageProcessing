import numpy as np 
import cv2
import glob
import pickle 

# config 
GRIDX = 8
GRIDY = 8


db = [cv2.imread(file) for file in glob.glob('./db2/*.jpg')]
db_g = []
idx = 0
for image in db:
    print("Grayscale for image : " + str(idx))
    idx += 1
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    db_g.append(image_g)

print("Computing LBP")
def compute_lbp(image):
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1,), (1, 0), (1, -1), (0, -1)]
    new_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            bin_str = ""
            for (dx, dy) in dirs:
                neighbx = i + dx 
                neighby = j + dy
                if image[neighbx][neighby] >= image[i][j]:
                    bin_str = bin_str + "1"
                else: # don't append zeroes at the beginning
                    bin_str = bin_str + "0"               
            new_img[i][j] = int(bin_str, 2)
    
    return new_img

db_lbp = []
idx = 0
for image in db_g:
    print("LBP for image : " + str(idx))
    idx += 1
    image_lbp = compute_lbp(image)
    db_lbp.append(image_lbp)

print("Creating histograms")
def create_histogram(image):
    h, w = image.shape[:2]
    gridh = int(h / GRIDX)
    gridw = int(w / GRIDY)
    hist = []

    for gx in range(GRIDX):
        for gy in range(GRIDY):
            local_hist = [0] * 256

            for x in range(gx * gridh, (gx + 1) * gridh):
                for y in range(gy * gridw, (gy + 1) * gridw):
                    if x < h and y < w:
                        local_hist[image[x][y]] += 1
            hist += local_hist
    return hist

db_hist = []
idx = 0
for image in db_lbp:
    print("Histogram for : " + str(idx))
    image_hist = create_histogram(image)
    db_hist.append(image_hist)

print("Storing histograms")
with open('db_hist', 'wb') as fp:
    pickle.dump(db_hist, fp)

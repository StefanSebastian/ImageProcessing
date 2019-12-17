import numpy as np 
import cv2
import glob

# config 
GRIDX = 8
GRIDY = 8

# read and prepare data
print("Reading data")
test_im = cv2.imread('./test/image_0071.jpg')
test_im_g = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)

db = [cv2.imread(file) for file in glob.glob('./db/*.jpg')]
db_g = []
for image in db:
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

test_im_lbp = compute_lbp(test_im_g)
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

test_hist = create_histogram(test_im_lbp)
db_hist = []
for image in db_lbp:
    image_hist = create_histogram(image)
    db_hist.append(image_hist)

print("Searching db for similarities")
def euclidean_dist(hist1, hist2):
    dist = 0
    for i in range(len(hist1)):
        dist += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i])
    return dist

distances = []
for image_hist in db_hist:
    dist = euclidean_dist(test_hist, image_hist)
    distances.append(dist)


# now sort db images by distance ; lower distance is preferred
most_similar = [im for (d, im) in sorted(zip(distances, db), key=lambda pair: pair[0])]

# display results
cv2.imshow("test image", test_im)
cv2.imshow("lbp", test_im_lbp)
for i in range(3):
    im = most_similar[i]
    cv2.imshow("im_" + str(i), im)

cv2.waitKey(0)
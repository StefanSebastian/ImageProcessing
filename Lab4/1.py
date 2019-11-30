import glob
import cv2

images = [cv2.imread(file) for file in glob.glob("./dataset/*.jpg")]

test_im = cv2.imread('./test_im/4.jpg')
cv2.imshow("test_image", test_im)

# convert to HSV
images_hsv = []
for image in images:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images_hsv.append(image_hsv)
test_im_hsv = cv2.cvtColor(test_im, cv2.COLOR_BGR2HSV)

# create bins for hue and saturation
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]

# value ranges
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # list concat

channels = [0, 1] # corresponding to HS

# compute the histograms
dataset_hist = []
for image in images_hsv:
    hist = cv2.calcHist([image], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    dataset_hist.append(hist)

test_im_hist = cv2.calcHist([test_im_hsv], channels, None, histSize, ranges, accumulate=False)
cv2.normalize(test_im_hist, test_im_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# apply compare methods
'''
Correlation ( CV_COMP_CORREL )
Chi-Square ( CV_COMP_CHISQR )
Intersection ( method=CV_COMP_INTERSECT )
Bhattacharyya distance ( CV_COMP_BHATTACHARYYA )
'''
for compare_method in range(4):
    comparisons = []
    for hist in dataset_hist:
        comp_hist = cv2.compareHist(test_im_hist, hist, compare_method)
        comparisons.append(comp_hist)

    if compare_method == 1 or compare_method == 3:
        should_rev = False    
    else:
        should_rev = True
    most_similar = [im for (c, im) in sorted(zip(comparisons, images), key=lambda pair: pair[0], reverse=should_rev)]
    
    for i in range(3):
        im = most_similar[i]
        cv2.imshow("comparison_" + str(compare_method) + "_im_" + str(i), im)

    cv2.waitKey(0)

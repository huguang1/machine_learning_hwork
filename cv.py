from sklearn.datasets import load_sample_image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img_rgb = load_sample_image('flower.jpg')
img_bgr = np.flip(img_rgb, axis=-1)
# plt.imshow(img_rgb)
# plt.show()


tuple_bgr = cv.split(img_bgr)
histSize = 256    # number of bins
histRange = (0, 256)

b_hist = cv.calcHist(tuple_bgr, [0], None, [histSize], histRange, accumulate=False)
g_hist = cv.calcHist(tuple_bgr, [1], None, [histSize], histRange, accumulate=False)
r_hist = cv.calcHist(tuple_bgr, [2], None, [histSize], histRange, accumulate=False)
# plt.plot(b_hist, label='blue', color='blue')
# plt.plot(g_hist, label='green', color='green')
# plt.plot(r_hist, label='red', color='red')
# plt.legend()
# plt.show()

r_hist[r_hist > 5000] = 5000
# plt.plot(b_hist, label = 'blue', color = 'blue')
# plt.plot(g_hist, label = 'green', color = 'green')
# plt.plot(r_hist, label = 'red', color = 'red')
# plt.legend()
# plt.show()


def ConvertToSingleChannel(img, idx):
  new = np.zeros_like(img)
  new[:,:,idx] = img[:,:,idx]
  return new


img_onlyR = ConvertToSingleChannel(img_bgr, 2)
img_onlyB = ConvertToSingleChannel(img_bgr, 0)
img_onlyG = ConvertToSingleChannel(img_bgr, 1)
# img_onlyR = np.flip(img_onlyR, axis=-1)
# img_onlyB = np.flip(img_onlyB, axis=-1)
# img_onlyG = np.flip(img_onlyG, axis=-1)
# plt.imshow(img_onlyR)
# plt.show()
# plt.imshow(img_onlyB)
# plt.show()
# plt.imshow(img_onlyG)
# plt.show()

img_onlyR[img_onlyR < 100] = 0    # Setting a threshold to segment the flower
# img_onlyR = np.flip(img_onlyR, axis=-1)
# plt.imshow(img_onlyR)
# plt.show()
np.where(img_onlyR[:, :, 0])

zero_idx = np.where(img_onlyR[:, :, 2] == 0)

img_onlyB[zero_idx[0], zero_idx[1], 0] = 0   # Makes the same indices zero in B channel
img_onlyG[zero_idx[0], zero_idx[1], 1] = 0

img_segmented_BGR = img_onlyR+img_onlyB+img_onlyG   # Since they all have one different non-zero channel, we can add them together.
# img_segmented_BGR = np.flip(img_segmented_BGR, axis=-1)
# plt.imshow(img_segmented_BGR)
# plt.show()

img_gray = cv.cvtColor(img_segmented_BGR, cv.COLOR_BGR2GRAY)
_, img_thresh = cv.threshold(img_gray, 30, 255, 0)    # (Source image, threshold, mapped value (if >threshold), threshold_mode)
# plt.imshow(img_thresh, cmap='gray')
# plt.show()

contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

canvas = np.zeros(img_segmented_BGR.shape)
cv.drawContours(canvas, contours, -1, (0,255,0), 1)   # -1 to draw all contours
# plt.imshow(canvas, cmap='gray')
# plt.show()

# print(hierarchy.shape)
# print(hierarchy)
hierarchy = np.squeeze(hierarchy)
# print(hierarchy)

top = hierarchy[hierarchy[:, 3] == -1]
a = hierarchy[7, :]    # How do we know? It has many children

mask = np.zeros(img_gray.shape, np.uint8)
cv.drawContours(mask, contours[7], -1, (255, 255, 255), 1)
# plt.imshow(mask, cmap='gray')
# plt.show()


area_inside = np.empty(img_gray.shape, dtype=np.int8)
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        area_inside[i, j] = cv.pointPolygonTest(contours[7], (j, i), measureDist=False)      # Determines whether the point is inside a contour, outside, or lies on an edge

area_inside[area_inside == -1] = 0    # Get rid of -1 values
gray_inside = area_inside * 255

x, y = np.where(area_inside == 0)
img_segmented_BGR[x, y, :] = 0
# img_segmented_BGR = np.flip(img_segmented_BGR, axis=-1)
# plt.imshow(img_segmented_BGR)
# plt.show()

img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)
# plt.imshow(h)
# plt.show()
# plt.imshow(s)
# plt.show()
# plt.imshow(v)
# plt.show()

# h, s, v = cv.split(img_hsv)
# for i in range(5):
#   h = h.astype(int)   # Originally, type is uint8 - only takes values (0,255)
#   h = np.clip(h+i*40, 0, 180)
#   h = h.astype('uint8')
#   img_hsv_saturated = cv.merge((h,s,v))
#   img_bgr_saturated = cv.cvtColor(img_hsv_saturated, cv.COLOR_HSV2BGR)
#   img_bgr_saturated = np.flip(img_bgr_saturated, axis=-1)
#   plt.imshow(img_bgr_saturated)
#   plt.show()


# h, s, v = cv.split(img_hsv)
#
# for i in range(5):
#   s = s.astype(int)   # Originally, type is uint8 - only takes values (0,255)
#   s = np.clip(s+i*50, 0, 255)
#   s = s.astype('uint8')
#   img_hsv_saturated = cv.merge((h,s,v))
#   img_bgr_saturated = cv.cvtColor(img_hsv_saturated, cv.COLOR_HSV2BGR)
#   img_bgr_saturated = np.flip(img_bgr_saturated, axis=-1)
#   plt.imshow(img_bgr_saturated)
#   plt.show()


# h, s, v = cv.split(img_hsv)
#
# for i in range(5):
#   v = v.astype(int)   # Originally, type is uint8 - only takes values (0,255)
#   v = np.clip(v-i*50, 0, 255)
#   v = v.astype('uint8')
#   img_hsv_saturated = cv.merge((h,s,v))
#   img_bgr_saturated = cv.cvtColor(img_hsv_saturated, cv.COLOR_HSV2BGR)
#   img_bgr_saturated = np.flip(img_bgr_saturated, axis=-1)
#   plt.imshow(img_bgr_saturated)
#   plt.show()

img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
l, a, b = cv.split(img_lab)
# plt.imshow(l)
# plt.show()
# plt.imshow(a)
# plt.show()
# plt.imshow(b)
# plt.show()
img_onlyL = ConvertToSingleChannel(img_lab, 0)
img_onlyA = ConvertToSingleChannel(img_lab, 1)
img_onlyB = ConvertToSingleChannel(img_lab, 2)

img_onlyL = cv.cvtColor(img_onlyL, cv.COLOR_Lab2BGR)
img_onlyA = cv.cvtColor(img_onlyA, cv.COLOR_Lab2BGR)
img_onlyB = cv.cvtColor(img_onlyB, cv.COLOR_Lab2BGR)
# img_onlyL = np.flip(img_onlyL, axis=-1)
# img_onlyA = np.flip(img_onlyA, axis=-1)
# img_onlyB = np.flip(img_onlyB, axis=-1)
#
# plt.imshow(img_onlyL)
# plt.show()
# plt.imshow(img_onlyA)
# plt.show()
# plt.imshow(img_onlyB)
# plt.show()


kernel = np.ones((5, 5), np.float32)/25
# print(kernel)
dst = cv.filter2D(img_bgr, -1, kernel)
# img_bgr = np.flip(img_bgr, axis=-1)
# dst = np.flip(dst, axis=-1)
# plt.imshow(img_bgr)
# plt.show()
# plt.imshow(dst)
# plt.show()

# print(cv.getGaussianKernel(5, sigma=1))  # Returns coefficients in 1D

blur = cv.GaussianBlur(img_bgr, (5, 5), 0)
# blur = np.flip(blur, axis=-1)
# plt.imshow(blur)
# plt.show()


img=np.zeros((7,7))
img[3,3]=1
print(img)

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
print(dst)





































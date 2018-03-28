import cv2
from pylab import *
import copy as cp
from scipy import ndimage
import math


def SIFT(image1, n):
    sigma = 1.7
    k = 2 ** 0.5
    scales = 5
    octaves = 4
    base_image = np.zeros((shape(image1)))
    base_image[:] = image1
    image_octaveList = []
    image_baseList = []
    for i in range(octaves):
        image_scaleList = []
        for j in range(scales):
            if i == 0 and j == 0:
                temp1 = cp.deepcopy(base_image)
                image_scaleList.append(temp1)
            elif i > 0 and j == 0:
                temp2 = ndimage.zoom(image_baseList[i - 1][0], 0.5, order=1)
                temp3 = cp.deepcopy(temp2)
                image_scaleList.append(temp3)
        image_baseList.append(image_scaleList)
    '''for list in image_baseList :
        cv2.imshow("image",list[0])
        cv2.waitKey(0)'''
    for i in range(octaves):
        image_scaleList = []
        for j in range(scales):
            if j == 0:
                temp1 = np.zeros(np.shape(image_baseList[i][0]))
                temp1[:] = image_baseList[i][0]
            sigma = math.pow(k, j) * 1.7
            histogram_size = int(math.ceil(7 * sigma))
            histogram_size = 2 * histogram_size + 1
            temp2 = temp3 = np.zeros(np.shape(temp1))
            temp2 = cv2.GaussianBlur(temp1, (histogram_size, histogram_size), sigma, sigma)
            '''cv2.imshow("abc",temp2)
            cv2.waitKey(0)'''
            image_scaleList.append(temp2)
        image_octaveList.append(image_scaleList)

    DoG_List = []
    for i in range(octaves):
        image_scaleList = []
        for j in range(1, scales):
            difference = np.zeros(np.shape(image_octaveList[i][0]))
            difference[:] = np.subtract(image_octaveList[i][j], image_octaveList[i][j - 1])
            image_scaleList.append(difference)
            '''cv2.imshow("abc", difference)
            cv2.waitKey(0)'''
        DoG_List.append(image_scaleList)
    c1 = 0
    image_extremumList = []
    for i in range(octaves):
        image_scaleList = []
        for j in range(1, scales - 2):
            image_extremum = np.zeros(DoG_List[i][j].shape, dtype=np.float64)
            for l in range(1, DoG_List[i][j].shape[0]):
                for m in range(1, DoG_List[i][j].shape[1]):
                    ext_points = DoG_List[i][j][l][m]
                    if ext_points == max(DoG_List[i][j][l - 1:l + 2, m - 1:m + 2].max(),
                                         DoG_List[i][j - 1][l - 1:l, m - 1:m + 2].max(),
                                         DoG_List[i][j + 1][l - 1:l + 2, m - 1:m + 2].max()):
                        image_extremum[l][m] = ext_points
                        c1 += 1
                    elif ext_points == min(DoG_List[i][j][l - 1:l + 2, m - 1:m + 2].min(),
                                           DoG_List[i][j - 1][l - 1:l + 2, m - 1:m + 2].min(),
                                           DoG_List[i][j + 1][l - 1:l + 2, m - 1:m + 2].min()):
                        image_extremum[l][m] = ext_points
                        c1 += 1
            image_scaleList.append(image_extremum)
        image_extremumList.append(image_scaleList)
    print "Number of Scaled Space Extremum Points:", c1
    key_points = 0
    sigma_nonzero = []
    extremum_nonzero = []
    for i in range(octaves):
        image_sigmaList = []
        image_scaleList = []
        for j in range(scales - 3):
            temp4 = []
            temp4[:] = np.transpose(image_extremumList[i][j].nonzero())
            key_points += len(temp4)
            image_scaleList.append(temp4)
            image_sigmaList.append(math.pow(k, j) * 1.6)
        extremum_nonzero.append(image_scaleList)
        sigma_nonzero.append(image_sigmaList)
    plt.gray()
    plt.figure(n + 1)
    plt.imshow(image1)
    print extremum_nonzero[1][0]
    for i in range(octaves):
        for j in range(0, 2):
            for l in range(len(extremum_nonzero[i][j])):
                x = math.pow(2, i) * extremum_nonzero[i][j][l][0]
                y = math.pow(2, i) * extremum_nonzero[i][j][l][1]
                x1 = [x]
                y1 = [y]
                plt.plot(y1, x1, 'r.')
    plt.title('Non-Zero Extremum Points')
    plt.show()


image = cv2.imread("C:\\Users\\ITCONTROLLER\\Desktop\\left.jpg", 0)
SIFT(image, 0)

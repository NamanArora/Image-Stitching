import sys
from matplotlib import pyplot as plt
import cv2

left = []
right = []
BF = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)

def stitchLeft():
    img1 = left[0]
    for i in range(1,len(left)):
        #Feature extraction 
        featureSet1 = getFeatures(img1)
        featureSet2 = getFeatures(left[i])
        desc1 = featureSet1[0]
        desc2 = featureSet2[0]
        kp1 = featureSet1[1]
        kp2 = featureSet2[1]

        #Image Matching
        matches  = BF.match(desc1,desc2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img1,kp1,left[i],kp2,matches[:10], flags=2)       
        plt.imshow(img3),plt.show() 


        #Finding homography



if __name__ == '__main__':
    args = sys.argv[1]
    print "File loaded : " + args[1]

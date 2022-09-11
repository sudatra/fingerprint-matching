import cv2
import os

sample = cv2.imread('SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_CR.BMP')
#cv2.imshow('sample', sample)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

image = None
fileName = None
kp1, kp2, mp = None, None, None
best_score = 0

for file in [file for file in os.listdir('SOCOFing/Real')][: 1000]:
    print(file)
    fingerprintImage = cv2.imread('SOCOFing/Real/' + file)
    sift = cv2.SIFT_create()

    keyPoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keyPoints_2, descriptors_2 = sift.detectAndCompute(fingerprintImage, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k = 2)
    matchPoints = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            matchPoints.append(p)

    keyPoints = 0
    if len(keyPoints_1) < len(keyPoints_2):
        keyPoints = len(keyPoints_1)
    else:
        keyPoints = len(keyPoints_2)

    if len(matchPoints) / keyPoints * 100 > best_score:
        best_score = len(matchPoints) / keyPoints * 100
        fileName = file
        image = fingerprintImage
        kp1, kp2, mp = keyPoints_1, keyPoints_2, matchPoints

print('BEST MATCH IS: ' + fileName)
print("MATCH SCORE IS: " + str(best_score))
result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx = 3, fy = 3)
cv2.imshow('Result: ', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

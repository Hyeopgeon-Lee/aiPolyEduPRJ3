import cv2

img1 = cv2.imread("../image/bit01.jpg")
img2 = cv2.imread("../image/bit02.jpg")

# 흰색 : 1 / 검정색 : 0
bit_and = cv2.bitwise_and(img1, img2)
bit_or = cv2.bitwise_or(img1, img2)
bit_not = cv2.bitwise_not(img2)
bit_xor = cv2.bitwise_xor(img1, img2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('bitwise_and', bit_and)
cv2.imshow('bitwise_or', bit_or)
cv2.imshow('bitwise_not', bit_not)
cv2.imshow('bitwise_xor', bit_xor)

cv2.waitKey()
cv2.destroyAllWindows()

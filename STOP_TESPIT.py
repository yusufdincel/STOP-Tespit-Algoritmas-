import cv2 as cv
import numpy as np

lower_red1 = np.array([0,100,50])
upper_red1 = np.array([10,255,255])

lower_red2 = np.array([170,100,50])
upper_red2 = np.array([179,255,255])

path = "/home/yusuf/Belgeler/Yazilim/Python/Stop Tespit/stop_sign_dataset/premium_photo-1731192705955-f10a8e7174d2.jpg"

img = cv.imread(path)
orj_w = img.shape[1]
orj_h = img.shape[0]
w = 645
h = 400
boyut = cv.resize(img,(w, h),interpolation=cv.INTER_CUBIC)
alan = 7000 * w *h/orj_w/orj_h

resized = boyut
cv.imshow("Image", resized)

blur =cv.GaussianBlur(resized,(7,7),0)
cv.imshow("Blur",blur)

hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
cv.imshow("HSV",hsv)

mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
mask = cv.bitwise_or(mask1,mask2)
cv.imshow("Mask", mask)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
cv.imshow("Mask 2", mask)

print("Maske Boyutu:", mask.shape)
print("Resim Boyutu", img.shape)

masked = cv.bitwise_and(resized,resized, mask=mask)
cv.imshow("Maskelenmis", masked)

kontur, hierarchies = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in kontur:
    area = cv.contourArea(cnt)

    if area<alan:
        continue

    x,y,w,h = cv.boundingRect(cnt)

    cv.rectangle(resized, (x,y), (x+w,y+h),(0,255,0),2)
    cv.circle(resized, (x+w//2,y+h//2), 4,(0,255,0),-1)
    print("Merkez: (" + str(x+w//2) + "," + str(y+h//2) +")")
    cv.putText(resized, "(" + str(x+w//2) + "," + str(y+h//2) +")", (x,y-(20)),cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), 1 )

cv.imshow("Sonuc",resized)
cv.imwrite("Resim_5.jpg", resized)


cv.waitKey(0)
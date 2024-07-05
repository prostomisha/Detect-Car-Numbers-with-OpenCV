import cv2
import numpy as np
import imutils
#import easyocr
import pytesseract
from matplotlib import pyplot as plt
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
#  detect car numbers using openCV
img = cv2.imread('images/2.jpg')

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_filter = cv2.bilateralFilter(grey, 9, 75, 75)
edges = cv2.Canny(img_filter, 100, 200)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None

for c in cont:
    aprox = cv2.approxPolyDP(c, 7, True)
    if len(aprox) == 4:
        pos = aprox
        break

mask = np.zeros(grey.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)
(x, y) = np.where(mask == 255)
(x1, y1) = np.min(x), np.min(y)
(x2, y2) = np.max(x), np.max(y)
cropped = grey[x1:x2, y1:y2]

#text = pytesseract.image_to_string(cropped)
#text = easyocr.Reader(['en'])
#text = text.readtext(cropped)
final_img = cv2.rectangle(img, (y1, x1), (y2, x2), (0, 255, 0), 4)

plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), cmap='hot')
plt.show()
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), cmap='grey')
plt.show()

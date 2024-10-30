import cv2
import numpy as np

# Görüntüyü oku
img = cv2.imread('C:\\Users\\MSI\\Desktop\\leaf2.jpg', 1)

# RGB'yi HSV'ye dönüştür
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Yeşil için HSV renk aralığı
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Sarı/Kahverengi için HSV renk aralığı
lower_brown = np.array([10, 100, 20])
upper_brown = np.array([20, 255, 200])

# Renk aralığına sahip pikselleri tespit et
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

# Maskeleme ile orijinal görüntü üzerindeki yeşil ve kahverengi alanları bul
result_green = cv2.bitwise_and(img, img, mask=mask_green)
result_brown = cv2.bitwise_and(img, img, mask=mask_brown)

# Sonuç görüntülerini göster
cv2.imshow('Original image', img)
cv2.imshow('Detected green color', result_green)
cv2.imshow('Detected brown color', result_brown)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import qrcode
from PIL import Image

# Read the blank test sheet
sheet = cv2.imread("test_07.png")

# Convert to grayscale
sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)

# Scaling constants
x_offset = 330
y_offset = 10

name = "Test name"

# Make QR code
qr_img = qrcode.make(name)
qr_img = qr_img.convert('L')  # Convert to grayscale
qr_img = np.array(qr_img, dtype=np.uint8)  # Convert PIL image to numpy array

# Resize QR code
qr_img = cv2.resize(qr_img, (100, 100))  # Adjust size as needed

# Calculate coordinates where the QR code should be placed
y1, y2 = y_offset, y_offset + qr_img.shape[0]
x1, x2 = x_offset, x_offset + qr_img.shape[1]

# Place the QR code on the sheet
sheet[y1:y2, x1:x2] = qr_img

# Write the image file
cv2.imwrite(name + ".png", sheet)

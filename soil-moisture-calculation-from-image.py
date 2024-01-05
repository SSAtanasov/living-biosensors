import cv2
import urllib.request
import numpy as np
import serial

# Open image from URL
url = "https://ftt.uni-sz.bg/DSC_5755.JPG"
response = urllib.request.urlopen(url)
img = np.asarray(bytearray(response.read()), dtype="uint8")
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

# Define the masks for the two quadrilaterals
mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
mask2 = np.zeros(img.shape[:2], dtype=np.uint8)

vertices1 = np.array([(1, 1300), (2105, 1425), (2129, 1473), (1, 3433)], np.int32)
vertices2 = np.array([(3985, 1017), (2249, 1417), (2209, 1457), (3985, 3153)], np.int32)

cv2.fillConvexPoly(mask1, vertices1, 1)
cv2.fillConvexPoly(mask2, vertices2, 1)

# Apply the masks to the image
masked_img = cv2.bitwise_and(img, img, mask=mask1 | mask2)

# Resize the masked image to fit the monitor height of 1080
scale_factor = 1080 / masked_img.shape[0]   # Compute the scale factor
resized_img = cv2.resize(masked_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Display the resized image
cv2.imshow("Resized Masked Image", resized_img)
cv2.waitKey(0)

# Save the masked image with a different name
cv2.imwrite("masked-image.jpg", masked_img)

# Count the number of pixels in each masked region
pixels1 = cv2.countNonZero(mask1)
pixels2 = cv2.countNonZero(mask2)

# Calculate the mean values of the R, G, and B color components in each masked region
RGBmean1 = cv2.mean(masked_img, mask=mask1)[::-1]
Rmean1, Gmean1, Bmean1 = RGBmean1[1], RGBmean1[2], RGBmean1[3]
RGBmean2 = cv2.mean(masked_img, mask=mask2)[::-1]
Rmean2, Gmean2, Bmean2 = RGBmean2[1], RGBmean2[2], RGBmean2[3]

# Define Gbp and calculate sm1 based on its value
Gbp = 133.89
if Gmean1 > Gbp:
    sm1 = ((Gmean1 - (-1727.7232) - 74.107143 * 19.03) / 19.020563)
    sm1 *= 1.034  # decrease sm2 by 6% error and x 1,1 (soil coeficient)

else:
    sm1 = ((Gmean1 - 137.12 - 2.566448 * 19.03) / -2.593588)
    sm1 *= 1.166  # increase sm2 by 6% error x 1,1 (soil coeficient)

# Calculate sm2 based on its value
if Gmean2 > Gbp:
    sm2 = ((Gmean2 - (-1727.7232) - 74.107143 * 19.03) / 19.020563)
    sm2 *= 1.034  # decrease sm2 by 6% error x 1,1 (soil coeficient)

else:
    sm2 = ((Gmean2 - 137.12 - 2.566448 * 19.03) / -2.593588)
    sm2 *= 1,166  # increase sm2 by 6% error x 1,1 (soil coeficient)

# Print the mean color values and pixel counts in each masked region
print("Row 1:")
print(f"  Mean color values (RGB): [{round(Rmean1, 2)}, {round(Gmean1, 2)}, {round(Bmean1, 2)}]")
print(f"  Number of pixels: {pixels1}")
print(f"  Soil Moisture: {round(sm1, 2)}")

# Determine whether to start or stop irrigation based on sm1 value
if sm1 < 20:
    print(f"  Irrigation started in Row 1")
    # Send the SMS to the Arduino
    ser.write(b'Start irrigation in Row 1')
elif sm1 >= 20 and sm1 <= 31:
    print(f"  Soil moisture is sufficient in Row 1. Irrigation not started.")
else:
    print(f"  Irrigation stopped in Row 1")
    # Send the SMS to the Arduino
    ser.write(b'Stop irrigation in Row 1')

print("Row 2:")
print(f"  Mean color values (RGB): [{round(Rmean2, 2)}, {round(Gmean2, 2)}, {round(Bmean2, 2)}]")
print(f"  Number of pixels: {pixels2}")
print(f"  Soil Moisture: {round(sm2, 2)}")

# Determine whether to start or stop irrigation based on sm2 value
if sm2 < 20:
    print(f"  Irrigation started in Row 2")
    # Send the SMS to the Arduino
    ser.write(b'Start irrigation in Row 2')
elif sm2 >= 20 and sm2 <= 31:
    print(f"  Soil moisture is sufficient in Row 2. Irrigation not started.")
else:
    print(f"  Irrigation stopped in Row 2")
    # Send the SMS to the Arduino
    ser.write(b'Stop irrigation in Row 2')

# Define the serial port and baudrate
ser = serial.Serial('COM3', 9600)

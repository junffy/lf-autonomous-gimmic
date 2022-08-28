import cv2

img = cv2.imread('./input.jpg')
output_img = img.copy()

img = img[0:540, 0:1920]
img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)

mask_red1 = cv2.inRange(img, (0, 127, 127), (30, 255, 255))
mask_red2 = cv2.inRange(img, (240, 127, 127), (255, 255, 255))
img = cv2.bitwise_or(mask_red1, mask_red2)

circles = cv2.HoughCircles(image=img,
                           method=cv2.HOUGH_GRADIENT,
                           dp=1.0,
                           minDist=100,
                           param1=100,
                           param2=10,
                           minRadius=0,
                           maxRadius=50)

for center_x, center_y, radius in circles.squeeze(axis=0).astype(int):
   cv2.circle(output_img, (center_x, center_y), radius+40, (0, 0, 255), 10)

cv2.imwrite('./output.jpg', output_img)
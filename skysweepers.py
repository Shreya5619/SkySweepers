import cv2
import numpy as np

def finding_rect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny edge detection to detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # center of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    # 1/10th of the image area
    min_area = (image.shape[0] * image.shape[1]) / 10

    # Initialize  minimum distance and closest rectangle
    min_distance = float('inf')
    closest_rect = None

    # Loop through contours to find the rectangle closest to the center
    for contour in contours:
        # bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # area of the rectangle
        rect_area = w * h
        
        # Check if the area is at least 1/10th of the image area(Main Condition)
        if rect_area >= min_area:
            # center of this rectangle
            rect_center = (x + w // 2, y + h // 2)
            # Calculate the Euclidean distance from the image center
            distance = np.sqrt((rect_center[0] - image_center[0]) ** 2 + (rect_center[1] - image_center[1]) ** 2)
            
            # Updatation of closest rectangle
            if distance < min_distance:
                min_distance = distance
                closest_rect = (x, y, w, h)

    # Draw the closest rectangle on the original image
    if closest_rect is not None:
        x, y, w, h = closest_rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the ROI using slicing
        roi = image[y:y + h, x:x + w]
    else:
        print("No rectangle meets the area requirement.")
    return roi

def dustdetect(image):
    blurred = cv2.GaussianBlur(image, (1, 1), 0)

    edges = cv2.Canny(blurred, 50, 150)
    # morphological operations to differentiate lines and dust
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    # Use morphological closing to fill small gaps in the detected lines
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #Fourier Transform to isolate the lines
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2

    # Mask creation: remove the center lines
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-10:crow+10, :] = 0  # Horizontal lines
    mask[:, ccol-10:ccol+10] = 0  # Vertical lines
    # Apply the mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # Convert to 8-bit for visualization
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100, minLineLength=90, maxLineGap=10)

    # Create a mask for the lines
    line_mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    # Invert the mask to remove lines
    line_removed = cv2.bitwise_not(line_mask)
    filtered_image = cv2.bitwise_and(image, image, mask=line_removed)
    # Thresholding the filtered image to detect dust
    _, dust_mask = cv2.threshold(filtered_image, 50, 255, cv2.THRESH_BINARY)
    # Combine the dust mask with the original to highlight dust areas
    dust_highlighted = cv2.bitwise_and(image, dust_mask)
    return dust_highlighted

def calculate_white_percentage(binary_image):
    # Calculate the number of white pixels (pixels with value 255)
    white_pixels = np.sum(binary_image >= 140)
    
    # Calculate the total number of pixels
    total_pixels = binary_image.size
    
    # Calculate the percentage of white pixels
    white_percentage = (white_pixels / total_pixels) * 100
    print(white_percentage)
    if(white_percentage>=2):
        return 1
    else:
        return 0

image = cv2.imread("C:\\Users\\Shreya Prasad\\Desktop\\Solartest4.jpg")
cv2.imshow("imagei",image)


roi= cv2.cvtColor(cv2.resize(image,(512,512)), cv2.COLOR_BGR2GRAY)
_, th4 = cv2.threshold(roi, 140, 255, cv2.THRESH_TOZERO)
roi = cv2.adaptiveThreshold(th4, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
th4=dustdetect(th4)
cv2.imshow("thresh",th4)

print(calculate_white_percentage(th4))

cv2.waitKey(0)
cv2.destroyAllWindows()

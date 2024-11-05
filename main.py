import cv2
import numpy as np
import mss
import time

# Function to detect a specific color
def detect_rectangle(frame, lower_color, upper_color, rectangle_counter, last_detection_time, wait_time, min_area):
    # Converts the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Creates a mask for pixels within the color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Applies morphological operations to clean the mask (opt)
    kernel = np.ones((5, 5), np.uint8)  # Kernel para operações morfológicas
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Limpeza de pequenas falhas na máscara
    
    # Find the contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Flag to know if at least one rectangle was found
    rectangle_detected = False

    # Goes through the contours and checks if there is a rectangle
    for contour in contours:
        # Approximates the contour to a polygon and checks if it is a rectangle
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust precision
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the contour has 4 points, it is a rectangle
        if len(approx) == 4:
            # rectangle area
            area = cv2.contourArea(contour)
            
            # Checks if the area is greater than the defined minimum area
            if area > min_area:
                # Draw the rectangle on the original image
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                rectangle_detected = True  # Detection Flag
                break  #Exit the loop after detecting a rectangle (avoids multiple counting in the same frame)

    # Rectangle counting only occurs after the rectangle has been drawn
    if rectangle_detected:
        current_time = time.time()  # Actual time

        # Cooldown Timer
        if current_time - last_detection_time >= wait_time:
            rectangle_counter += 1  # Increment the sequential counter
            last_detection_time = current_time  # Updates last detection time

    # Add text with rectangle count
    cv2.putText(frame, f"Laps: {rectangle_counter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame, mask, rectangle_counter, last_detection_time

# Set the color limits for the rectangle HSV Format
##lower_red = np.array([12, 165, 157])
##upper_red = np.array([13, 166, 158])
lower_red = np.array([0, 130, 255])   
upper_red = np.array([0, 130, 255])  


# Initializes the rectangle counter and last detection time
rectangle_counter = 0  # Initial counter value
last_detection_time = 0  # Initializes the last detection time
wait_time = 10  # Cooldown time (in secounds)
min_area = 700  # Minimum area for the rectangle to be detected (in pixels)

with mss.mss() as sct:
    monitor = {"top": 10, "left": 300, "width": 640, "height": 480}  # Detecton window size

    while True:
        last_time = time.time()

        # Capture the screen
        img = np.array(sct.grab(monitor))

        # Detects the rectangle in the image and updates the counter and time of last detection
        img_with_rect, mask, rectangle_counter, last_detection_time = detect_rectangle(
            img, lower_red, upper_red, rectangle_counter, last_detection_time, wait_time, min_area
        )

        # Displays the image with the rectangle drawn
        cv2.imshow("Detector", img_with_rect)

        # Display the binary mask (optional)
        cv2.imshow("Mask", mask)

        # FPS calc
        fps = 1 / (time.time() - last_time)
        cv2.putText(img_with_rect, f"FPS: {fps:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Uncomment to dislays a FPS on screen
        ##cv2.imshow("FPS Display", img_with_rect)

        # Exit when 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

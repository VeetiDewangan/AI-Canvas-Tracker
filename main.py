import cv2
import numpy as np

# 1. Setup Color Range (Adjust these if your blue object isn't detected)
blueLower = np.array([90, 50, 50])
blueUpper = np.array([150, 255, 255])

# 2. Points storage: A list of lists to handle separate strokes
points = [[]]
paintIndex = 0

# 3. Initialize Camera
cap = cv2.VideoCapture(0)

# 4. Create a blank white canvas
# Using 480x640 to match standard webcam resolution
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

print("Controls: 'q' to Quit | 'c' to Clear Canvas")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame for a mirror effect (more natural for drawing)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 5. Create a mask for the blue color
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 6. Find contours (the outline of your blue object)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any blue object is detected
    if len(cnts) > 0:
        # Find the largest contour (the brush)
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Only draw if the object is big enough (filters out background noise)
        if radius > 15:
            # Calculate the center (centroid) of the object
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            # Add the current center to the active stroke
            points[paintIndex].append(center)
            
            # Visual feedback: Draw a circle around the object being tracked
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
    else:
        # PEN UP LOGIC: If no blue is seen, finalize the current stroke
        # and prepare a new empty list for the next time blue appears.
        if len(points[paintIndex]) > 0:
            points.append([])
            paintIndex += 1

    # 7. Draw the stored points on both the camera feed and the canvas
    for i in range(len(points)):
        for j in range(1, len(points[i])):
            if points[i][j - 1] is None or points[i][j] is None:
                continue
            # Draw blue lines (BGR: 255, 0, 0) with thickness 3
            cv2.line(frame, points[i][j - 1], points[i][j], (255, 0, 0), 3)
            cv2.line(canvas, points[i][j - 1], points[i][j], (255, 0, 0), 3)

    # 8. Display the windows
    cv2.imshow("Tracking (Input)", frame)
    cv2.imshow("Canvas (Output)", canvas)

    # 9. Handle Keyboard Interrupts
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'q' to exit
    if key == ord('q'):
        break
        
    # Press 'c' to clear the canvas
    if key == ord('c'):
        points = [[]]
        paintIndex = 0
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Cleanup
cap.release()
cv2.destroyAllWindows()
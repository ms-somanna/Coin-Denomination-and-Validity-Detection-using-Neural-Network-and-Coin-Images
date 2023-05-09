import os # For manipulating local directories to read and write image files.
import cv2 as cv # OpenCV library
import numpy as np
from rich.progress import track # For displaying progressbar while images are being processed

print('Currently installed OpenCV version: ', cv.__version__)
print('Code tested on OpenCV version 4.7.0\n')

def crop_coin():
    input_path = ''
    output_path = ''
    
    # Get the no. of files in the input path
    _, _, files = next(os.walk(input_path))
    file_count = len(files)
    
    processing_count = 1 # Counter for no. of files being processed
    
    # Process
    for filename in track(os.listdir(input_path), description='Processing'):
        
        # Print the index of the file currently in process
        print('Processing Image ', processing_count, ' of ', file_count)
        
        processing_count += 1
        
        # Get the full filename of the input image
        file_path = os.path.join(input_path, filename)
        
        print('File: ', filename)
        
        # Read the file into an image array
        img = cv.imread(file_path)
        assert img is not None, "file could not be read, check with os.path.exists()"
        
        # Apply Gaussian blur to smooth out any irregularities in the image
        blurred_img = cv.GaussianBlur(img, (5, 5), 0)
        
        # Convert the blurred image to grayscale
        gray_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)
        
        # Apply threshold to the grayscale image to obtain a binary image
        _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        # Perform morphological operations to clean up the binary image
        kernel = np.ones((5,5), np.uint8)
        binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)
        
        # Find contours in the binary image
        contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Get the contour with the largest area
        largest_contour = max(contours, key=cv.contourArea)
        
        # Fit a circle to the obtained contour. Used for locating the coin in the image.
        center, radius = cv.minEnclosingCircle(largest_contour)
        
        # Convert the center and radius to integer values
        center = tuple(map(int, center))
        radius = int(radius)
        
        # Calculate the coordinates of the bounding box around the coin
        x, y = center[0] - radius, center[1] - radius
        w, h = 2*radius, 2*radius
        
        # Create a black image with the same size as the input image.
        # This for replacing the image background with black color.
        mask = np.zeros_like(img)
        
        # Draw a white filled circle on the black image using the circle coordinates
        cv.circle(mask, center, radius, (255, 255, 255), -1)
        
        # Set the pixels outside the circle to black
        masked_img = cv.bitwise_and(img, mask)
        
        # Crop the input image using the bounding box coordinates to obtain the region of interest.
        cropped_img = masked_img[y:y+h, x:x+w]
        
        # Convert image to grayscale if the cropped image array is not empty.
        if cropped_img.size != 0:
            gray_img = cv.cvtColor(cropped_img, 6)
        else:
            print('\033[31mEmpty image array was produced!\033[0m') # Printing in red forecolor
        
        # Resize the grayscale image to a square size.
        resized_img = cv.resize(gray_img, (256,256))
        
        # Save the grayscale image to a local directory
        output_file_path = os.path.join(output_path, filename)
        cv.imwrite(output_file_path, resized_img)
        print('Done', '\n')
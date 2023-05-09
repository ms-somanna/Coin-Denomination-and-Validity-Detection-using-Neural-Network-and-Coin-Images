The following image processing steps were executed to detect the coin in the original image, crop it along it's region and save it as a new image.  
## Import required libraries and modules

```python
import os # For manipulating local directories to read and write image files.
import cv2 as cv # OpenCV library
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline

print('Currently installed OpenCV version: ', cv.__version__)
print('Code tested on OpenCV version 4.7.0\n')
```

## Load Image

```python
path = 'sample_image.jpg'

img = cv.imread(path)
assert img is not None, "file could not be read, check with os.path.exists()"

img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Optional

plt.imshow(img)
```

> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236855720-e197296e-d6ff-4a5a-b713-05c4708ab021.png)

## Apply Gaussian Blur to smooth out any irregularites
By blurring the image, noise and small varieations in lighting and texture can be removed and the larger features such as the coin can be made more prominant. This makes it easier for the the thresholding and contour detection steps.

```python
blurred_img = cv.GaussianBlur(img, (5, 5), 0)
plt.imshow(blurred_img)
```
> Output
>
>![image](https://user-images.githubusercontent.com/32904377/236856015-e3be3a6b-4a9f-4c4a-98c7-f9a0af93bce5.png)

## Convert the blurred image to grayscale
Since the image contains a coin of any color and a near black background, and we are intrested only in detecting the edges of the coin, color information is not necessary. 
By removing the color from the image, the further processing steps will only have to deal with one intensity value instead of three color channels. This will make it easier of the them to focus only the pixel brightness values which is only usefull in detecting the contours.

```python
gray_img = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)
plt.imshow(gray_img, 'gray')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236856128-e6b62a5b-1c8e-42e8-a9dc-9a4fa23ba8a2.png)

## Apply threshold to obtain a binary image
By thresholding a grayscale image, we can convert it to a binary image where each pixel is either black or white. Thus making the the image more consistant. Such consistancy can improve the accuracy of further steps such as contour detection.

```python
_, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# The cv.THRESH_BINARY+cv.THRESH_OTSU flag enables automatic determination of optimal threshold value.

plt.imshow(binary_img, 'gray')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236856191-ea6ff83a-5811-47b1-97bf-a80262db052d.png)

## Perform morphological operations to clean up the binary image
Morphological operations are preformed to manipulate the structure of the image. I.e. perform operations on pixels based on the values of their neighbouring pixels.  
Here morphological operations use a kernal to define the neighbouring pixels, and based on the kernel cleans the binary image off of any holes and gaps and obtain a smoother binary image.

```python
kernel = np.ones((5,5), np.uint8)

binary_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)
# The cv.MORPH_CLOSE flag is used to clos any small gaps and holes.

plt.imshow(binary_img, 'gray')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236856246-6995d2ba-3d10-4e9a-8cb2-5e1c8b8dc95f.png)

## Find contours in the binary image
Contours are basically boundries of an object in an image. Contours are used to locate the boundry of the coin in the binary image.
Contours of the coin can be detected using the `findContours` function with flags `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`.  
The function `findContours` returns a tupple of arrays, where each array contains location of a contour.  
The flag `RETR_EXTERNAL` returns only external boundries. This means it returns only the external boundries of the coin ignoring the internal contours within the coin.  
The flag `RETR_EXTERNAL` is used to compress the contours, and thus reduces the computation costs to store and manipulate them.

```python
contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# The above code returns a tupple of arrays. Where each array contains location points of a contour


# Plot the contours
x = [] # List of x-coordinates of all points in all contours 
y = [] # List of y-coordinate of all points in all contours

for contour in contours:
    for point in contour:
        x.append(point[0,0])
        y.append(point[0,1])

fig, ax = plt.subplots()        

ax.imshow(binary_img, 'gray')

ax.scatter(x=x, y=y, s=1, c='blue')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236856361-c3b66992-806a-4b8e-85c4-f9e0c6cec75e.png)

## Obtain the largest contour
The largest contour is assumed to be the external contour of the coin.  
The function `contourArea` is used to find the area of all the contours and is passed to the `max` function to obtain the contour with the largest area.

```python
largest_contour = max(contours, key=cv.contourArea)

x_l = largest_contour[:,:,0] # List of x-coordinates of all points in the largest contour. 
y_l = largest_contour[:,:,1] # List of y-coordinates of all points in the largest contour.

ax.scatter(x=x_l, y=y_l, s=1, c='r')
fig
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236856451-acc01375-d70c-46b9-84af-536aa2fb5bfe.png)

## Fit a circle to the obtained contour
Once the largest contour is obtained, a circle can be fitted to the contour using the function `minEnclosingCircle`.  
This function finds the minimum enclosing circle for a given set of points. In this case the set of points are the contour. And thus the function returns a a tupple with the coordinates of the center of the circle and single value radius. Using the center and radius values, we can find the coordinates of a bounding box that fits around the coin.

```python
center, radius = cv.minEnclosingCircle(largest_contour)
print('Center: ', center)
print('Radius: ', radius)
ax.plot(center[0], center[1], 'go')
ax.plot([center[0], center[0]+radius],[center[1],center[1]], 'g--')
fig
```
> Output
> 
> Center:  (1114.1502685546875, 2007.5736083984375)  
> Radius:  443.1164855957031  
>![image](https://user-images.githubusercontent.com/32904377/236856582-05283959-295e-4fdc-b203-2dec55c6427b.png)

```python
# Convert the center and radius to integer values
center = tuple(map(int, center))
radius = int(radius)

print('Center: ', center)
print('Radius: ', radius)
```
> Output
> 
> Center:  (1114, 2007)  
> Radius:  443  

## Create a bounding box around the coin
This bounding box is our Region Of Interest (ROI)

```python
# Get the coordinates of the top-left corner of the bounding box
x, y = center[0] - radius, center[1] - radius

# Get the width and height of the bounding box
w, h = 2*radius, 2*radius 

# Plot the bonding box
rect = patches.Rectangle((x,y), w, h, facecolor='none', linewidth=1, edgecolor='lime', linestyle='--')
ax.add_patch(rect)
fig
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236856973-a4274760-40bc-4c2c-8b4c-7dce733a3513.png)

## Creating a Mask
Since the background of the original image doesnt contain any useful information, we can replace it with a blank background, thus making it easier for ML models to ignore the background while learning the features in the image.  
We can create a blank background by setting all the pixels in the background to one uniform color. For simplicity, we'll choose the color black.  
In order to replace the image background with all black, we'll have first create a mask containing a black background and a white filled circle on the region of the coin. Then, when we do a bitwise-AND operation between the original image and the mask, the pixel values in the background of the original image get replaced with black and the region containing the coin is preserved as it is. This effectly accomplishes the task of making the background of the original image black.

The first step in creating the mask is obtaining a black image of the same size as the original image.
### Create a black image with the same size as the input image. 

```python
# Create an image array of the same size as the original image array, with all pixels having 0 values.
mask = np.zeros_like(img)
plt.imshow(mask, 'gray')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236857037-f7434694-03c0-4553-b230-ddf67058bee2.png)

### Draw a white filled circle on the region containing the coin.
The white filled circle on the mask can be drawn using the circle coordinates obtained in the previous steps.  
This ensures that the circle is drawn where the region of the coin exists.

```python
cv.circle(mask, center, radius, (255, 255, 255), -1)
plt.imshow(mask)
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236857091-ada28822-35fc-4bf6-96b4-9eb73fe7a751.png)

## Perform masking

```python
masked_img = cv.bitwise_and(img, mask)
plt.imshow(masked_img)
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236857172-2faecce2-dce5-4cd9-ab45-153526890675.png)

As you can see, the background in the image is now completely black.

## Crop the input image using the bounding box coordinates to obtain the region of interest

```python
cropped_img = masked_img[y:y+h, x:x+w]
plt.imshow(cropped_img)
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236857306-802ec592-f9c8-468f-878a-6f4e61d2a626.png)

## Convert the image to grayscale
This step is optional. Since I had decided to remove any color information as it isn't important for detecting the denomination or validity of the coin by an ML model, I liked to convert the image to grayscale. But if for some reason you want to keep the colors, you can skip this step.

```python
gray_img = cv.cvtColor(cropped_img, 6)
plt.imshow(gray_img, 'gray')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236857344-e01332c5-c5b4-4f18-ab5d-627ea0c9848a.png)

## Resize the image to a desired size
The size of the cropped image above seems to be unnecessarily large for a Neural Network model, so I decided to resize it to 256x256 pixels. Which is considered an ideal size for NN models. You can set any size you want, or keep the original size as it is.  

```python
resized_img = cv.resize(gray_img, (256,256))
plt.imshow(resized_img, 'gray')
```
> Output
> 
>![image](https://user-images.githubusercontent.com/32904377/236857413-5aaf5ac7-e6c4-4589-91a6-aa8cfc4479cb.png)

## Save the image
Once the above steps are completed you can save the image using OPenCV's `imwrite` function.  

> Note  
> Though converting the image to grayscale removes all the three color channels from the image, while it is saved as an image file by the `imwrite` function, all three color channels with same intensities are added back to the image. Since all three channels posses the same intensities, the image will still appear as grayscale.

That's it.

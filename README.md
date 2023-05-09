# Indian-Coin-Denomination-and-Validity-Detection-using-Coin-Images-and-Neural-Network
This a project I made to build a Neural Network model that can classify Indian coins based on their denomination and can also classify weather the coin is valid or fake/invalid. The Neural Network was built on a ConvNet archetecture and has learnt on around 2000 images of coins, both valid and invalid.  

Vending machines and money changers use several methods to detect and identify coins that are inserted into them. Though these methods are effective, they still cause some probelms, which can be potentially solved by replaceing the existing methods with a Neural Network model that can classify the coins based on their denominations and validity.  
The main goal of this project is to build one such NN model to solve the existing problems. The probelms and solution are discussed in detail in the sections below.

## Problem
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12921587%2F8cdd2e3fbfb1a3e97a9ff5733068564c%2Fvending%20machine.jpg?generation=1683391775041836&alt=media)

Vending machines and money changers deal with taking in cash form a user, identifying the denomination of the cash, and detect cash's validity ( i.e is it a real or a counterfeit (fake)). Coins seem to be a major type of cash that are exchanged in these machines. The current methods of identifying the denomination of coins by the these machines commonly involves measuring the weight, size, and thickness of the coin using techniques such as light sensing, and detecting weather the coin is real or fake involves obtaining the composition of the coin to detect if it is metallic or not using techniques such as electromagnetic field transmission. These methods do work, except there are two problems:
1. In many countries (like India), the versions of coins keep changing overtime and consequently the dimensions and compositions of the coins change too making it difficult to identify and detect coins using the afore mentioned methods.
2. Fake coins can be made of metallic plates that resemble the size and composition of real coins, when such plates are inserted into the machine, they might get falsely detect as valid coins. 

## Solution
One potential way to solve these problems would be to use Machine Learning to identify the coin denomination and detect it's validity. But how? An ML model can be trained on a dataset of images of both real and fake coins of various denominations. If the images were captured with a flash, the amount of light reflected off of the surface of metallic coins would be higher compared to that of non-metallic fake coins. Also the higher reflection from metallic surface would produce regions of bright spots only in images of metallic coins.

The figure below shows some images captured with flash, taken from the dataset, showing some differences present between the images of real and fake coins.

<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12921587%2F5bf4e9119046d51e7c9258eada047ae0%2FRealVsFakeCoins.jpg?generation=1683396232330130&alt=media' width='75%'>

(a) & (b) are images of a real coin, showing reflected light as bright spots on the coin.
(c) is the image of a fake non-metallic coin, not showing any apparent bright spots of light.
(d) is fake too but is a metallic plate and thus showing spots of brightness, but lacks the presence of imprints of denomination, the rupee symbol and artistry.
(e) is fake, made by printing the image of a real coin on a piece of paper and pasted on a plastic plate, and thus the image contains both the spots of brightness and the information and artistry similar to that of a real coin but lacks the lustrous appearance of a real coin. Also the paper surface appears rougher compared to a real coin.

Keeping in mind these differences, a dataset of images was created containing 1750 images of various real and fake coins captured with flash and the images are classified into:

1. The front face of real/valid Indian coins consisting the imprint of common denominations- Rs 1, Rs 2, and Rs 5.

   <img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12921587%2F3cbee0087a974d872ff3684709e9ad2d%2Freal_coins_front.png?generation=1683302664681760&alt=media' width='60%'>

   The images of front face of 1 rupee coins are contained in the folder named '1_rupee', images of 2 rupees in folder named '2_rupee' and 5 rupees in '5_rupee'.

2. The back or reverse face of real/valid coins that do not show their denomination values.

   <img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12921587%2F23601ab0ffde0cf1465058ea94e03f80%2Freal_coins_back.png?generation=1683302796939179&alt=media' width='60%'>

   These images are contained in the folder named 'reverse'

3. Fake/Invalid coins.

   <img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12921587%2Fb4adf4b82266250237fcf61722732aa5%2Ffake_coins.png?generation=1683302878176987&alt=media' width='60%'>

   These images are contained in the folder named 'invalid'


The images were originally captured with a 12MP camera with a resolution of around 2000x4000 pixels and were processed in OpenCV to detect the region of the coin within the image, crop it along the region and resize it to a resolution of 256x256 pixels. 

<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F12921587%2Fa9b0997b7704c2d679eb12db880e5f5f%2FImageProcessing.png?generation=1683394637379764&alt=media' width='75%'>

This preprocessing was done to make it easier for ML models to learn important features. The code for processing is present in the file `Detect Coins.py` or the notebook version `Detect Coins.ipynb` and the steps involved in the processing are described in detail in the [Image Processing guide](https://github.com/ms-somanna/Indian-Coin-Denomination-and-Validity-Detection-using-Coin-Images-and-Neural-Network/blob/4791e5bd5ace60b79f69f92bd503de4d3796517d/Image%20Processing%20Steps.md). 

### Classification by the NN Model
The NN model trained on this dataset can classify images as either:
1. 1 Rupee Coin
2. 2 Rupee Coin
3. 5 Rupee Coin
4. Reverse face of coin
5. Inavlid image / fake coin

### Arcitecture of the Model

![NN Model 1 Architecture-Page-2 drawio(2)](https://user-images.githubusercontent.com/32904377/236687394-d39f5997-cd9e-4bc9-a66a-83efe1216625.svg)

### Performance of the Model
Test Loss: 0.06  
Test Accuracy: 0.98

<img src='https://user-images.githubusercontent.com/32904377/236687810-adb99363-d445-4c40-8cc5-e5582b0bd41a.jpg' height='75%'>

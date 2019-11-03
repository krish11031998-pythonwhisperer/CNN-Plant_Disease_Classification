# Plant Disease Classifier

## Dataset

The images of different plants were taken in healthy and unheatlhy states .. the raw plant pictures were then converted from color to *grayscale* and *segmented* (The picture area of the plant cropped out from the background) versions.

For this projects , I considered using the color raw images of the plant. For more information , pls check out the following:
[git repo for the dataset ] (https://github.com/spMohanty/PlantVillage-Dataset)

The Image were read using the OpenCV library , where the images were transformed into scaled version of *image_arrays* that was eventually fed into the CNN


## CNN Architecture

The initial design of the CNN consisted of:
* Convulational Layer (2D).
* Activation Layer.
* BatchNormalization (proved to be very advantageous for this problem especially).
* MaxPool Layer.
* Dropout Layer.
* Final Layers:
    * Dense.
    * Flatten.
        
The CNN was compiled using the *Adam* Optimizer , with a learning rate = 1e-3, with binary_crossentropy loss function.
        
    
The neurons, dropout ratio, pooling_size, activation function and optimizer design were chosen based on the hyperparameter tuning results (Use can try tuning the CNN yourself with the code I provided)


### Have fun trying this classifier model out , Future plan : Create a API and use it in a browser



    
    





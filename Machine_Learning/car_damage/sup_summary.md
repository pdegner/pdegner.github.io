# Car Damage Detection

##### Patti Degner, Bethany Keller, and Chris Sexton

*This document is a work in progress*

## Goal

The goal of this project is to to predict the location and severity of damage to a car given a provided image of the damaged car. This information could be used for faster insurance assessment and claims processing.

## Data

Training and validation data was made available through Kaggle: https://www.kaggle.com/anujms/car-damage-detection#0001.JPEG

51.5% of the test images are whole. So, to demonstrate improvement from always guessing whole, we must do better than 51.5% accuracy. 


# Traditional Supervised Learning

The first attempt to classify cars as damaged or whole uses traditional supervised learning techniques Naive Bayes and KNN. In each of these methods, I compare the results of the original images and images that were blurred using a bilateral filter. 

##### What is bilateral filtering?
To understand bilateral filtering, you must first understand Gaussian blurring. Gaussian blurring looks at each pixel, then replaces that pixel value with the average of all pixels around it. The result is a blurred image with smoothed edges. This means that image has less noise, which can be beneficial for image classification. However, it also means that the edges are less defined, and in the case of car damage detection, this could be a bad thing. 

Bilateral filtering is similar to Gaussian blurring in that it will replace each pixel value with the average value of all pixels around it. However, if the pixel is part of an edge, the values will not be changed. The resulting image appears slightly blurred, but still has crisp edges. This means there is less noise in each image, and the edges have not been sacrificed. The hope is that bilateral filtering will increase the accuracy of Naive Bayes and KNN. Below is an example of bilateral filtering using an image from the dataset. 

![bilateral_filter_example](bilateral_filter.png)

### KNN
The first attempt at classifying a car as damaged or whole uses K-nearest neighbors. Using a naive guess of k=5, my results were as follows:
  ```
  Accuracy when k=5 on unfiltered data is 60.99%
  Accuracy when k=5 on filtered data is 59.64%
  ```
  
This is not great. The default for Scikit learn's `KNeighborsClassifier` weight parameter is `weights = uniform`. This means that all points in each neighborhood are weighted equally. If `weights = distance`, then the points are weighted by the inverse of their distance. This means that closer neighbors will have a greater influence than more distant neighbors. After changing the weights parameter to `weights = distance`, my results were as follows:
  ```
  Accuracy when k=5 on unfiltered data is 61.66%
  Accuracy when k=5 on filtered data is 60.31%
  How much did accuracy change?
	  For unfiltered data: 0.6726%
	  For filtered data: 0.6726%
  The change is the same for both models: True
  ```
This is a little better, but still not much better than random guessing. Perhaps I need a different value for k?
![k_values](k_values.png)

With a maxium accuracy of %62.11, perhaps KNN is not the best choice for this task. Next, I will try Naive Bayes.


### Naive Bayes
A simple Bernolli Niave Bayes model yields the following results:
  ```
  Accuracy on unfiltered data is 61.66%
  Accuracy on filtered data is 49.78%
  ```
Here, the filtered data is doing worse than the unfiltered data. Laplace smoothing made this model even worse, which makes sense because it removes edges that may help detect car damage. Because of this low accuracy, I did not persue Naive Bayes further.

### Why the issues?
Why are KNN and Naive Bayes performing so poorly? One problem is the diversity of images in the dataset. 

![image_problems](image_problems.png)

1. The images in the dataset are taken from different angles and distances.
2. Sometimes there are subjects other than cars in the images.
3. As a human, it can be difficult to tell where, or even if, a car is damaged.
4. Some of the cars are so badly damaged it is difficult to tell that it is even a car. 
5. The damage is in different areas; sometimes it is on the wheel, sometimes the body, the windshield, etc. 

I believe all of these issues are contributing to the inaccuracy of my models. 

## Next step: Neural Networks
When traditional machine learning fails, it is time to call in the big guns: neural networks. Please check out part 2. 


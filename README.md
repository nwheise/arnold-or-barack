# arnold-or-barack
Project for course in Computer Vision to classify images as either Arnold Schwarzenegger or Barack Obama.  
Includes use of OpenCV and a self-written implementation of Principal Component Analysis (PCA).

## Intro
Face detection and recognition is a major topic in computer vision.  
In this project we develop a program to detect faces on pictures and recognize Barack Obama and Arnold Schwarzenegger.

## Outline of the approach
<ol>
<li>Use the built-in face detector of OpenCV to detect the faces in the images</li>
<li>Crop and resample the images to the region of interest (the face) and save them in another directory</li>
<li>Do a principal component analysis on the previously created images and build a model for 
the face of Arnold and another for the face of Barack</li>
<li>Classify each image in the test set (these were not used for building the model):</li>
  <ol>
  <li>Represent the test image in both models</li>
  <li>See which model represents the image the best (minimal sum of squared differences between
  the original and the reconstructed image</li>
  </ol>
</ol>

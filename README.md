# Exoplanet Detection Using Machine Learning
an attempt to identify exoplanets from the tiny changes they cause in the brightness of star.

## Dataset
The dataset used in this project comes from Kaggle's Kepler Labelled Time Series Data dataset, which is based on observations from NASA's Kepler mission.

the main problem with this dataset which i noticed is, that it is extremely imbalanced. there are roughly around 5050 stars out of which there are only 37 positive planet examples.
training a model with such training data was challenging

another problem was, the model could simple learn to predict:
no planet
no planet 
not planet 
...
...
.
to anything and still get high accuracy since the majority are non planets.
---
## approches
**initial preprocessing**
- pandas was used for handling dataset csv file
- StandardScaler for feature scaling
- train/test splitting using scikit-learn
- RandomOverSampler for dealing with the minority class
- smoothening light curves
- SMOTE
- training with MLP
- training with CNN
- randomforest

  One of the SMOTE evaluations produced this confusion matrix:
  [[504, 27],
   [  6,  1]]

## Neural Network
My first main model was a fully connected neural network using TensorFlow/Keras.

The architecture was:

Input
  ↓
Dense(256, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(128, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(64, ReLU)
  ↓
Dense(1, Sigmoid)

I used Adam as the optimizer and binary cross-entropy as the loss function.

## threshold problem
fixing a threshold was harder than i thought it would be. the low positive cases in the trainingdata set made the model fluctuate between confidence 

## Accuracy isn't enough

Imagine there are 5,000 stars and only a few actually contain planets.

A model that predicts "no planet" for almost everything can get a high accuracy.

But it could still miss almost every planet.

That's why I care about precision, recall and F1-score here.

## Precision

Of the stars the model called planets, how many were actually planets?

## Recall

Of all the actual planets, how many did the model find?

## F1-score

F1 gives a balance between precision and recall.

For this problem, that balance is much more interesting than accuracy by itself.
---
## made with
Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
imbalanced-learn
Matplotlib
Git
GitHub

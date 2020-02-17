# Sleep Apnea Detection
<p align="center"><img src="https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/icon_cover.png" width="700" title="Sleep Apnea Detection"></p>

# Project Overview
Over 18 million Americans suffer from Sleep Apnea, which leads to sleep deprivation, hypertension, heart diseases, and even stroke. Many people are not aware of it (thinking that they just snore a lot) or are concerned about the cost and effort of a sleep study. 

The purpose of this app is to detect sleep apnea using signals from the heart rate measurement, so that wearable devices can provide early warning and direct people to sleep studies for diagnosis and treatment. 

## Background
<p align="center"><img src="https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/Slide_HR_character.png" width="450" title="Heart rate during apnea"></p>
This figure shows a typical pattern of heart rate data during apnea. Due to the periodical breathing interruption, the heart rate fluctuates with a larger range at a specific frequency. As such, using features in the time and frequency domains, predictions can be made on whether the user is undergoing sleep apnea.

## Data
The prediction model is trained using a [dataset](https://physionet.org/content/apnea-ecg/1.0.0/) on the PhysioNet, which provides the electrocardiogram recording of 70 participants during their sleep (~8 hours). The condition of these participants ranges from normal to severe apnea. The recording is paired with sleep apnea labels for each minute marked by experts using various types of measurement. Since most wearable devices do not track electrocardiograms, only the heart rate information is extracted to form the training dataset.

## Model
Stratified cross-validation is performed to compare the prediction performance of Logistic Regression, Light GBM and Multilayer Perceptron using the same set of features. An end-to-end solution using pre-trained Convolutional Neural Networks and the wavelet spectrogram as input images is also tested. The overall accuracy is similar for these models (~80%), which means the features in the time and frequency domains convey most of the useful information. 

## Want to know more about this project?
This is a personal project for the Insight Data Science program.  

* [Heroku App](https://apnea-ecg.herokuapp.com/): Self-evaluation of sleep apnea using your own heart rate data.
* [Google Slides](https://docs.google.com/presentation/d/1WwZyvJ4VLjRcUPeKftsnVOTlXbZ1NYcIuLxvsKsN9ew/edit): Demo slides for feature engineering and performance evaluation. 

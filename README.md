# Sleep Apnea Detection
![alt text](https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/icon_cover.png "Sleep Apnea Detection")
Detect Sleep Apnea using heart rate measurements from wearable devices.

# Project Overview
Over 18 million Americans suffer from Sleep Apnea, which leads to sleep deprivation, hypertension, heart diseases, and even stroke. Many people either are not aware of it (thinking that they just snore a lot) or are concerned about the cost and effort of a sleep study. 

The purpose of this app is to detect sleep apnea using signals from the heart rate measurement, so that wearable devices can provide early warning and direct people to sleep studies for diagnosis and treatment. 

## How to detect sleep apnea from heart rate measurements?
![alt text](https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/Slide_HR_character.png=250x "Heart rate during apnea")
This figure shows a typical pattern of heart rate data during apnea. Due to the periodical breathing interruption, the heart rate fluctuates with a larger range at a specific frequency. As such, using features in the time and frequency domains, predictions can be made on whether the user is undergoing sleep apnea.

## How the prediction model is trained?
The prediction model is trained using a [dataset](https://physionet.org/content/apnea-ecg/1.0.0/) on the PhysioNet, which provides the electrocardiogram recording of 70 participants during their sleep (~8 hours). The condition of these participants ranges from normal to severe apnea. The recording is paired with sleep apnea labels for each minute marked by experts. 

Since most wearable devices do not track electrocardiogram, only the heart rate information is extracted to form the training dataset. A stratified cross-validation is performed to compare the prediction performance of Logistic Regression, Light GBM and Multilayer Perceptron using the same set of features. An end-to-end solution using pre-trained Convolutional Neural Networks and the wavelet spectrogram as input images is also tested. The overall accuracy is similar for these models (~80%), which means the features in the time and frequency domains convey most of the useful information. 

## Want to know more about this project?
This is a personal project for the Insight Data Science program.  

An [Oneline App](https://apnea-ecg.herokuapp.com/) is deployed on Heroku, where you can upload your own heart rate to evaluate sleep apnea, or using some sample data to see how it works. More information on feature engineering and performance evaluation can be found [here](https://docs.google.com/presentation/d/1WwZyvJ4VLjRcUPeKftsnVOTlXbZ1NYcIuLxvsKsN9ew/edit). 

# Portfolio
#### Data Science / Machine Learning Portfolio


# [Project 1: Pet Classification Model Using CNN.](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Deep_learning_pet_classification_project.ipynb)

   Build a CNN model that classifies the given pet images correctly into dog and 
cat images.

#### Tasks: 
   Import necessary packages and dataset, checking and cleaning the dataset, generating additional images from given set of images, converting the images to numpy array, reshaping 
the array, create, compile and optimize the model. 

#### Packages and Tools:
   Tensorflow with Keras, Pandas, Numpy, Sklearn, Matplotlib, ImageDataGenerator, img_to_array from Keras. Sequential Model with Convolutional Neural Network


# [Project 2: Mercedes-Benz Testing Time Consumption](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/ML_Mercedes-Benz_Project.ipynb)
   This is Regession problem where we have to predict the time taken for a Mercedes-Benz car for overall testing process before hitting the road. To ensure the safety and reliability of each and every unique car configuration before they hit the road, Daimler’s engineers have developed a robust testing system and they want to optimizing the speed of their testing system for so many possible feature combinations is complex and time-consuming without a powerful ML algorithmic approach. The dataset representing different permutations of Mercedes-Benz car features to predict the time it takes to pass testing.
   * The main challenge in the data is its huge dimention, they provided many possible permutations of features of the testing procedure with a output variable 'Y' as time consumption.
   * The features are not labeled with a specific test procedure, rather just mentioned with some 'X' values.
   * Also the data contains some categirical features and we should transform that to numeric before using for the prediction. We are using Sklearn's LabelEncoder to convert.
   * from sklearn.decomposition library we are using PCA to tackle the curse of dimensionality.
   * XGBootRegressor is used to predict the time taken for the a car in testbanch.
 
  
 
 # [Project 3: Churn_rate_modeling_ANN_Model](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Churn_rate_modeling_ANN_Model_.ipynb)
   Classifying the data check the churn_rate of a bank coustomer whether HE/She continue the bank services or not in the future. We are predicting the churn rate for a bank, whether a coustomer of a bank will leave the bank or not in thr future. Basically the bank provided the data of there coustomers with there bank balance, credit score ect. to predict whether the coustomer will stop his banking activities in the future or they will continue the service of the bank. We are classifying the data for the class 1 and 0 to ckeck the churn_rate using ANN.

# [Project 4: Amazon Review Sentiment Analysis](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Amazon%20Review%20Sentiment%20Analysis%20Prediction.ipynb)
   Sentiment analysis by classifying the text to check whether the coustomer is happy or not.Sentiment analysis using Amazon's Reviews given by the Amazon users on the products purchesed by the amazon coustomer. This data is real business data with all the info like name of coustomer, user_id of a coustomer and meny more with 1 to 5 rating and the reviews on specific products. This dataset consists of a few million Amazon customer reviews (Text) and star ratings (Score) for learning how to train Reviews.csv for sentiment analysis. The idea here is a dataset is more than a toy - real business data on a reasonable scale - but can be trained in minutes on a modest laptop.
* We are actually predict the sentiment of a coustomer by classifying the review text to class 1 or 0, 1 means Happy, 0 means Not happy
### Task:
   Sampling of this huge data for less time computation, Text preprossing and transformation, model building for sentiment analysis
### ML Algorithm:
   Naive Bayes Algotithm
 ### Tools:
   Pandas, Numpy, Matplotlib, NLTK, Sklearn

# [Project 5: FAKE NEWS Classification.](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/FAKE%20NEWS%20Classification.ipynb)
   A text classification problem to classify a news that is fake or real. The dataset contains the auther of the news article with a detail description of tha news and with the label of 1 or 0. The label 1 indicates the news is Fake and 0 indicates the news is real.
   * First we are cleaning the text data by removing stopwords and other entries like numbers and special characters (+-*<>!@#$%.,.ect).
   * We are stemmimg the words to there respective root words for better understing fo rthe algorithm.
   * We are converting the text data into numerical  vectors before feeding the data to the algorithm using TfidfVectorizer.  
   * Using Sklearn's Logistic regresion algorithm we are classifying the text.
#### * Packages and Tools: 
   Pandas, Numpy, Matplotlib, Seaborn.Sklearn, NKTL.
   
  
# [project 6: Heart Disease Prediction.](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Heart%20Disease%20Prediction.ipynb)
   This ia a classification problem which is to predict the whether a person have heart disease or not by using there health replort including the madication they are following for other diseases. The data has a target variable with class 1 or 0, 1 indicates the person is suffering form heart related disese and 0 indicates a person is not suffering form heart related disease. By this project we can predict a person may suffer from heart related diseases in the future or not, if he/she continues the same health care including the medications he/she consuming.
   * First we are doing some data preprocessing and data  visualization for better understanding of the data.
   * we are using the GridsearchCV to check which model and model parameters are suitable for the prediction.
   * we are trying with RandomForest, KNN and XGBoost models for the better prediction.


# [Project 7: Spotify Song Attributes](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Spotify%20Song%20Attributes.ipynb)
   A dataset of 2017 songs with attributes from Spotify. Each song is labeled "1" which mean people will like it and "0" for songs people don't like. I used this to data to see if I could build a classifier that could predict whether or not people would like a song. This data really doesnot contains any audio but it has information about 'acousticness','danceability','duration_ms','instrumentalness','loudness','song_title','artist', ect. to classify the song whether people will like it or not.
 * We are doing some data preprocessing and  data  visualization for better analysis of the data.
 * Some features are really skewed to left or right and we performaed some feature transformation techniques to remove the skewness of the data.
 * We did feature scaling before feeding the data to the model.

 

# [Project 8: Kaggle - Car Price Prediction](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Kaggle%20-%20Car%20Price%20Prediction%20.ipynb)
   This is Regession problem where we have to predict the price of a used car. The data used here is from kaggle and in the data we have details of the usd car like, 'Car_Name','Year','Kms_Driven','Owner', ect. to predict the cost of that used car. If we are buying a used car we must be carfull about the condition of the car and we need a expert advise about the car condition and the price demand of the owner. So this ML model will help to solve the basic confusion about buying the car.
  * Did some data preprocessing and data cleaning.
  * Visulizing the data for the better understanding of correlation betweeen the features and target variable.
  * Used RandomizedSearchCV to choose diffetent parameters of RandomForest algorithm for the best prediction.


# [Project 9: Advance House price prediction](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/Kaggle%20-%20Advance%20House%20price%20prediction.ipynb)
   This is Regession problem where we have to predict the price of a house. The data is taken from the kaggle and the data contains the features related to the land located area, sqfeet of the house, year of built, number of bed rooms and ect. to predict the peice of the house. If we have a data about the house and if we give tha data to our model then the model will predict the perice of the data. 
   * We have a good number of featueres i.e we have 54 features and we must clean and preprocess the data before using it.
   * We did feature scaling and feature selection for the better prediction.
   * We did some visualization to know the relationship b/w the independent and dependent feature.


# [Project 10: Comcast Telecom Consumer Complaints EDA](https://github.com/Kiranshankarbhat007/Data-Science-and-ML-Projects-/blob/master/ComCast%20.ipynb)

Comcast is an American global telecommunication company. The firm has be
en providing terrible customer service. They continue to fall short despite repeated promises 
to improve. Only last month (October 2016) the authority fined them a $2.3 million, after 
receiving over 1000 consumer complaints.
The existing database will serve as a repository of public customer complaints filed against 
Comcast. It will help to pin down what is wrong with Comcast's customer service.

#### Task: 
   Exploratory Data Analysis.

#### Packages and Tools: 
   Pandas, Numpy, Matplotlib, Seaborn.

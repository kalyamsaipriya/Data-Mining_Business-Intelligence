# Chapter-7
# K-Nearest Neighbors (k-NN) Method

# k-nearest-neighbors algorithm can be used
# for classification (of a categorical outcome)
# or prediction (of a numerical outcome).

# In this section, we focus on classification of a categorical outcome.
# For example, we will predict if the next person is a "buyer" or "non-buyer"
# as opposed to a numerical outcome (e.g. purchase amount) in regression.

# The "theme" or basic idea in k-nn is "you are who your friends are"
# or "you are who your neighbors are".
# If most of your neighbors' income is greater than $150k, probably your
# income level is also greater than $150k. 
# If most of your friends speak Spanish, you probably speak Spanish.
# and so on.

# The idea in k-nearest-neighbors methods is to identify k records in the training
# dataset that are similar to a new record that we wish to classify. We then use these
# similar (neighboring) records to classify the new record into a class, assigning the
# new record to the predominant class among these neighbors.

# Issues: 
# 1) How do you define the borders of the neighborhood? How many blocks, streets,
# houses are considered "in" the neighborhood?
# 2) How do you measure distance from you to your neighbors?
# 3) How do you determine the general characteristic of the neighborhood?
# 4) When we say "most of", does it mean 50% or more?

# K-NN is a nonparametric method because it does not involve estimation of parameters
# in an assumed function form, such as the linear form assumed in linear regression


# Procedure:
# 1) Assume outcome variable Y has two levels: Y1 and Y2.
# 2) A new record will be labeled as Y1 or Y2 based on its predictors, X1, X2, ...
# 3) Calculate the distance from the new record to the existing records (i.e. each neighbors)
# 4) Which ones have the shortest distance (i.e.closest neighbors)
# 5) Determine the borders of the neighborhood. i.e. how many closest records are considered "in" the
#    neighborhood. This is parameter k.
# 6) What is the general characteristics of these neighbors? Mostly Y1, or mostly Y2?
#    i.e. Count! What is the "majority vote"?
# 7) The new record is labeled as the majority characteristic (Y1 or Y2) of the neighbors.


# Distance calculation

# A central question is how to measure the distance between records based on
# their predictor values. The most popular measure of distance is the Euclidean
# distance. The Euclidean distance between two records, (x1, x2, ... xp) and
# (u1, u2,.. up) with p predictors is:

#  SQRT( (x1 - u1)^2 + (x2 - u2)^2 + ...  (xp - up)^2   )

# Although there are other distance measures, Euclidean is the most popular.

# NOTE: To equalize the scales that the various predictors may have, 
# predictors should first be standardized before computing a Euclidean
# distance. 
# Also note validation data and new records need to be standardized
# as well but the means and standard deviations used to standardize
# them should be those of the training data.

# Determining k

# The simplest case is k = 1, where we look for the record that is closest
# (the nearest neighbor) and classify the new record as belonging to
# the same class as its closest neighbor.

# The idea of the 1-nearest neighbor can be extended to k > 1 neighbors as
# follows:
# 1. Find the nearest k neighbors to the record to be classified.
# 2. Use a majority decision rule to classify the record, where the record is
#    classified as a member of the majority class of the k neighbors.

#----------- Example: Riding Mowers

# A riding-mower (tractor) manufacturer would like to find a way of classifying families
# in a city into those likely to purchase a riding mower and those not likely to
# buy one.  

# A pilot random sample is undertaken of 12 owners and 12 nonowners
# in the city. We first partition the data into
# training data (60% -> 14 households) and validation data (40% -> 10 households).

# install.packages("caret")
# install.packages("FNN", dependencies = TRUE)

library(caret)
library(FNN)

mower.df <- read.csv("RidingMowers.csv")
View(mower.df)

set.seed(2022)
train.rows <- sample(row.names(mower.df), 0.6*dim(mower.df)[1])  
valid.rows <- setdiff(row.names(mower.df), train.rows)  
train.data <- mower.df[train.rows, ]
valid.data <- mower.df[valid.rows, ]

## two new households in town!
new.df <- data.frame(Income = 60, Lot_Size = 20)

## scatter plot for training
plot(train.data$Lot_Size ~ train.data$Income)
# This plot does not convey information on who is buyer or not.

# The argument "pch" in plot() below, is for plotting a character
# 1 is for circle, 2 for triangle, 3 for plus sign, 4 is x

# owner -> o, nonowner -> x
plot(Lot_Size ~ Income, data=train.data, pch=ifelse(train.data$Ownership=="Owner", 1, 4))

# Now write the new household on the chart
text(new.df$Income, new.df$Lot_Size, "N")

# Now legend on the chart
 legend("topright", c("owner", "non-owner", "newhousehold"), pch = c('o', 'x', 'N'))

# Write row number next to each dot (pos = 4 means right of the dot)
text(train.data$Income, train.data$Lot_Size, rownames(train.data), pos=4)

# After visually inspecting the plot, 
# if we use a 1-NN classifier, we would classify the new household as an owner,
# because household ? is the nearest and it is an owner.

# If we use k = 3, the three nearest households are 4, 9, and 14, 
# as can be seen visually in the scatter plot.
# Two of these neighbors are owners of riding mowers, and one is a nonowner.
# The majority vote is therefore "owner" (2 out of 3), and the new
# household would be classified as an owner.

# Now lets do the same analysis using FNN package!

# Since income and size variables are in different magnitudes,
# we should normalize these two columns. 
# First, let's make a copy of the original datasets:
train.norm.df <- train.data
valid.norm.df <- valid.data

# standardize the datasets
# use preProcess() function from the caret package to normalize Income and Lot_Size.
# method = "center" subtracts the mean of the predictor's data
# from the predictor values (i.e, x - xbar) 
# while method = "scale" divides by the standard deviation.
# Just like z = (x - xbar)/s

library(caret)

norm.values <- preProcess(train.data, method=c("center", "scale"))
summary(norm.values)

# norm.values is like a placeholder holding important information about the normalized data
# In other words, it keeps mean and sd calculated from the training set.
# These mean and sd are then used to standardize all the remaining data.
# We do this using predict() function:

train.norm.df <- predict(norm.values, train.data)
valid.norm.df <- predict(norm.values, valid.data)

new.norm.df   <- predict(norm.values, new.df)

# use knn() to compute knn. 
# knn() is available in library FNN

library(FNN)
# knn( train = training dataset columns, ie. predictors, test = new data columns, 
#         cl=true classification column, i.e. the outcome, k=number of neighbors)

# Let's try 3 nearest neighbors, k=3

knn3 <- knn(train = train.norm.df[ , 1:2], test = new.norm.df, cl = train.norm.df[, 3], k = 3, prob = TRUE)

# cl = the actual value of the target variable in the training set
# prob = True means it will show the percent of the majority decision

knn3

# knn3 has all the information about classification

# show the classification result: owner or nonowner?
summary(knn3)   # Shows the prediction

# show neighbors' index in the training set
nb = attr(knn3,"nn.index")
train.rows[nb]   # shows three nearest neighbors

# show neighbor's distance in the training set
attr(knn3,"nn.dist")

# show the majority vote percent (ie. probability)
attr(knn3,"prob")

# show the entire information about the classification
knn3

# Another example:

new.df <- data.frame(Income = 50, Lot_Size = 15)
new.norm.df   <- predict(norm.values, new.df)
knn3 <- knn(train = train.norm.df[ , 1:2], test = new.norm.df, cl = train.norm.df[, 3], k = 3, prob = TRUE)
summary(knn3)
attr(knn3,"prob")
knn3

#------- How do we determine the best k value ---------------
# lower k -> possibility of overfitting
# higher k -> possibility of overgeneralization -> naive rule

# naive rule: the new record is classified as the majority characteristic
# in the dataset, ignoring predictor information. 

# Example: In a dataset of 1400 buyers and 1200 nonbuyers,
# the naive rule classification results in "buyer" due to buyer's majority
# It means, every new customer is classified as "buyer" 
# but this prediction is correct only 1400 out of (1400+1200) times
# Accuracy = 1400/(1400+1200)
# Can we beat this accuracy with our data mining models, like KNN?


# Try different k values developed in training data, then
# apply it on validation data and calculate
# accuracy. Keep the one with the most accurate k. 
# Use this k for new predictions.

# install.packages("e1071")
# Package e1071: Miscellaneous functions for statistics, probability and classification

library(e1071)
library(caret)   # short for Classification And REgression Training

#----- try k=1 on validation 
knn1 <- knn(train.norm.df[, 1:2], test = valid.norm.df[, 1:2], 
            cl = train.norm.df[, 3], k = 1)

# present the results in a nice dataframe combining all columns!
info1.df = as.data.frame(cbind(
  income=valid.data$Income,
  lot_size=valid.data$Lot_Size,
  actual=valid.data$Ownership,
  knn1
))

View(info1.df)

info1.df$knn1 = ifelse(info1.df$knn1=="1", "Nonowner","Owner")

View(info1.df)

# How many correct predictions?
# accuracy: 7 out 10 correct predictions 70%
# error rate: 3 out of 10 incorrect predictions  30%

#----- try k=3 on validation 
knn3 <- knn(train.norm.df[, 1:2], test = valid.norm.df[, 1:2], 
                cl = train.norm.df[, 3], k = 3)

# present the results in a nice dataframe combining all columns!
info3.df = as.data.frame(cbind(
  income=valid.data$Income,
  lot_size=valid.data$Lot_Size,
  actual=valid.data$Ownership,
  knn3
))

knn3
# last row shows order of the levels: Nonowner=1 , Owner=2

View(info3.df)

info3.df$knn3 = ifelse(info3.df$knn3=="1", "Nonowner","Owner")

View(info3.df)

#----- try k=5 on validation 
knn5 <- knn(train.norm.df[, 1:2], test = valid.norm.df[, 1:2], 
            cl = train.norm.df[, 3], k = 5)

# present the results in a nice dataframe combining all columns!
info5.df = as.data.frame(cbind(
  income=valid.data$Income,
  lot_size=valid.data$Lot_Size,
  actual=valid.data$Ownership,
  knn5
))

info5.df$knn5 = ifelse(info5.df$knn5=="1", "Nonowner","Owner")

View(info5.df)

# how many incorrect predictions?

table(info5.df$actual == info5.df$knn5)

# error rate = 2/(2+8)
# accuracy = 8/(2+8)

#---- Assessing the performance of the classification -----------

# confusionMatrix: A matrix showing what is predicted and what is actual

# confusionMatrix(pedicted, actual, positive = the class we are interested in)

#-------------------------------------------------
#                Actual Class (ie. Reference)
#                     C1         C2
# Predicted    C1     n11        n21
# Class        C2     n12        n22
#-------------------------------------------------

# n11 = number of C1 records classified correctly
# n12 = number of C1 records classified incorrectly as C2
# n21 = number of C2 records classified incorrectly as C1
# n22 = number of C2 records classified correctly

# Error Rate (ie. misclassification rate) = (n12 + n21)/(n11+n12+n21+n22)
# Accuracy = 1 - Error Rate = (n11 + n22)/(n11+n12+n21+n22)

# Example-1:
#                  Reference
#    Prediction   Nonowner  Owner
#     Nonowner      110      42
#     Owner         26       48

# Calculate error rate and accuracy!
accuracy = (110+48)/(110+48+42+26)
errorrate = 1 - accuracy


#--------- Sensitivity and Specificity ------------
# Sometimes it is more important to predict membership correctly 
# in one class than the other class. For example, predicting bankruptcy
# of a company seems to be a more significant task, 
# rather than predicting if it is in good shape.
# Say the important category for you is C1 (not C2)

# The "sensitivity" (also termed "recall") of a classifier is 
# its ability to detect the important class members correctly.

# Sensitivity = n11/(n11 + n12) = number of correctly identified C1 / total C1

# What is the sensitivity for detecting "owners"? Calculate using the previous example.

#----------------------------------------------------
#              Reference (Assuming positive = C1)
# Prediction          C1                  C2
#     C1        True Positive      False Positive
#     C2        False Negative     True Negative  
#----------------------------------------------------

# The "specificity" of a classifier is its ability to rule out C2 members correctly.
# specificity =  n22/(n21 +n22) = the percentage of C2 members classified correctly.

# In medical diagnosis, sensitivity is the ability of 
# a test to correctly identify those with the disease
# (true positive rate = true positives out of all positives),
# whereas test specificity is the ability of the test to correctly 
# identify those without the disease (true negative rate = true negatives out of all negatives).

# Let's get the confusion matrix for the example on the validation data
confusionMatrix(as.factor(info1.df$knn1), as.factor(valid.norm.df[, 3]), positive = "Owner")

# Accuracy    : 0.7 meaning there were 5 owners and 5 nonowners. We correctly predicted them
#               only 7 out of 10.
# Sensitivity : 0.6 meaning there were 5 owners, we correctly predicted only 3.  
# Specificity : 0.8 meaning there were 5 nonowners, we correctly predicted only 4.

#---------- Propensities ----------------
# The first step in most classification algorithms is to estimate the probability that
# a record belongs to each of the classes. These probabilities are also called propensities.
# Propensities are typically used either as an interim step for generating
# predicted class membership (classification), or for rank-ordering the records by
# their probability of belonging to a class of interest

# For example, a customer is mapped to a neighborhood with k=5. In this neighborhood,
# 3 out 5 are owners. Therefore, this customer's propensity to be an owner is 3/5.
# Since this value is greater than 0.5 (default cutoff in majority rule), we predict
# that this customer is an owner.
# However, this standard cutoff may not be desirable all the time. Sometimes,
# you may want to use a higher cutoff, sometimes a lower cutoff. 
# With higher cutoff, you are more cautious and imposing a higher standard for
# class membership. In other words, it might be more important to
# classify owners properly than nonowners. A misclassification here could be costly.
# For example, the company will send some gifts/promotional materials to really serious buyers only.
# On the other hand, lowering the cutoff, means many nonowners will pass incorrectly as owners.
# For example, you are trying to sell your product on the phone, and each call is not so expensive.
# If the call did not yield a sale, it is okay.

owner.df <- read.csv("ownerExample.csv")
View(owner.df)
owner.df

# Class column shows the actual class
# Probability shows more like probability of belonging a class

# let's try a few different cutoff values

confusionMatrix(as.factor(ifelse(owner.df$Probability>0.5, 'owner', 'nonowner')), 
                as.factor(owner.df$Class), positive = 'owner')
confusionMatrix(as.factor(ifelse(owner.df$Probability>0.25, 'owner', 'nonowner')), 
                as.factor(owner.df$Class), positive = 'owner')
confusionMatrix(as.factor(ifelse(owner.df$Probability>0.75, 'owner', 'nonowner')), 
                as.factor(owner.df$Class), positive = 'owner')


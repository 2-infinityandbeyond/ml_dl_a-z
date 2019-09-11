# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')


dataset$State = factor(dataset$State,
                       levels = c('California', 'Florida', 'New York'),
                       labels = c(1, 2, 3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
split = sample.split(dataset$Profit, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


regressor = lm(formula = Profit ~ R.D.Spend  ,
               data=dataset)
pred = predict(regressor,
               newdata = test_set)
summary(regressor)

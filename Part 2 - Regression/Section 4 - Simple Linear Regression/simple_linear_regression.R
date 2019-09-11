#simple linear regression
dataset = read.csv('Salary_Data.csv')

#spliting into test set and training set
library(caTools)
split= sample.split(dataset$Salary,SplitRatio = 2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

# #feature scaling
# training_set[,1:1]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])

regressor= lm(formula = Salary ~ YearsExperience,
              data = training_set)
pred = predict(regressor,newdata = test_set)
# visualisinf training set
library(ggplot2) 
ggplot()+
  geom_point(aes(x= training_set$YearsExperience,y=training_set$Salary),
             colour = 'red')+
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
            colour = 'blue')+
  ggtitle('salary vs years of experience(training set)')+
  xlab('Years of experience')+
  ylab('Salary')

# visualising test set
library(ggplot2) 
ggplot()+
  geom_point(aes(x= test_set$YearsExperience,y=test_set$Salary),
             colour = 'red')+
  geom_line(aes(x=test_set$YearsExperience,y=predict(regressor,newdata = test_set)),
            colour = 'blue')+
  ggtitle('salary vs years of experience(test set)')+
  xlab('Years of experience')+
  ylab('Salary')

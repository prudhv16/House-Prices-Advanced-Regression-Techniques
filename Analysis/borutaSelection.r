options(warn=-1)
#library(caret)
library(data.table)
library(Boruta)
library(plyr)
library(dplyr)
library(pROC)

set.seed(16)

trainData <- read.csv("train.csv",header = T,stringsAsFactors = F)
#THis data set looks like pretty clean and most of the missing values mean some thing so we 
#planned to not treat missing values.

#Since burrato will not be albe to handle NA's we convert it to MISSING string
features <- setdiff(names(trainData),c("Id","SalePrice"))
dataType <- sapply(features,function(x){class(trainData[[x]])})
#table(dataType)
#print(dataType)

#explanatoryattr <- setdiff(names(trainData),c("Id","SalePrice"))
classesData <- sapply(features,function(x){class(trainData[[x]])})

unique_classes = unique(classesData)

attrdatatypes <- lapply(unique_classes,function(x){names(classesData[classesData==x])})
names(attrdatatypes) <- unique_classes


Target <- trainData$SalePrice
trainData <- trainData[features]


for (x in attrdatatypes$integer){
  trainData[[x]][is.na(trainData[[x]])] <- -1
}

for (x in attrdatatypes$character){
  trainData[[x]][is.na(trainData[[x]])] <- 
}

borutaResults <- Boruta(trainData,Target,maxRuns = 100,doTrace = 0)

print (borutaResults)
getSelectedAttributes(borutaResults)
plot(borutaResults)

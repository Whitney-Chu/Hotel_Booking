## ----setup, include=FALSE-----------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## -----------------------------------------------------------------------------------------------------------
library(plyr)
library(tm)
library(wordcloud)
library(cluster)
library(text2vec)
library(stringr)
library(ggplot2)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(doSNOW)
library(tidyverse)
library(factoextra)


## -----------------------------------------------------------------------------------------------------------
data <- read.csv("Hotel.csv")
head(data)
summary(data)
dim(data)
str(data)


## -----------------------------------------------------------------------------------------------------------
names(data) <- c("Review_Number", "Text")
View(data)


## -----------------------------------------------------------------------------------------------------------
length(which(!complete.cases(data)))


## -----------------------------------------------------------------------------------------------------------
data$Review_Number <- as.factor(data$Review_Number)


## -----------------------------------------------------------------------------------------------------------
#character lengths
data$charLength <- nchar(data$Text)
summary(data$charLength)
hist(data$charLength)

#word length
data$wordlength <- sapply(strsplit(data$Text, " "), length)
data$wordlength
summary(data$wordlength)
hist(data$wordlength)


## -----------------------------------------------------------------------------------------------------------
nrow(data)
set.seed(1000)
rows <- c(1:nrow(data))
split <- sample(rows, size = (nrow(data)*0.70))
train <- data[split,]
test <- data[-split,]
nrow(train)
nrow(test)


## -----------------------------------------------------------------------------------------------------------
# Tokenize reviews
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, split_hyphens = TRUE)

# Take a look at specific review
train.tokens[[100]]


# make tokens Lower case 
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[100]]

#remove stopwords
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[100]]


# Perform stemming on the tokens.
#get root words
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[100]]



## -----------------------------------------------------------------------------------------------------------
# Create a document-feature matrix (dfm)).
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.dfm

# Transform to a matrix
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)

# Setup the feature data frame with labels.
train.tokens.df <- cbind(Review_Number = train$Review_Number, as.data.frame(train.tokens.matrix))
train.tokens.df[1:6,1:10]

# additional pre-processing
names(train.tokens.df)[c(100, 102)]

# Cleanup column name into reasonable format
names(train.tokens.df) <- make.names(names(train.tokens.df))

names(train.tokens.df)[c(100, 102)]

#10 most frequent words
tsf <- textstat_frequency(train.tokens.dfm)[1:10,]
ggplot(tsf, aes(feature,frequency)) + geom_bar(stat="identity")


## -----------------------------------------------------------------------------------------------------------
as.data.frame(dfm_tfidf(train.tokens.dfm))[,-1]
summed <- as.data.frame(colSums(as.data.frame(dfm_tfidf(train.tokens.dfm))[,-1]))
summed<- setNames(cbind(rownames(summed), summed, row.names = NULL), c("term","col_weight"))
summed
tfidx_order <- summed[order(summed$col_weight, decreasing = TRUE),][1:10,]
ggplot(tfidx_order, aes(term, col_weight)) + geom_bar(stat="identity")


## -----------------------------------------------------------------------------------------------------------
compare_tsf_tfidx<- data.frame(tsf[,1:2],tfidx_order)
compare_tsf_tfidx


## -----------------------------------------------------------------------------------------------------------
reviews <- readLines("TripAdvisorReviews.txt")


## -----------------------------------------------------------------------------------------------------------
cs <- VCorpus(VectorSource(reviews))
cs <- tm_map(cs,content_transformer(PlainTextDocument))
cs <- tm_map(cs,content_transformer(stripWhitespace))
cs <- tm_map(cs,content_transformer(removePunctuation))
cs <- tm_map(cs,content_transformer(tolower))




## -----------------------------------------------------------------------------------------------------------
#clean data
reviews_clean <- sapply(cs,function(t) t$content)
reviews_clean[1]


## -----------------------------------------------------------------------------------------------------------
pw <- readLines("positive-words.txt")
nw <- readLines("negative-words.txt")


## -----------------------------------------------------------------------------------------------------------
sentimentScore <- function(Sentence,pos_words,neg_words) {
  
  word_splits <- str_split(Sentence,"\\s+")
  words <- unlist(word_splits)
  
  pos_matches <- match(pos_words,words)
  neg_matches <- match(neg_words,words)
  
  pos_matches <- !is.na(pos_matches)
  neg_matches <- !is.na(neg_matches)
  
  score <- sum(pos_matches)-sum(neg_matches)
  return(score)
  }


## -----------------------------------------------------------------------------------------------------------
sentimentScore(reviews_clean[1],pw,nw)

scores <- lapply(reviews_clean,function(x) sentimentScore(x,pw,nw))


## -----------------------------------------------------------------------------------------------------------
s.data <- read.csv("TripAdvisorReviewsY.csv")
head(s.data)
s.data$scores <- as.numeric(scores)
head(s.data)


## -----------------------------------------------------------------------------------------------------------
s.data$y <- s.data$sentiment==1
s.data


## -----------------------------------------------------------------------------------------------------------
fit1 <- glm(y~scores,family="binomial",data=s.data)
summary(fit1)


## -----------------------------------------------------------------------------------------------------------
fit1$fitted
fit1$fitted[1:4]
cut <-0.5


## -----------------------------------------------------------------------------------------------------------
predictedSentiment <- fit1$fitted>cut
Error <- mean(predictedSentiment!=s.data$y)
Error


## -----------------------------------------------------------------------------------------------------------
new <- read.csv("Hotel.csv")


## -----------------------------------------------------------------------------------------------------------
#consider as vector since its text
cs <- VCorpus(VectorSource(new$text))
cs <- tm_map(cs,content_transformer(PlainTextDocument))
cs <- tm_map(cs,content_transformer(stripWhitespace))
cs <- tm_map(cs,content_transformer(removePunctuation))
cs <- tm_map(cs,content_transformer(tolower))



## -----------------------------------------------------------------------------------------------------------
reviews_clean1 <- sapply(cs,function(t) t$content)
reviews_clean1[127]
#sentimentScore(reviews_clean1[127],pw,nw) #check


## -----------------------------------------------------------------------------------------------------------
scores <- lapply(reviews_clean1,function(x) sentimentScore(x,pw,nw))

new$Scores <- as.numeric(scores)
new$text <- reviews_clean1


## -----------------------------------------------------------------------------------------------------------
k2 <- kmeans(new$Scores, centers = 2, nstart = 15)

new$clusters<- k2$cluster
new

#cluster 1
C1 <- new[(new$clusters == 1),]
mean(C1$Scores)
length(new$clusters[new$clusters == 1])

#cluster 2
C2 <- new[(new$clusters == 2),]
mean(C2$Scores)
length(new$clusters[new$clusters == 2])


## -----------------------------------------------------------------------------------------------------------
d<- dist(new$Score, method = 'euclidean')
clustering <-hclust(d, method = 'ward')
plot(clustering)
rect.hclust(clustering, k = 2, border = 2:5)


## -----------------------------------------------------------------------------------------------------------
data[2]
#clean up
cs <- VCorpus(VectorSource(data$Text))
cs <- tm_map(cs,content_transformer(PlainTextDocument))
cs <- tm_map(cs,content_transformer(stripWhitespace))
cs <- tm_map(cs,content_transformer(removePunctuation))
cs <- tm_map(cs,content_transformer(tolower))

#sentiment scores
reviews_clean3 <- sapply(cs,function(t) t$content)
scores <- lapply(reviews_clean3,function(x) sentimentScore(x,pw,nw))
wq<-t(as.data.frame(scores))
wq<-as.data.frame(wq)
wq$V1
wq<-setNames(cbind(rownames(wq), wq, row.names = NULL), c("Rev", "Score"))
wq$Score<-as.numeric(wq$Score)

#top 10 hotels
wq[order(wq$Score,decreasing = TRUE),][1:10,]



## -----------------------------------------------------------------------------------------------------------
#train.tokens[[100]]
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
#train.tokens[[100]]
#train.tokens = sapply(strsplit(train.tokens, "_"), function(x) x[3])

# Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
newdf <- train.tokens.dfm[,105:413]
newdf


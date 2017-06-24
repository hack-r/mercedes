pacman::p_load(arules, arulesViz, vcd)

sample <- fread("sample_submission.csv")
train  <- fread("train.csv")
test   <- fread("test.csv")


cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}



# PLS ---------------------------------------------------------------------
sample <- fread("sample_submission.csv")
train  <- fread("train.csv")
test   <- fread("test.csv")

# Fix Categorical
train.id <- train$ID
test.id  <- test$ID
labels   <- as.numeric(train$y)
train$ID <- NULL
train$y  <- NULL
test$ID  <- NULL


train$X0 <- ifelse(train$X0 %in% intersect(unique(train$X0), unique(test$X0)),train$X0, NA)
test$X0  <- ifelse(test$X0 %in% intersect(unique(train$X0), unique(test$X0)),test$X0, NA)
train$X0 <- ifelse(is.na(train$X0), "other", train$X0)
test$X0  <- ifelse(is.na(test$X0), "other", test$X0)

train$X1 <- ifelse(train$X1 %in% intersect(unique(train$X1), unique(test$X1)),train$X1, NA)
test$X1  <- ifelse(test$X1 %in% intersect(unique(train$X1), unique(test$X1)),test$X1, NA)
train$X1 <- ifelse(is.na(train$X1), "other", train$X1)
test$X1  <- ifelse(is.na(test$X1), "other", test$X1)

train$X2 <- ifelse(train$X2 %in% intersect(unique(train$X2), unique(test$X2)),train$X2, NA)
test$X2  <- ifelse(test$X2 %in% intersect(unique(train$X2), unique(test$X2)),test$X2, NA)
train$X2 <- ifelse(is.na(train$X2), "other", train$X2)
test$X2  <- ifelse(is.na(test$X2), "other", test$X2)

train$X3 <- ifelse(train$X3 %in% intersect(unique(train$X3), unique(test$X3)),train$X3, NA)
test$X3  <- ifelse(test$X3 %in% intersect(unique(train$X3), unique(test$X3)),test$X3, NA)
train$X3 <- ifelse(is.na(train$X3), "other", train$X3)
test$X3  <- ifelse(is.na(test$X3), "other", test$X3)

train$X4 <- ifelse(train$X4 %in% intersect(unique(train$X4), unique(test$X4)),train$X4, NA)
test$X4  <- ifelse(test$X4 %in% intersect(unique(train$X4), unique(test$X4)),test$X4, NA)
train$X4 <- ifelse(is.na(train$X4), "other", train$X4)
test$X4  <- ifelse(is.na(test$X4), "other", test$X4)

train$X5 <- ifelse(train$X5 %in% intersect(unique(train$X5), unique(test$X5)),train$X5, NA)
test$X5  <- ifelse(test$X5 %in% intersect(unique(train$X5), unique(test$X5)),test$X5, NA)
train$X5 <- ifelse(is.na(train$X5), "other", train$X5)
test$X5  <- ifelse(is.na(test$X5), "other", test$X5)

train$X6 <- ifelse(train$X6 %in% intersect(unique(train$X6), unique(test$X6)),train$X6, NA)
test$X6  <- ifelse(test$X6 %in% intersect(unique(train$X6), unique(test$X6)),test$X6, NA)
train$X6 <- ifelse(is.na(train$X6), "other", train$X6)
test$X6  <- ifelse(is.na(test$X6), "other", test$X6)

train$X8 <- ifelse(train$X8 %in% intersect(unique(train$X8), unique(test$X8)),train$X8, NA)
test$X8  <- ifelse(test$X8 %in% intersect(unique(train$X8), unique(test$X8)),test$X8, NA)
train$X8 <- ifelse(is.na(train$X8), "other", train$X8)
test$X8  <- ifelse(is.na(test$X8), "other", test$X8)

#train[,1:8] <- apply(train[,1:8],2,as.factor)
#test[,1:8]  <- apply(test[,1:8],2,as.factor)
train <- as.data.frame(train)
test  <- as.data.frame(test)

for(i in 1:8){
  train[,i] <- as.factor(train[,i])
  test[,i]  <- as.factor(test[,i])
}

train$y <- labels

# Compile cross-validation settings
set.seed(100)
myfolds <- createMultiFolds(train$y, k = 5, times = 10)
control <- trainControl("repeatedcv", index = myfolds, selectionFunction = "oneSE")

# Train PLS model
mod1 <- caret::train(y ~ ., data = train,
                     method = "pls",
                     metric = "Rsquared",
                     tuneLength = 20,
                     trControl = control,
                     preProc = c("zv","center","scale"))

# Check CV profile
plot(mod1)
summary(mod1) # 5comp = 59.24

# Predict
pred <- predict(mod1, as.data.frame(test)) #, type = "response"
plot(density(pred))
lines(density(train$y),col="red")

sample$y <- pred
fwrite(sample,"..//output//pls.csv") # Public LB: 0.50923

# PCA-DA
# mod2 <- caret::train(y ~ ., data = train,
#                      method = "lda",
#                      metric = "Accuracy",
#                      trControl = control,
#                      preProc = c("zv","center","scale","pca"))
#
# pred2 <- predict(mod2, as.data.frame(test)) #, type = "response"

# RF
set.seed(100)
myfolds <- createMultiFolds(train$y, k = 5, times = 10)
control <- trainControl("repeatedcv", index = myfolds, selectionFunction = "oneSE")

mod3 <- caret::train(y ~ ., data = train,
              method = "ranger",
              metric = "Rsquared",
              trControl = control,
              tuneGrid = data.frame(mtry = seq(10)), #seq(10,.5*ncol(train),length.out = 6)
              preProc = c("zv","center","scale"))
plot(mod3)
mod3 # mtry 10 = 0.5440910... LB = 0.54440  # STABLE CV!?!?!?!
pred3 <- predict(mod3, as.data.frame(test)) #, type = "response"
plot(density(pred3))

sample$y <- pred3
fwrite(sample, "..//output//ranger.csv")

# Compile models and compare performance
models <- resamples(list("PLS-DA" = mod1, "PCA-DA" = mod2, "RF" = mod3))
bwplot(models, metric = "Accuracy")

plot(varImp(mod1), 10, main = "PLS-DA")
#plot(varImp(mod2), 10, main = "PCA-DA")

# PCR
mod4  <- pcr(y~., data = train, validation = "CV", ncomp=40) #, scale = TRUE
summary(mod4)
pred4 <- predict(mod4, as.data.frame(test))
plot(density(pred4)) # uh...

plot(density(train$y, bw = .2), xlim=c(72,160),ylim=c(0,.3))
lines(density(questionable3$y, bw = .2), xlim=c(72,160), col="red")
lines(density(pred, bw = .2), xlim=c(72,160), col="green")
lines(density(pred3, bw = .2), xlim=c(72,160), col="blue")

# Ranger RF with Latent
train_latent <- fread("train_latent.csv")
test_latent  <- fread("test_latent.csv")

mod5 <- caret::train(y ~ ., data = train_latent,
                     method = "ranger",
                     metric = "Rsquared",
                     trControl = control,
                     tuneGrid = data.frame(mtry = seq(10,.5*ncol(train_latent),length.out = 6)),
                     preProc = c("zv","center","scale"))
plot(mod5)
mod5 # mtry 10 =0.5332928... worse than without latent
pred5 <- predict(mod5, as.data.frame(test_latent)) #, type = "response"
plot(density(pred5))

sample$y <- pred5
fwrite(sample, "..//output//ranger_latent.csv")

# Ranger RF with extra non-latent features
train$flag_sum <- rowSums(train[,9:(ncol(train)-1)])
test$flag_sum  <- rowSums(test[,9:(ncol(test))])

consec <- function(ex){
  r<-rle(ex)
  max(r$length[r$values == 1])
}

train$consec <- apply(train,1,consec)
test$consec  <- apply(test,1,consec)

mod6 <- caret::train(y ~ ., data = train,
                     method = "ranger",
                     metric = "Rsquared",
                     trControl = control,
                     tuneGrid = data.frame(mtry = seq(10,.5*ncol(train),length.out = 6)),
                     preProc = c("zv","center","scale"))
plot(mod6)
mod6 # mtry 10 = 0.5438294
pred6 <- predict(mod6, as.data.frame(test)) #, type = "response"
plot(density(pred6))

sample$y <- pred6
fwrite(sample, "..//output//ranger_extra.csv")

# RF again
set.seed(100)
myfolds <- createMultiFolds(train$y, k = 5, times = 10)
control <- trainControl("repeatedcv", index = myfolds, selectionFunction = "oneSE")

#tmp <- train[,!colnames(train) %in% c("y")]
#tmp <- tmp[,9:ncol(tmp)]
#train$row_means  <- rowMeans(tmp)
#train$row_means2 <- log(train$row_means)
#train$x5x2       <- paste0(train$X5, train$X3)

mod7 <- caret::train(y ~ ., data = train,
                     method = "ranger",
                     metric = "Rsquared",
                     trControl = control,
                     #mtry = 45.9,
                     tuneGrid = data.frame(mtry = c(44)),
                     preProc = c("zv","center","scale"))
plot(mod7)
mod7 # 0.5481063
pred7 <- predict(mod7, as.data.frame(test)) #, type = "response"
plot(density(pred7))

sample$y <- pred7
fwrite(sample, "..//output//ranger_mod7.csv")

# Resampling that didnt work ----------------------------------------------

#library(keras)
#install_tensorflow()

tmp <- data.frame()
for(i in unique(test$X0)){
  if(i %in% unique(train$X0)){
    #cat(i)
    tmp <- rbind(tmp,sample_n(train[train$X0==i,],length(test$X0[test$X0==i]),replace=T))
  } else{
    cat(i, "is MISSING in train \n")
  }
}

tmp2 <- sample_n(train, nrow(test)-nrow(tmp))

train2 <- rbind(tmp,tmp2)

fwrite(train2, "..//input//train_v2.csv")


# arules ------------------------------------------------------------------
assoc(~ X0 + X1 + X2 + X3 + X4 + X5 + X6 + X8, data=train, shade=TRUE)

assoc(~ X0 + X1, data=train, shade=TRUE)
assoc(~ X0 + X1, data=train[2:4,], shade=TRUE, abbreviate_labs=6)
assoc(~ X0 + X2, data=train[2:8,], shade=TRUE, abbreviate_labs=6)
assoc(~ X0 + X3, data=train, shade=TRUE)

assoc(~ X10 + X12, data=train, shade=TRUE, abbreviate_labs=6)
table(train$X10, train$X12) # never both 1
table(test$X10, test$X12)   # never both 1

# discretize(iris$Sepal.Length, method = "frequency", 3)

for(c in colnames(train[,3:ncol(train)])) train[[c]] <- as.logical(train[[c]])
trans <- as(train[,3:ncol(train)], "transactions")

summary(trans)
# most frequent items:
#   X205     X74    X111    X361    X229 (Other)
# 4208    4206    4103    4066    4041  223596

itemFrequencyPlot(trans, topN=50,  cex.names=.5)

d <- dissimilarity(sample(trans, 4000), method = "phi", which = "items")
d[is.na(d)] <- 1 # get rid of missing values

pdf(file="similarity.pdf", width=25)
plot(hclust(d), cex=.5)
dev.off()



itemsets <- apriori(trans, parameter = list(target = "frequent",
                                            supp=500/nrow(trans), minlen = 2, maxlen=4))
summary(itemsets)

quality(itemsets)$lift <- interestMeasure(itemsets, measure="lift", trans = trans)
inspect(head(sort(itemsets, by = "lift"), n=10))

plot(head(sort(itemsets, by = "lift"), n=50), method = "graph", control=list(cex=.8)) # 3 groups...

itemsets <- sort(itemsets, by = "lift")
plot(head(itemsets, n=50), method = "graph", control=list(cex=.8))

r <- apriori(trans, parameter = list(supp=0.001, maxlen=4))
inspect(head(sort(r, by="lift"), n=10))

r <- apriori(trans, parameter = list(supp=100/nrow(trans), maxlen=4))
inspect(head(sort(r, by="lift"), n=10))
rr <- as.data.frame(inspect(head(sort(r, by="lift"), n=100)))

# X194, X187, X85, X283, X154, X374, X321
# X50, X129, X49, X263, X137, X324, X70, X361, X205, X58, X136, X74
# X161, X202, X45, X377, X356, X186, X362, X334, X133

train <- as.data.frame(train)
tmp   <- train[,colnames(train) %in%c("X194", "X187", "X85", "X283", "X154", "X374", "X321")]
head(tmp)

tmp2 <- train$X194 + train$X187+ train$X85+ train$X283+ train$X154+ train$X374+ train$X321
cor(train$y,tmp2)
cor(train$y,train$X194)
cor(train$y,train$X187)
cor(train$y,train$X85)

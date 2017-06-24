
# Feats to try ------------------------------------------------------------
# To try:

## Create neutonian mined model (train)
train$X5_n <- ifelse(train$X5=="n",1,0)
train$X5_w <- ifelse(train$X5=="w",1,0)
y = 66.5386547916928 + train$X5_n + 25.4051960894959*train$X263 + 7.89681427726105*train$X47 + 4.84424759927184*train$X222 + (1.39519609054033 + train$X119)/(0.123227966633056 + train$X127) - train$X5_w
summary(y)

## in sample R2
rsq <- function (x, y) cor(x, y) ^ 2
rsq(y, train$y) # 0.5716891

## OLS version with same inputs
#m <- lm(y~train$X5_n + train$X263 + train$X47 + train$X222 + I(1.39519609054033 + train$X119)/I(0.123227966633056 + train$X127) - train$X5_w, data=train)
m <- lm(y~train$X5_n + train$X263 + train$X47 + train$X222 + I((1.39519609054033 + train$X119)/(0.123227966633056 + train$X127)) - train$X5_w, data=train)

summary(m) # Multiple R-squared:  0.5707,	Adjusted R-squared:  0.5702
cor(predict(m), train$y)^2  #0.570726

## Create neutonian mined model (test)
test$X5_n <- ifelse(test$X5=="n",1,0)
test$X5_w <- ifelse(test$X5=="w",1,0)
y = 66.5386547916928 + test$X5_n + 25.4051960894959*test$X263 + 7.89681427726105*test$X47 + 4.84424759927184*test$X222 + (1.39519609054033 + test$X119)/(0.123227966633056 + test$X127) - test$X5_w
summary(y)
plot(density(y))
y_ols <- predict(m, test)
lines(density(y_ols),col="red")
lines(density(train$y), col = "blue")

## write out
sample$y <- y
fwrite(sample, "..//output//neuton2.csv") # LB 0.55

# Tried and worked:
## arules_group variables increased in-fold (10 fold) R2 by 0.0024439 (.00017 for xgb 5 fold)
train$arules_group1 <- train$X194 + train$X187 + train$X85 + train$X283 + train$X154 + train$X374 + train$X321
train$arules_group2 <- train$X50  + train$X129 + train$X49 + train$X263 + train$X137 + train$X324 + train$X70 + train$X361 + train$X205 + train$X58 + train$X136 + train$X74
train$arules_group3 <- train$X161 + train$X202 + train$X45 + train$X377 + train$X356 + train$X186 + train$X362 + train$X334 + train$X133


# Tried and failed:
# train$X5_n <- ifelse(train$X5=="n",1,0)
# y = 66.6103926037201 + train$X5_n + 25.1934259991532*train$X263 + 7.88386830060197*train$X47 + 4.79113310156974*train$X222 + ((1.38268096139112 + train$X119)/(0.121327993859323 + train$X127))
# summary(y)
#
# rsq <- function (x, y) cor(x, y) ^ 2
# rsq(y, train$y)
#
# m <- lm(y~X118, data=train)
#
# cor(predict(m),train$y)^2
#
# test$X5_n <- ifelse(test$X5=="n",1,0)
# y = 66.6103926037201 + test$X5_n + 25.1934259991532*test$X263 + 7.88386830060197*test$X47 + 4.79113310156974*test$X222 + ((1.38268096139112 + test$X119)/(0.121327993859323 + test$X127))



# cn <- colnames(train)[colnames(train) %in% colnames(test)]
# cn <- cn[10:length(cn)]
# df <- as.data.frame(train)
# train$pattern <- do.call(paste, c(df[cn], sep = ""))
# df            <- as.data.frame(test)
# test$pattern  <- do.call(paste, c(df[cn], sep = ""))
# train$pattern <- ifelse(train$pattern %in% test$pattern,train$pattern,"other")
# test$pattern  <- ifelse(test$pattern %in% train$pattern,test$pattern,"other")
# train$pattern <- as.factor(train$pattern)
# test$pattern  <- as.factor(test$pattern)

# train$flag_sum <- rowSums(train[,9:(ncol(train)-1)])
# test$flag_sum  <- rowSums(test[,9:(ncol(test))])
#
# consec <- function(ex){
#   r<-rle(ex)
#   max(r$length[r$values == 1])
# }
#
# train$consec <- apply(train,1,consec)
# test$consec  <- apply(test,1,consec)


# Model to test feats -----------------------------------------------------
set.seed(100)
myfolds <- createMultiFolds(train$y, k = 5, times = 1)
control <- trainControl("repeatedcv", index = myfolds, selectionFunction = "oneSE")

mod3    <- caret::train(y ~ ., data = train,
                             method = "ranger",
                             metric = "Rsquared",
                             trControl = control,
                             tuneGrid = data.frame(mtry = 10), #0.5414098
                             preProc = c("zv","center","scale"))

plot(mod3)
mod3 # mtry 10 = 0.5440910... LB = 0.54440  # STABLE CV!?!?!?!
pred3 <- predict(mod3, as.data.frame(test)) #, type = "response"
plot(density(pred3))

sample$y <- pred3


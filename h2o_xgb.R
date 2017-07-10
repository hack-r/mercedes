# Libraries and Options ---------------------------------------------------
options(scipen=20)
setwd("..//input")
pacman::p_load(car, caret, data.table, dplyr, lightgbm,
               Matrix, mctest, Metrics, mlbench, MLmetrics, mlr, RRF,
               stringr, sqldf, xgboost)
pacman::p_load(bit64, caret, data.table, dplyr, extraTrees, h2o, h2oEnsemble,
               lubridate, nnet, RODBC, sendmailR, sas7bdat.parso, seasonal,
               sqldf,  sqlutils, xlsx)

system("java -Xmx20g -jar E://Jason//h2o-3.13.0.3943//h2o.jar", wait = F)
h2o.init(nthreads = -1)


# Load data ---------------------------------------------------------------
train      <- fread("train_model0.csv")
test       <- fread("test_model0.csv")
sample     <- fread("sample_submission.csv")
train_raw  <- fread("train.csv")
test_raw   <- fread("test.csv")

train$V1 <- NULL
train$ID <- NULL

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

# H2O XGB ------------------------------------------------------------
print("Load datasets")
val <- sample_n(data.frame(index=1:4209),400)

trainHex <- as.h2o(train[!val$index,])
testHex  <- as.h2o(test)
valHex   <- as.h2o(train[val$index,])


print("Train xgboost model")
xgb <- h2o.xgboost(x = setdiff(colnames(trainHex), c("y","ID"))
                   ,y = "y"
                   ,training_frame = trainHex
                   ,validation_frame = valHex
                   #,model_id = "xgb_model_1"
                   ,stopping_rounds = 3
                   ,stopping_metric = "AUTO"
                   ,distribution = "AUTO"
                   #,score_tree_interval = 10
                   ,learn_rate=0.1
                   ,ntrees=100
                   #,subsample = 0.75
                   #,colsample_bytree = 0.75
                   #,tree_method = "hist"
                   #,grow_policy = "lossguide"
                   #,booster = "gbtree"
                   #,gamma = 0.0
)

print("Make predictions")
sPreds <- as.data.table(h2o.predict(xgb, test.hex))
sPreds <- data.table(order_id=test$order_id, product_id=test$product_id, testPreds=sPreds$C3)
testPreds <- sPreds[,.(products=paste0(product_id[testPreds>0.21], collapse=" ")), by=order_id]
set(testPreds, which(testPreds[["products"]]==""), "products", "None")
print("Create submission file")
fwrite(testPreds, "submission.csv")

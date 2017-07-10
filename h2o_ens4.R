# Libraries and Options ---------------------------------------------------
options(scipen=20)
setwd("..//input")
pacman::p_load(car, caret, data.table, dplyr, lightgbm,
               Matrix, mctest, Metrics, mlbench, MLmetrics, mlr, RRF,
               stringr, sqldf, xgboost)
pacman::p_load(bit64, caret, data.table, dplyr, extraTrees, h2o, h2oEnsemble,
               lubridate, nnet, RODBC, sendmailR, sas7bdat.parso, seasonal,
               sqldf,  sqlutils, xlsx)

system("java -Xmx20g -jar E://Jason//h2o-3.10.5.2//h2o.jar", wait = F)
#system("java -Xmx20g -jar E://Jason//h2o-3.13.0.3943//h2o.jar", wait = F)
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

# H2O ens ------------------------------------------------------------
print("Load datasets")
val <- sample_n(data.frame(index=1:4209),400)

trainHex <- as.h2o(train[!val$index,])
testHex  <- as.h2o(test)
valHex   <- as.h2o(train[val$index,])

# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = setdiff(colnames(trainHex), c("ID", "y")),
                  y = "y",
                  training_frame = trainHex,
                  validation_frame = valHex,
                  distribution = "AUTO",
                  max_depth = 4,
                  min_rows = 2,
                  learn_rate = 0.02,
                  nfolds = nfolds,
                  fold_assignment = "Modulo",
                  keep_cross_validation_predictions = TRUE,
                  seed = 1)

# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = setdiff(colnames(trainHex), c("ID", "y")),
                          y = "y",
                          training_frame = trainHex,
                          validation_frame = valHex,
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1,
                          ntrees = 200)

# Train & Cross-validate a DNN
my_dl <- h2o.deeplearning(x = setdiff(colnames(trainHex), c("ID", "y")),
                          y = "y",
                          training_frame = trainHex,
                          validation_frame = valHex,
                          l1 = 0.001,
                          l2 = 0.001,
                          hidden = c(200, 200, 200),
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

my_dl2 <- h2o.deeplearning(x = setdiff(colnames(trainHex), c("ID", "y")),
                          y = "y",
                          training_frame = trainHex,
                          validation_frame = valHex,
                          l1 = 0.002,
                          l2 = .001,
                          hidden = c(200, 200, 200),
                          nfolds = nfolds,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE,
                          seed = 1)

my_dl3 <- h2o.deeplearning(x = setdiff(colnames(trainHex), c("ID", "y")),
                           y = "y",
                           training_frame = trainHex,
                           validation_frame = valHex,
                           l1 = 0,
                           l2 = 0.9,
                           hidden = c(200, 200),
                           nfolds = nfolds,
                           fold_assignment = "Modulo",
                           keep_cross_validation_predictions = TRUE,
                           seed = 1)

# Train a stacked ensemble using the H2O and XGBoost models from above
base_models <- list(my_gbm@model_id, my_rf@model_id, my_dl@model_id, my_dl2@model_id,
                    my_dl3@model_id)

ensemble <- h2o.stackedEnsemble(x = setdiff(colnames(trainHex), c("y", "ID", "X207")),
                                y = "y",
                                training_frame = trainHex,
                                validation_frame = valHex,
                                base_models = base_models)

v <- as.data.frame(valHex$y)
v <- v$y
p <- h2o.predict(ensemble, valHex)
p <- as.data.frame(p)
p <- p$predict

cor(p, v)^2 # 0.6183782

# scoring
score <- h2o.predict(ensemble, testHex)
score <- as.data.frame(score)

sample$y <- as.numeric(as.character(score$predict))

saveRDS(sample, "..//output//h2o_ens4b.RDS")
fwrite(sample, "..//output//h2o_ens4b.csv")

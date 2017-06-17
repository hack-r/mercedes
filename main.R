
# Libraries and Options ---------------------------------------------------
options(scipen=20)
setwd("..//input")
pacman::p_load(car, caret, data.table, dplyr, lightgbm,
               Matrix, mctest, Metrics, mlbench, MLmetrics, mlr, RRF,
               stringr, sqldf, xgboost)

# Stack -------------------------------------------------------------------
sample <- fread("sample_submission.csv")

my_kernel_dart <- fread("jason.csv")
my_priv_dart   <- fread("xgb_12pca_12ica_1grp_8srp_py_500tr_dart.csv")
#questionable   <- fread("..//baseLine.csv")
questionable2  <- fread("stacked-models.csv")
questionable3  <- fread("questionable3.csv")
sample$y <- my_kernel_dart$y * .38 + my_priv_dart$y * .10 + questionable3$y * .52

fwrite(sample, "..//output//stack_mykerneldart38_cvdart10_questionable3_52.csv")

# stack_mykerneldart38_cvdart05_questionable3_57  LB 0.56859
# stack_mykerneldart38_cvdart10_questionable3_55  LB 0.56859
# stack_mykerneldart38_cvdart10_questionable3_52  LB 0.56859
# stack_mykerneldart40_cvdart05_questionable3_55  LB 0.56859
# stack_mykerneldart35_cvdart10_questionable3_55  LB 0.56857
# stack_mykerneldart34_cvdart09_questionable3_57  LB 0.56857
# stack_mykerneldart38_cvdart02_questionable3_60  LB 0.56856
# stack_mykerneldart33_cvdart02_questionable3_65  LB 0.56854
# stack_mykerneldart40_cvdart0_questionable3_60   LB 0.56852
# stack_mykerneldart28_cvdart02_questionable3_70  LB 0.56851
# stack_mykerneldart28_cvdart02_questionable2_70  LB 0.56848
# stack_mykerneldart20_cvdart02_questionable3_78  LB 0.56841
# stack_mykerneldart28_cvdart02_questionable70    LB 0.56815
# stack_mykerneldart38_cvdart02_questionable60    LB 0.56814
# stack_mykerneldart08_cvdart02_questionable2_90  LB 0.56813
# stack_mykerneldart48_cvdart02_questionable50    LB 0.56808
# stack_mykerneldart005_cvdart005_questionable3_99   0.56798
# 100% questionable2                                 0.56793

# Functions ---------------------------------------------------------------

R2gauss<- function(y,model){
  moy<-mean(y)
  N<- length(y)
  p<-length(model$coefficients)-1
  SSres<- sum((y-predict(model))^2)
  SStot<-sum((y-moy)^2)
  R2<-1-(SSres/SStot)
  Rajust<-1-(((1-R2)*(N-1))/(N-p-1))
  return(data.frame(R2,Rajust,SSres,SStot))
}

# Load Raw Data -----------------------------------------------------------
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

train$tc <- 1
test$tc  <- 0

dat <- rbind(train,test)

x0 <- data.frame(x0_code=seq(1:length(unique(dat$X0))),X0=unique(dat$X0))
x1 <- data.frame(x1_code=seq(1:length(unique(dat$X1))),X1=unique(dat$X1))
x2 <- data.frame(x2_code=seq(1:length(unique(dat$X2))),X2=unique(dat$X2))
x3 <- data.frame(x3_code=seq(1:length(unique(dat$X3))),X3=unique(dat$X3))
x4 <- data.frame(x4_code=seq(1:length(unique(dat$X4))),X4=unique(dat$X4))
x5 <- data.frame(x5_code=seq(1:length(unique(dat$X5))),X5=unique(dat$X5))
x6 <- data.frame(x6_code=seq(1:length(unique(dat$X6))),X6=unique(dat$X6))
x8 <- data.frame(x8_code=seq(1:length(unique(dat$X8))),X8=unique(dat$X8))

dat <- merge(dat,x0,by="X0")
dat <- merge(dat,x1,by="X1")
dat <- merge(dat,x2,by="X2")
dat <- merge(dat,x3,by="X3")
dat <- merge(dat,x4,by="X4")
dat <- merge(dat,x5,by="X5")
dat <- merge(dat,x6,by="X6")
dat <- merge(dat,x8,by="X8")

dat$X0 <- NULL
dat$X1 <- NULL
dat$X2 <- NULL
dat$X3 <- NULL
dat$X4 <- NULL
dat$X5 <- NULL
dat$X6 <- NULL
dat$X8 <- NULL

train <- dat[dat$tc==1,]
test  <- dat[dat$tc==0,]
train$tc <- NULL
test$tc  <- NULL

train <- apply(train,2,as.numeric)
test  <- apply(test,2,as.numeric)

train <- as.data.frame(train)
test  <- as.data.frame(test)

# EDA ---------------------------------------------------------------------


# FE ----------------------------------------------------------------------
##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]

  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}
train <- train[,!colnames(train) %in% toRemove]
test  <- test[,!colnames(test) %in% toRemove]

# Break out factors and fix classes
factors <- train[,colnames(train)[grepl("code",colnames(train))]]
train   <- train[,colnames(train)[!grepl("code",colnames(train))]]

factors1 <- test[,colnames(test)[grepl("code",colnames(test))]]
test   <- test[,colnames(test)[!grepl("code",colnames(test))]]

factors$tc  <- 1
factors1$tc <- 0

factors_all <- rbind(factors, factors1)

for(i in 1:ncol(factors_all)){
  factors_all[,i] <- as.factor(factors_all[,i])
  factors_all[,i] <- addNA(factors_all[,i])
}

factors  <- factors_all[factors_all$tc==1,]
factors1 <- factors_all[factors_all$tc==0,]

# MC testing ####
#mctest(train,labels,type="i")

correlationMatrix <- cor(train)
highlyCorrelated  <- findCorrelation(correlationMatrix, cutoff=0.9, verbose=T,names=T)
print(highlyCorrelated)

train <- train[,!colnames(train) %in% highlyCorrelated]
test  <- test[,!colnames(test) %in% highlyCorrelated]

##### PCA #####
x <- train
x <- x[,lapply(x,class)=="numeric"]
x <- x[,apply(x, 2, function(col) { length(unique(col)) > 1 })]
x <- na.roughfix(x)

x.pca <- prcomp(x,
                center = TRUE,
                scale. = TRUE,
                tol = .25)

x.pca.pred <- predict(x.pca, newdata = train)
dim(x.pca.pred)
train <- cbind(train, x.pca.pred[,1:5])

x.pca.pred <- predict(x.pca, newdata = test)
dim(x.pca.pred)
test <- cbind(test, x.pca.pred[,1:5])

# Sum of Flags
train$flag_sum <- rowSums(train[,1:(ncol(train)-5)])
test$flag_sum  <- rowSums(test[,1:(ncol(test)-5)])

# # Recursive Feature Elim ##
# set.seed(7)
# require(mlbench)
# require(caret)
#
# # define the control using a random forest selection function
# control <- rfeControl(functions=rfFuncs, method="cv", number=10)
#
# # run the RFE algorithm
# results <- rfe(train, labels, sizes=c(3:10), rfeControl=control)
#
# # summarize the results
# print(results)
# # list the chosen features
# predictors(results)
# # plot the results
# plot(results, type=c("g", "o"))
#

# Add back factors ####
train <- cbind(train, factors)
test  <- cbind(test,factors1)

for(i in 1:ncol(train)){ # used to find factors with only 1 level
  if(class(train[,i])[1] == "factor"){
    train[,i] <- addNA(train[,i])
    if(length(levels(train[,i])) == 1){cat(colnames(train)[i])}
  }
}

train$tc <- NULL
test$tc  <- NULL


# Modeling ----------------------------------------------------------------
train_labels <- labels[1:4000]
valid_labels <- labels[4001:length(labels)]

train_dat <- train[1:4000,]
valid_dat <- train[4001:nrow(train),]

train_dat$y <- train_labels
valid_dat$y <- valid_labels

mod <- glm(y~PC1+PC2+PC3,data=train_dat)
summary(mod)
R2gauss(train_labels,mod)
sig <- data.frame(var=names(summary(mod)$coef[summary(mod)$coef[,4] <= .05, 4]),
                  p=summary(mod)$coef[summary(mod)$coef[,4] <= .05, 4])
sig
cat(shQuote(sig$var), "\n", sep = ",")

mod2 <- glm(y~X49+X196+X207+X322+X323+x1_code,
            data=train_dat)

summary(mod2)
R2gauss(train_labels,mod2)

# DRF ---------------------------------------------------------------------
require(h2o)
# H2O Setup
system("java -Xmx20g -jar E://Jason//h2o//h2o.jar", wait = F)
h2o.init(nthreads = -1)

# Send data to H2O
trainHex <- as.h2o(train_dat)
validHex <- as.h2o(valid_dat)
scoreHex <- as.h2o(test)

drf <- h2o.grid(algorithm="randomForest",
                x= colnames(train_dat),
                y="y",
                training_frame = trainHex,
                validation_frame=validHex,
                ignore_const_cols = T,
                #ntrees = 200,
                max_depth = 9,
                mtries=10,
                sample_rate=.632,
                stopping_metric = "logloss",
                hyper_params=list(ntrees = c(200)#,
                                  #max_depth=c(18)
                                  #,mtries=c(10)
                                  #,sample_rate=c(0.632,)
                                  #col_sample_rate_per_tree=c(.8,1)
                ))

summary(drf)
drf.best <- h2o.getModel("Grid_DRF_train_dat_model_R_1496350093658_1_model_0")

summary(drf.best)
h2o.varimp(drf.best)
vi <- as.data.frame(h2o.varimp(drf.best)); saveRDS(vi, "vi_reusable.RDS")

# Validation
valid_pred <- predict(drf.best, validHex)

# Scoring

pred <- predict(drf.best, scoreHex)
pred <- as.data.frame(pred)

sample$y <- pred

fwrite(sample, "drf.csv", row.names = F) #
saveRDS(sample, "drf.RDS")

# Stack
kernal <- fread("kernal.csv")
sample$y <- kernal$y #*.99 + pred$predict * .01
fwrite(sample,"..//output//sub.csv")

# .99 .01 LB 0.55811

# train_mat <- as.matrix(train[1:4000,])
# valid_mat <- as.matrix(train[4001:nrow(train),])
# test_mat  <- as.matrix(test)

# t1_sparse <- Matrix(as.matrix(train_mat), sparse=TRUE) #[,colnames(train_mat)[sapply(train_mat,class)%in%c("nuermic","integer")]]
# s1_sparse <- Matrix(as.matrix(test_mat), sparse=TRUE) #, with=FALSE
# v1_sparse <- Matrix(as.matrix(valid_mat), sparse=TRUE)
#
# grid_search <- expand.grid(Depth = 8,
#                            L1 = 0:5,
#                            L2 = 0:5,
#                            learning_rate = seq(.02:1,.02))
#
# model <- list()
# perf <- numeric(nrow(grid_search))
#
# for (i in 1:nrow(grid_search)) {
#   model[[i]] <- lgb.train(list(objective = "regression",
#                                metric = "l2",
#                                lambda_l1 = grid_search[i, "L1"],
#                                lambda_l2 = grid_search[i, "L2"],
#                                max_depth = grid_search[i, "Depth"]),
#                           dtrain,
#                           2,
#                           valids,
#                           min_data = 10,
#                           learning_rate = 1,
#                           early_stopping_rounds = 5)
#   perf[i] <- min(rbindlist(model[[i]]$record_evals$test$l2))
# }
#
#
#
# bst <- lightgbm(data = train_mat,
#                 label = train_labels,
#                 #num_leaves = 4,
#                 #nrounds = 2,
#                 nthread = 2,
#                 num_iterations = 350,
#                 learning_rate = 1,
#                 nrounds = 1000,
#                 boosting='dart',
#                 objective = "regression_l2",
#                 max_bin=9, #100
#                 metric='l2',
#                 sub_feature=.5,
#                 bagging_fraction=1, #.85
#                 bagging_freq=10,
#                 min_data=50,
#                 min_hessian=.05,
#                 early_stopping_rounds = 20,
#                 objective = "regression")
#
#
# dtrain <- lgb.Dataset(data = train_mat,
#                       label = train_labels,
#                       free_raw_data = FALSE
#                       ,colnames = colnames(train_mat)
#                       #,categorical_feature = which(colnames(train_mat) %in% cf)
#                       )
#
# lgb.Dataset.construct(dtrain)
# saveRDS(dtrain, "dtrain.RDS")
#
# dtest     <- lgb.Dataset.create.valid(dtrain,
#                                       data          = valid_mat,
#                                       label         = valid_labels,
#                                       free_raw_data = FALSE)
# lgb.Dataset.construct(dtest)
#
# lgb.Dataset.save(dtrain, "dtrain.buffer")
# lgb.Dataset.save(dtest, "dtest.buffer")
#
# # valids is a list of lgb.Dataset, each of them is tagged with name
# # valids allows us to monitor the evaluation result on all data in the list
# valids <- list(train = dtrain, test = dtest)
#
# # We can change evaluation metrics, or use multiple evaluation metrics
# print("Train lightgbm using lgb.train with valids, watch logloss and error")
# bst <- lgb.train(data = dtrain,
#                  valids = valids,
#                  eval = c("mean_absolute_error"), #, "binary_logloss",
#                  nthread = 2,
#                  num_leaves = 128,
#                  learning_rate = 0.0021,
#                  num_iterations = 1000,
#                  min_data_in_leaf = 50,
#                  nrounds = 300,
#                  boosting='gbdt',
#                  objective = "regression_l2",
#                  max_bin=9, #100
#                  metric='l2',
#                  sub_feature=.5,
#                  bagging_fraction=1, #.85
#                  bagging_freq=10,
#                  min_data=50,
#                  min_hessian=.05,
#                  #colnames = colnames(train_mat),
#                  #max_depth = 20,
#                  verbose = 1
#                  #,categorical_feature = cf
# ) #0.0655485 Py 0.0648528
#
#
# # Validation prediction / error
# label.valid = getinfo(dtest, "label")
# pred.valid <- predict(bst, as.matrix(valid_mat))
# err        <- Metrics::mae(label.valid, pred.valid)
# print(paste("test-error=", err)) # 0.0655485186283588
#
# # OOS Scoring
# t <- Sys.time()
# score <- predict(bst, as.matrix(test_mat))
# Sys.time()-t #

# Save Predictions --------------------------------------------------------
saveRDS(pred.valid, "lgb_pred_valid.RDS")
saveRDS(score,      "lgb_pred_score.RDS")


# Write Out Results -------------------------------------------------------
sample$y <- score
fwrite(sample, "naive_lgb3.csv")

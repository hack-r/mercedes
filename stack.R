options(scipen=20)
setwd("..//input")
pacman::p_load(car, caret, data.table, dplyr, lightgbm,
               Matrix, mctest, Metrics, mlbench, MLmetrics, mlr, RRF,
               stringr, sqldf, xgboost)

# Stack -------------------------------------------------------------------
sample <- fread("sample_submission.csv")
train  <- fread("train.csv")
test   <- fread("test.csv")

my_kernel_dart <- fread("jason.csv")
my_priv_dart    <- fread("xgb_12pca_12ica_1grp_8srp_py_500tr_dart.csv")
#questionable   <- fread("..//baseLine.csv")
#questionable2  <- fread("stacked-models.csv")
questionable3  <- fread("questionable3.csv")
#model_0_xgb_d  <- fread("..//layer1_test//model_0_xgb_decomp_pred_layer1_test.csv")
#model_1_xgb_d  <- fread("..//layer1_test//model_1_xgb_decomp_pred_layer1_test.csv")
#model_2_xgb_d  <- fread("..//layer1_test//model_2_xgb_decomp_pred_layer1_test.csv")
#krr1           <- fread("..//layer1_test//model_1_krr_pred_layer1_test.csv")
#model_3_xgb_d  <- fread("..//layer1_test//model_3_xgb_decomp_pred_layer1_test.csv")
#pls            <- fread("..//output//pls.csv")
#ranger         <- fread("..//output//ranger_mod7.csv")
#my_pipe        <- fread("T://RNA//Baltimore//Jason//ad_hoc//mb//output//jdm_stacked_models2.csv")
#hmmm           <- fread('T://RNA//Baltimore//Jason//ad_hoc//mb//output//jdm_stacked_models3_z5_j7.csv')
z9j7           <- fread('T://RNA//Baltimore//Jason//ad_hoc//mb//output//jdm_stacked_models3_z9_j7.csv')
neuton2        <- fread("..//output//neuton2.csv") # CV neutonian .65, oos .57, LB 0.55
#pipe4          <- fread('T://RNA//Baltimore//Jason//ad_hoc//mb//output//model_4_pipe.csv')
pipe5          <- fread('T://RNA//Baltimore//Jason//ad_hoc//mb//output//model_5_pipe.csv')

z0 <- 0.342
z1 <- 0.045
z2 <- 0.513
z3 <- .1
z4 <- 0

if(z0+z1+z2+z3+z4 == 1){
  print("Building stack")
  sample$y       <- my_kernel_dart$y * z0 +
                    my_priv_dart$y   * z1 +
                    questionable3$y  * z2 +
                    z9j7$y           * z3
                   #+pipe5$y          * z4
} else(cat("WE HAVE A PROBLEM!!!"))


plot(density(sample$y,bw=.2))
lines(density(my_kernel_dart$y), col = "gold")
lines(density(my_priv_dart$y), col = "green")
lines(density(questionable3$y,bw=.2),col="red")
lines(density(z9j7$y,bw=.2),col="blue")
#lines(density(pipe4$y,bw=.2),col="yellow")

summary(sample$y)

fwrite(sample, "..//output//kerndart_342_cvdart045_qstnbl3_513_z9j7_10.csv")


# kerndart_342_cvdart045_qstnbl3_513_z9j7_10            LB 0.56913
# kerndart_332_cvdart045_qstnbl3_503_z9j7_10_neuton2_02 LB 0.56911
# kerndart_332_cvdart045_qstnbl3_503_z9j7_10_pipe5_02   LB 0.56906


# Older -------------------------------------------------------------------
# ykerneldart38_cvdart05_questionable3_57_z9j7_10 LB 0.56913
# dart38_cvdart05_questionable3_57_z9j7_10_neu_01 LB 0.56913
# dart38_cvdart05_questionable3_57_z9j7_10_neu_03 LB 0.56912
# ykerneldart38_cvdart05_questionable3_57_z9j7_11 LB 0.56912
# ykerneldart38_cvdart05_questionable3_57_z9j7_09 LB 0.56912
# ykerneldart38_cvdart05_questionable3_57_z9j7_12 LB 0.56911
# ykerneldart38_cvdart05_questionable3_57_z9j7_07 LB 0.56908
# ykerneldart38_cvdart05_questionable3_57_z9j7_05 LB 0.56899
# eldart38_cvdart05_qtable3_57_z9j7_10_pipe4_02   LB 0.56881
# mykerneldart38_cvdart05_questionable3_57_hmmm05 LB 0.56876
# mykerneldart38_cvdart05_questionable3_57_hmmm02 LB 0.56870
# mykerneldart38_cvdart05_questionable3_57_hmmm01 LB 0.56865
# ykerneldart38_cvdart05_questionable3_57_z9j7_20 LB 0.56860
# stack_mykerneldart38_cvdart05_questionable3_57  LB 0.56859
# stack_mykerneldart38_cvdart10_questionable3_55  LB 0.56859
# stack_mykerneldart38_cvdart10_questionable3_52  LB 0.56859
# stack_mykerneldart40_cvdart05_questionable3_55  LB 0.56859
# mykerneldart38_cvdart05_questionable3_57_hmmm10 LB 0.56858
# stack_mykerneldart35_cvdart10_questionable3_55  LB 0.56857
# kerneldart38_cvdart05_questionable3_56_ranger_1 LB 0.56857
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
# kerneldart38_cvdart05_questionable3_57_pipe3_10 LB 0.56810
# stack_mykerneldart48_cvdart02_questionable50    LB 0.56808
# stack_mykerneldart005_cvdart005_questionable3_99   0.56798
# 100% questionable2                                 0.56793
# mykerneldart38_cvdart05_questionable3_57_z7j9_10LB 0.56769
# stack_mykerneldart38_mydart_035_questionable3_57_mod0_005_mod1_005_mod2_005  LB 0.56768
# mykerneldart38_cvdart05_questionable3_55_krr_02 LB 0.56715
# stack_mykerneldart38_questionable3_52_model1xgb_05_model2xgb_05 0.55462
# kerneldart38_cvdart05_questionable3_55_mod3_02  LB 0.53988
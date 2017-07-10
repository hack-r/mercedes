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
my_kernel_dartp <- fread("jasonp.csv") # added probed
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
h2oens         <- fread("..//output//ens.csv")
#h2oens2        <- fread("..//output//ens2.csv")
#h2oens4         <- readRDS("..//output//h2o_ens4.RDS")
h2oens4b         <- readRDS("..//output//h2o_ens4b.RDS")
z9j7p            <- fread("T://RNA//Baltimore//Jason//ad_hoc//mb//output//model_2_pipe_z7_j9.csv") # added probed
mod9           <- fread("T://RNA//Baltimore//Jason//ad_hoc//mb//output//caret_mod9.csv")

z0 <- 0.627
z1 <- 0.03
z2 <- 0.0
z3 <- .23
z4 <- .113


if(z0+z1+z2+z3+z4 == 1){
  print("Building stack")
  sample$y       <- my_kernel_dart$y  * z0 +
                    my_kernel_dartp$y * z1 +
                    questionable3$y   * z2 +
                    z9j7p$y           * z3 +
                    h2oens$y          * z4
} else(cat("WE HAVE A PROBLEM!!!"))


plot(density(sample$y,bw=.2))
lines(density(my_kernel_dart$y), col = "gold")
lines(density(my_priv_dart$y), col = "green")
lines(density(questionable3$y,bw=.2),col="red")
lines(density(z9j7$y,bw=.2),col="blue")
lines(density(h2oens$y,bw=.2),col="yellow")

lines(density(my_kernel_dartp$y),col="blue") # much more like train but shifted right

summary(sample$y)

fwrite(sample, "..//output//kerndart_627_z9j7p_23_h2oens113_kerndartp_03.csv")

# kerndart_627_z9j7p_23_h2oens113_kerndartp_03          LB 0.57156
# kerndart_617_z9j7p_23_h2oens143_kerndartp_01          LB 0.57104
# kerndartp_617_z9j7p_23_h2oens153                      LB 0.57068
# kerndart_607_z9j7p_23_h2oens163                       LB 0.57067
# kerndart_607_qstnbl3_04_z9j7p_23_h2oens123            LB 0.57065
# kerndart_60_z9j7p_23_h2oens17                         LB 0.57065
# kerndart_607_qstnbl3_02_z9j7p_25_h2oens123            LB 0.57065
# kerndart_607_z9j7p_27_h2oens123                       LB 0.57063
# kerndart_607_qstnbl3_03_z9j7p_23_h2oens123_mod9_01    LB 0.57055
# kerndart_607_qstnbl3_14_z9j7p_13_h2oens123            LB 0.57044
# kerndart_607_qstnbl3_16_z9j7p_11_h2oens123            LB 0.57036
# kerndart_607_qstnbl3_16_z9j7_11_h2oens123             LB 0.57034
# kerndart_607_qstnbl3_159_z9j7_111_h2oens123           LB 0.57034
# kerndart_607_qstnbl3_161_z9j7_109_h2oens123           LB 0.57034
# kerndart_607_qstnbl3_15_z9j7_12_h2oens123             LB 0.57034
# kerndart_62_qstnbl3_147_z9j7_11_h2oens123             LB 0.57034
# kerndart_607_qstnbl3_17_z9j7_10_h2oens123             LB 0.57033
# kerndart_61_qstnbl3_16_z9j7_11_h2oens12               LB 0.57033
# kerndart_657_qstnbl3_12_z9j7_10_h2oens123             LB 0.57031
# kerndart_507_qstnbl3_27_z9j7_10_h2oens123             LB 0.57031
# kerndart_607_qstnbl3_13_z9j7_14_h2oens123             LB 0.57031
# kerndart_617_qstnbl3_27_z9j7_10_h2oens113             LB 0.57029
# kerndart_707_qstnbl3_07_z9j7_10_h2oens123             LB 0.57027
# kerndart_637_qstnbl3_27_z9j7_07_h2oens123             LB 0.57022
# kerndart_407_qstnbl3_37_z9j7_10_h2oens123             LB 0.57021
# kerndart_607_qstnbl3_10_z9j7_17_h2oens123             LB 0.57019
# kerndart_307_qstnbl3_47_z9j7_095_h2oens128            LB 0.57002
# kerndart_307_qstnbl3_47_z9j7_10_h2oens123             LB 0.57002
# kerndart_31_qstnbl3_47_z9j7_10_h2oens12               LB 0.57002
# kerndart_307_qstnbl3_47_z9j7_09_h2oens123_ens4_01     LB 0.57002
# kerndart_307_qstnbl3_47_z9j7_09_h2oens133             LB 0.57002
# kerndart_307_qstnbl3_46_z9j7_10_h2oens133             LB 0.57002
# kerndart_307_qstnbl3_47_z9j7_09_h2oens113_ens4_02     LB 0.57001
# kerndart_307_qstnbl3_46_z9j7_11_h2oens123             LB 0.57001
# kerndart_30_qstnbl3_47_z9j7_10_h2oens13               LB 0.57001
# kerndart_307_qstnbl3_478_z9j7_10_h2oens115            LB 0.57001
# kerndart_307_cvdart005_qstnbl3_478_z9j7_10_h2oens11   LB 0.56999
# kerndart_307_qstnbl3_47_z9j7_09_h2oens103_ens4_03     LB 0.56999
# kerndart_307_qstnbl3_44_z9j7_09_h2oens153             LB 0.56999
# kerndart_307_qstnbl3_47_z9j7_09_h2oens113_ens4b_02    LB 0.56998
# kerndart_307_cvdart015_qstnbl3_478_z9j7_10_h2oens10   LB 0.56994
# kerndart_307_cvdart035_qstnbl3_478_z9j7_10_h2oens09   LB 0.56982
# kerndart_307_cvdart045_qstnbl3_478_z9j7_10_h2oens07   LB 0.56974
# kerndart_302_cvdart045_qstnbl3_473_z9j7_10_h2oens08   LB 0.56974
# kerndart_312_cvdart045_qstnbl3_483_z9j7_10_h2oens06   LB 0.56969
# kerndart_317_cvdart045_qstnbl3_488_z9j7_10_h2oens05   LB 0.56962
# kerndart_322_cvdart045_qstnbl3_493_z9j7_10_h2oens04   LB 0.56955
# kerndart_327_cvdart045_qstnbl3_498_z9j7_10_h2oens03   LB 0.56946
# kerndart_332_cvdart045_qstnbl3_503_z9j7_10_h2oens02   LB 0.56936
# kerndart_342_cvdart045_qstnbl3_503_z9j7_10_h2oens01   LB 0.56926
# kerndart_207_qstnbl3_47_z9j7_10_h2oens223             LB 0.56919
# kerndart_342_cvdart045_qstnbl3_513_z9j7_10            LB 0.56913
# kerndart_332_cvdart045_qstnbl3_503_z9j7_10_neuton2_02 LB 0.56911
# kerndart_332_cvdart045_qstnbl3_503_z9j7_10_pipe5_02   LB 0.56906
# kerndart_307_cvdart045_qstnbl3_478_z9j7_10_ens2_07    LB 0.56903
# kerndart_307_qstnbl3_47_z9j7_10_h2oens3_123           LB 0.56712
# kerndartp_607_z9j7p_23_h2oens163                      LB 0.48555

# temp code - looking at how density changed
a <- fread("..//output//kerndart_312_cvdart045_qstnbl3_483_z9j7_10_h2oens06.csv")
b <- fread("..//output//kerndart_317_cvdart045_qstnbl3_488_z9j7_10_h2oens05.csv")
c <- fread("..//output//kerndart_322_cvdart045_qstnbl3_493_z9j7_10_h2oens04.csv")
d <- fread("..//output//kerndart_342_cvdart045_qstnbl3_503_z9j7_10_h2oens01.csv")
e <- fread("..//output//stack_mykerneldart38_cvdart05_questionable3_57_z9j7_10.csv")

plot(density(e$y, bw=.2))
lines(density(d$y, bw=.2), col = "green")
lines(density(c$y, bw=.2), col = "blue")
lines(density(b$y, bw=.2), col = "purple")
lines(density(a$y, bw=.2), col = "red")

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
# Jason D. Miller

import os
import pandas as pd
import csv

file = open("template0.py","r")
template0 = file.readlines()

file = open("template1.py","r")
template1 = file.readlines()


models = [#"linear_model.ARDRegression",
            "linear_model.BayesianRidge",
            "linear_model.ElasticNet",
            "linear_model.ElasticNetCV",
            "linear_model.HuberRegressor",
            #"linear_model.Lars", # crazy scores, check predictions
            #"linear_model.LarsCV", # crazy scores, check predictions
            "linear_model.Lasso",
            "linear_model.LassoCV",
            "linear_model.LassoLars",
            "linear_model.LassoLarsCV",
            "linear_model.LassoLarsIC",
            "linear_model.LinearRegression",
            #"linear_model.LogisticRegression",
            #"linear_model.LogisticRegressionCV",
            #"linear_model.MultiTaskLasso",
            #"linear_model.MultiTaskElasticNet",
            #"linear_model.MultiTaskLassoCV",
            #"linear_model.MultiTaskElasticNetCV",
            "linear_model.OrthogonalMatchingPursuit",
            "linear_model.OrthogonalMatchingPursuitCV",
            #"linear_model.PassiveAggressiveClassifier",
            #"linear_model.PassiveAggressiveRegressor",
            #"linear_model.Perceptron",
            #"linear_model.RandomizedLasso", # no method predict
            #"linear_model.RandomizedLogisticRegression",
            "linear_model.RANSACRegressor",
            "linear_model.Ridge",
            #"linear_model.RidgeClassifier",
            #"linear_model.RidgeClassifierCV",
            "linear_model.RidgeCV"#,
            #"linear_model.SGDClassifier",
            #"linear_model.SGDRegressor",# consider for removal # crazy scores, check predictions
            #"linear_model.TheilSenRegressor", # very slow consider for removal
            #"linear_model.lars_path", # args
            #"linear_model.lasso_path", # args
            #"linear_model.lasso_stability_path", # args
            #"linear_model.logistic_regression_path", # args
            #"linear_model.orthogonal_mp",
            #"linear_model.orthogonal_mp_gram"
]

params = ["n_iter, tol",
            "n_iter, tol",
            "alpha, l1_ratio",
            "l1_ratio, eps",
            "epsilon",
            "fit_intercept, verbose",
            "fit_intercept",
            "alpha, fit_intercept",
            "eps, n_alphas",
            "alpha",
            "fit_intercept",
            "criterion",
            "",
            "penalty",
            "Cs",
            "alpha",
            "alpha",
            "eps",
            "",
            "",
            "",
            "",
            "C",
            "penalty, alpha",
            "alpha",
            "",
            "",
            "alpha, fit_intercept",
            "alpha",
            "alphas",
            "alphas",
            "loss, penalty",
            "loss, penalty",
            "",
            "X, y, Xy, Gram",
            "X, y, eps",
            "X, y",
            "X, y",
            "X, y",
            "Gram, Xy"]

class create_code:
    def doit(self):
        n = 0
        for m in models:
            print(m)
            print(n)
            code = [s.replace('MODELNUMBER', n.__str__()) for s in template0] # for s in l
            code = [s.replace('MODELNAME', m) for s in code]
            if n == 6:
                code = [s.replace('PARAMETERS', "alpha = 0.01, copy_X = True, fit_intercept = True, fit_path = True, max_iter = 500, normalize = True, positive = False, precompute = 'auto', verbose = False") for s in code]  # params[n])
            else:
                code = [s.replace('PARAMETERS', "") for s in code] #params[n])
            output_file = 'model_%i.py' % n
            thefile = open(output_file, 'w')
            for item in code:
                thefile.write("%s" % item)
            n += 1

class create_code1:
    def doit(self):
        n = 0
        for m in models:
            print(m)
            print(n)
            code = [s.replace('MODELNUMBER', n.__str__()) for s in template1] # for s in l
            code = [s.replace('MODELNAME', m) for s in code]
            if n == 6:
                code = [s.replace('PARAMETERS', "alpha = 0.01, copy_X = True, fit_intercept = True, fit_path = True, max_iter = 500, normalize = True, positive = False, precompute = 'auto', verbose = False") for s in code]  # params[n])
            else:
                code = [s.replace('PARAMETERS', "") for s in code] #params[n])
            output_file = 'model_%i_decomp.py' % n
            thefile = open(output_file, 'w')
            for item in code:
                thefile.write("%s" % item)
            n += 1

cc   = create_code()
cc.doit()

cc   = create_code1()
cc.doit()

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
# transactions as itemMatrix in sparse format with
# 4209 rows (elements/itemsets/transactions) and
# 356 columns (items) and a density of 0.1629868
#
# most frequent items:
#   X205     X74    X111    X361    X229 (Other)
# 4208    4206    4103    4066    4041  223596
#
# element (itemset/transaction) length distribution:
#   sizes
# 31  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70
# 1   4   1   1   6  33  24   9  34  36  39  59  64  53  77  76 107 161 123 251 171 194 222 238 250 238 187 192 274 195 154 129  93  88  65  44  80
# 71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  94
# 36  47  25  24  23  15  13  11  10  12   3   8   1   2   2   1   1   1   1
#
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 31.00   53.00   58.00   58.02   63.00   94.00


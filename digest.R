

library(digest)
suppressMessages(library(dplyr))

train=read.csv('../input/train.csv', stringsAsFactors=F)
test=read.csv('../input/test.csv', stringsAsFactors=F)


hash_names = setdiff(names(train),c('ID','y'))
train$hash = apply(train[,hash_names], 1, digest)
test$hash = apply(test[,hash_names], 1, digest)


hlist <- train %>%
  group_by(hash) %>%
  summarise(n=n(),
            avg = mean(y),
            sd = sd(y)) %>%
  filter(n > 1) %>%
  arrange(desc(n))

hlist



plot(hlist$avg, hlist$sd, main = "StDev vs Avg Test Time for Identical Features",
     xlab = "Avg Test Time", ylab = "StDev of Test Time")



plotdat <- train %>%
  filter(hash %in% hlist$hash) %>%
  select(hash,y) %>%
  left_join(hlist)

plot(plotdat$avg, plotdat$y, main = "Observation Test Time vs Avg Test Time",
     xlab = "Avg Test Time for Cars with Identical Features",
     ylab = "Test Time for Car")
abline(0,1, col = "Blue") # Add mean of Y


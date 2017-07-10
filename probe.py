
train_test<-train_test %>%
  mutate(X118_X127_X238 = paste(X118 X127 X238 sep="_")
         X0group=case_when(
           X118_X127_X238 == "0_0_0" ~ "Group 1"
           X118_X127_X238 == "1_1_1" ~ "Group 2"
           X118_X127_X238 == "1_1_0" ~ "Group 2"
           X118_X127_X238 == "0_1_1" ~ "Group 2"
           X118_X127_X238 == "0_0_1" ~ "Group 3"
           X118_X127_X238 == "1_0_1" ~ "Group 4"
           X118_X127_X238 == "1_0_0" ~ "Group 4"
           TRUE                      ~ "Group X"
         ))


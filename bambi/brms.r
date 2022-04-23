library(brms)
setwd("/home/tomas/Desktop/Cosas/oss/bambinos/bambi")


data <- read.csv("toy.csv")
data$n <- with(data, fu_1 + fu_2 + fu_3 + fu_4)
data$y <- with(data, cbind(fu_1, fu_2, fu_3, fu_4))

fit <- brm(
  bf(y | trials(n) ~ 1 + (1 | state)), 
  data = data, 
  family = multinomial()
)


pairs(fit)

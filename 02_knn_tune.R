# KNN Tuning 

library(tidymodels)
library(tidyverse)
library(tictoc)
library(doMC)
tidymodels_prefer()

load("results/tuning_setup.rda")

#########################
# Parallel processing
registerDoMC(cores = 8)
#########################
# define model engine 

knn_model <- nearest_neighbor(neighbors() = tune()
                         ) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

knn_params <- extract_parameter_set_dials(knn_model) 

knn_grid <- grid_regular(knn_params, levels = 5) 
# Elastic Net Tuning 

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

en_model <- logistic_reg(mixture = tune(),
                       penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

en_params <- extract_parameter_set_dials(en_model) 

en_grid <- grid_regular(en_params, levels = 5) 

# update recipe 
wildfire_interact <- wildfire_recipe %>% 
  step_interact(~ all_numeric_predictors()^2)

wildfire_interact %>% 
  prep(wildfire_train) %>% 
  bake(new_data = NULL)

stopCluster(cl)

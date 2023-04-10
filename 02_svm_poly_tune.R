# SVM Tuning 

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

svm_poly_model <- svm_poly(
  mode = "classification",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab")

svm_poly_params <- extract_parameter_set_dials(svm_poly_model) 

svm_poly_grid <- grid_regular(svm_poly_params, levels = 5) 

svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(wildfire_recipe)

# Tune grid 
# clear and start timer
tic.clearlog()
tic("SVM Polynomial")

svm_poly_tune <- tune_grid(
  svm_poly_workflow,
  resamples = wildfire_folds,
  grid = svm_poly_grid,
  control = control_grid(save_pred = TRUE, # creates extra column for each prediction 
                         save_workflow = TRUE, # lets you use extract_workflow 
                         parallel_over = "everything") # this helps with parallel processing
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

svm_poly_tictoc <- tibble(model = time_log[[1]]$msg,
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(svm_poly_tune, svm_poly_tictoc, 
     file = "results/svm_poly_tuned.rda" )

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

knn_model <- nearest_neighbor(mode = "classification",
                              neighbors = tune()) %>% 
  set_engine("kknn")

knn_params <- extract_parameter_set_dials(knn_model) 

knn_grid <- grid_regular(knn_params, levels = 5) 

knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(wildfire_recipe)

########################################################################
# Tune grid 
# clear and start timer
tic.clearlog()
tic("K Nearest Neighbors")

knn_tune <- tune_grid(
  knn_workflow,
  resamples = wildfire_folds,
  grid = knn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything") # this helps with parrallel processing
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(knn_tune, knn_tictoc, 
     file = "results/knn_tuned.rda" )



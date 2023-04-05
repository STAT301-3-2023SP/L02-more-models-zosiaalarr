# Random Forest Tuning 

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

rf_model <- rand_forest(mtry = tune(), min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(c(1, 15)))

rf_grid <- grid_regular(rf_params, levels = 5)

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(wildfire_recipe)

########################################################################
# Tune grid 
# clear and start timer
tic.clearlog()
tic("Random Forest")

rf_tune <- tune_grid(
  rf_workflow,
  resamples = wildfire_folds,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything") # this helps with parrallel processing
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(rf_tune, rf_tictoc, 
     file = "results/rf_tuned.rda" )



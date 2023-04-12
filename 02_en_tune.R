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
# define model engine and workflow

en_model <- logistic_reg(mixture = tune(),
                       penalty = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

en_params <- extract_parameter_set_dials(en_model) 

en_grid <- grid_regular(en_params, levels = 5) 

# update recipe 
wildfire_interact <- wildfire_recipe %>% 
  step_interact(~all_numeric_predictors()^2)

# wildfire_interact %>% 
#   prep(wildfire_train) %>% 
#   bake(new_data = NULL)

en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(wildfire_interact)

########################################################################
# Tune grid 
# clear and start timer
tic.clearlog()
tic("Elastic Net")

en_tune <- tune_grid(
  en_workflow,
  resamples = wildfire_folds,
  grid = en_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything") # this helps with parrallel processing
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

en_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)
en_tictoc

save(en_tune, en_tictoc, 
     file = "results/en_tuned.rda" )



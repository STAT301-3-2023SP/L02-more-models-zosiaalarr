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

mars_model <- mars(
  mode = "classification", # or "regression"
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 23)))

mars_grid <- grid_regular(mars_params, levels = 5) 

mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(wildfire_recipe)

# Tune grid 
# clear and start timer
tic.clearlog()
tic("MARS")

mars_tune <- tune_grid(
  mars_workflow,
  resamples = wildfire_folds,
  grid = mars_grid,
  control = control_grid(save_pred = TRUE, # creates extra column for each prediction 
                         save_workflow = TRUE, # lets you use extract_workflow 
                         parallel_over = "everything") # this helps with parallel processing
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

mars_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)


save(mars_tune, mars_tictoc, 
     file = "results/mars_tuned.rda" )

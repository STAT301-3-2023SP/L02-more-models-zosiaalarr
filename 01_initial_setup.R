# Load package(s)
library(tidymodels)
library(tidyverse)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data
wildfires_dat <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-burned)

# visualize response variables 
# if there is class imbalance use downsampling or upsampling 
ggplot(wildfires_dat, aes(x = wlf)) + 
  geom_bar()

# missingness- none 
# if present we could impute using step_impute_mean _median _mode (for categories)
# _knn (predicts it) _linear (predicts it)

# check for factors with many options 

##############################################################################
# split the data 
set.seed(3013)
wildfire_split <- initial_split(wildfires_dat, prop = 0.8, strata = wlf)

wildfire_train <- training(wildfire_split)
wildfire_test <- testing(wildfire_split)

# v-fold cross validation 
wildfire_folds <- vfold_cv(wildfire_train, v = 5, repeats = 3,
                           strata = wlf)



###############################################################################
# recipes 
wildfire_recipe <- recipe(wlf ~ ., data = wildfire_train) %>% 
  step_novel(all_nominal_predictors()) %>% # if a factor level was not seen in training set, it will be added at testing-- error catching method 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% # always put before step_normalize 
  step_normalize(all_predictors()) 
  
  

wildfire_recipe %>% 
  prep(wildfire_train) %>%  
  bake(new_data = NULL) %>% 
  view()

save(wildfire_train, wildfire_test, wildfire_folds, wildfire_recipe, 
     file = "results/tuning_setup.rda" )




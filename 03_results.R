library(tidymodels)
library(tidyverse)

tidymodels_prefer()

set.seed(25)

result_files <- list.files("results/", "*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

###########################################################################
# baseline/null model 

null_mod <- null_model(mode = "classification") %>% 
  set_engine("parsnip")

null_wkflw <- workflow() %>% 
  add_model(null_mod) %>% 
  add_recipe(wildfire_recipe)

null_fit <- null_wkflw %>% 
  fit_resamples(resamples = wildfire_folds,
                control = control_resamples(save_pred = TRUE))

null_fit %>% 
  collect_metrics()

###########################################################################
# organize results to find best overall 

# individual model results, good to put into an appendix  
autoplot(en_tune, metric = "roc_auc")

# we can show the best model, good for looking at a single model  
en_tune %>% 
  show_best(metric = "roc_auc")

######################################################################
# put all our tune_grids together 
model_set <- as_workflow_set(
  "elastic_net" = en_tune,
  "rand_forest" = rf_tune,
  "knn" = knn_tune,
  "boosted_tree" = bt_tune,
  "neural_net" = nn_tune,
  "mars" = mars_tune,
  "svm_poly" = svm_poly_tune,
  "svm_radial" = svm_radial_tune
)

## plot of our results 

mars_tune %>% 
  autoplot(metric = "roc_auc")

en_tune %>% 
  autoplot(metric = "roc_auc")

rf_tune %>% 
  autoplot(metric = "roc_auc")

bt_tune %>% 
  autoplot(metric = "roc_auc")

svm_poly_tune %>% 
  autoplot(metric = "roc_auc")

svm_radial_tune %>% 
  autoplot(metric = "roc_auc")

nn_tune %>% 
  autoplot(metric = "roc_auc")

knn_tune %>% 
  autoplot(metric = "roc_auc")



model_set %>% 
  autoplot(metric = "roc_auc")

model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal() +
  geom_text(aes(y = mean - 0.03, label = wflow_id), angle = 90, hjust = 1) +
  ggtitle(label = "Best Results") +
  ylim(c(0.7, 0.9)) + 
  theme(legend.position = "none")
# save as image for the report 

## table of our results 
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>% 
  select(best) %>% 
  unnest(cols = c(best)) %>% 
  slice_max(mean)

## computation time 
model_times <- bind_rows(en_tictoc,
                         bt_tictoc,
                         rf_tictoc,
                         knn_tictoc,
                         nn_tictoc, 
                         svm_poly_tictoc,
                         svm_radial_tictoc,
                         mars_tictoc)


##########################################################################
# fit the best model to training set and predict testing set 

# Elastic Net 
best_en <- en_tune %>% 
  show_best(metric = "roc_auc") %>% 
  slice_head()

# Random Forest 
best_rf <- rf_tune %>% 
  show_best() %>% 
  slice_head()
best_rf

best_rf <- show_best(rf_tune, metric = "rmse")[1,]

# KNN 
best_knn <- knn_tune %>% 
  show_best() %>% 
  slice_head()
best_knn

# Boosted Tree 
best_bt <- bt_tune %>% 
  show_best() %>% 
  slice_head()
best_bt

# Neural Network 
best_nn <- nn_tune %>% 
  show_best() %>% 
  slice_head()
best_nn

# SVM Poly 
best_svm_poly <- svm_poly_tune %>% 
  show_best() %>% 
  slice_head()
best_svm_poly

# SVM Radial 
best_svm_radial <- svm_radial_tune %>% 
  show_best() %>% 
  slice_head()
best_svm_radial

# MARS 
best_mars <- mars_tune %>% 
  show_best() %>% 
  slice_head()
best_mars



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
  group_by(wflow_id) %>% # because need best model for every workflow id
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>% # create new column, use show_best to show roc_auc results
  select(best) %>% 
  unnest(cols = c(best)) #%>% 
#slice_max(mean) # name of variable we want the maximum of 

## computation time 
model_times <- bind_rows(en_tictoc, 
                         bt_tictoc,
                         rf_tictoc,
                         knn_tictoc,
                         nn_tictoc,
                         svm_poly_tictoc,
                         svm_radial_tictoc,
                         mars_tictoc) %>% 
  mutate(wflow_id = c(  "elastic_net",
                        "boosted_tree",
                        "random_forest", 
                        "knn", 
                        "nn", 
                        "svm_poly",
                        "svm_radial",
                        "mars"))


result_table <- merge(model_results, model_times) %>% 
  select(model, mean, runtime) %>% 
  rename(roc_auc = mean)

save(result_table,file =  "results/result_table.rda")


##########################################################################
# fit the best model to training set and predict testing set 

# finalize the workflow

nn_workflow <- nn_workflow %>% 
  finalize_workflow(select_best(nn_tune, metric = "roc_auc"))

# fit training data to final workflow 

final_fit <- fit(nn_workflow, wildfire_train)

# predict the testing data 
final_pred <- predict(final_fit, wildfire_test) %>% 
  bind_cols(wildfire_test %>% select(wlf))


# final roc_auc 

metric <- metric_set(roc_auc)

final_pred %>% 
  metric(truth = wlf, estimate = .pred)

# confusion plot of results 



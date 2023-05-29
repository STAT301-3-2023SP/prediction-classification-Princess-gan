# Get final model results

library(tidymodels)
library(tidyverse)
library(yardstick)
library(caret)
tidymodels_prefer()

test <- read_csv("data/test.csv") %>%
  janitor::clean_names() %>% 
  select(-id) %>% 
  mutate_at(cat_var, as.factor)

result_files <- list.files("results/","*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

#---------------------------------------------------
# baseline/null model
null_model <- null_model(mode = "regression") %>% 
  set_engine("parsnip")

null_wflow <- workflow() %>% 
  add_model(null_model) %>% 
  add_recipe(my_recipe)

null_fit <- null_wflow %>% 
  fit_resamples(resamples = my_fold,
                control = control_resamples(save_pred = TRUE))
               # whether it saves each pred of resamples
null_fit %>% 
  collect_metrics()

#---------------------------------------------------
# organize results to find best overall 

# Individual model results - tune_grid
# ⬆️Recommended to put in the appendix
autoplot(en_tune, metric = "rmse") # want to find the lowest one

en_tune %>% 
  show_best(metric = "rmse")
# predict(en_results, new_data = test)

mars_tune %>% 
  show_best(metric = "rmse")

mlp_tune %>% 
  show_best(metric = "rmse")

knn_tune %>% 
  show_best(metric = "rmse")

rf_tune %>% 
  show_best(metric = "rmse")
#-----------------------------------------------
# put all our tune_grids together 
model_set <- as_workflow_set(
  "elastic_net" = en_tune, 
  "rand_forest" = rf_tune,
  "knn" = knn_tune,
  "boosted_tree" = bt_tune,
  "mlp" = mlp_tune, # neural network
  #"svm_poly" = svm_poly_tune,
  "svm_rbf" = svm_rbf_tune,
  "mars" = mars_tune
)


## plot of our results
model_set %>% 
  autoplot(metric = "roc_auc")
  # can see the tuning results but cannot clearly see the best

## plot just the best
model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal()+
  geom_text(aes(y = mean- 0.03, label = wflow_id), angle = 90, hjust = 1) +
                         # wflow_id shown in "model_set"
  #ylim(c(0.7, 0.9)) +
  theme(legend.position = "none")
# will want this in report! either save image or will need to include this code

## table of our results
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc"), n = 1) %>% # with n=1, can only show top results
  select(wflow_id, best) %>% 
# show_best gives more info than select_best function!
  unnest(cols = c(best)) #%>% 
  #slice_max(mean)
en_tictoc
## computation time
model_times <- bind_rows(en_tictoc,
                         bt_tictoc, 
                         rf_tictoc,
                         knn_tictoc,
                         mlp_tictoc,
                         svm_poly_tictoc,
                         svm_rbf_tictoc,
                         mars_tictoc) %>%
  mutate(wflow_id = c("elastic_net" ,
                       "rand_forest",
                       "knn" ,
                       "boosted_tree",
                       "mlp", # neural network
                       "svm_poly" ,
                       "svm_rbf",
                       "mars"))
# model_times %>% 
#   colnames()
model_results
# model_times %>% 
#   view()
result_table <- 
  merge(model_results, model_times) %>% 
  select(wflow_id, mean, runtime) %>% 
  rename(roc_auc = mean) %>% 
  arrange(desc(roc_auc))

result_table_filtered <- result_table %>% 
  group_by(wflow_id) %>%
  top_n(1, wt = roc_auc)

result_table_filtered
save(result_table, result_table_filtered, file = "results/result_table.rda")
#------------------------------------------------
# fit the best model to training set and predict testing set 
### best model is elastic net

# finalize  the workflow
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tune, metric = "roc_auc"))

en_workflow_tuned
# fit testing data to final workflow
en_results <- fit(en_workflow_tuned, wildfire_train)

# predict the testing data 
en_predict <- 
  predict(en_results, new_data = wildfire_test) %>% 
  bind_cols(wildfire_test %>% select(wlf))

en_predict

# en_test_metrics <- en_predict %>%
#   metrics(
#     truth = wlf,
#     estimate = .pred_class
#     metric_set = metric_set(yardstick::roc_auc(), yardstick::accuracy())
#   )
# 
# en_test_metrics

# final roc_auc
en_metric <- metric_set(rmse)

en_pred <- 
  predict(en_results, new_data = wildfire_test) %>% 
  bind_cols(wildfire_test %>% select(wlf))
 
pred_prob <- predict(en_results, wildfire_test, type = "prob") 

en_pred_2 <- en_pred %>% 
  bind_cols(pred_prob)

en_pred_2

roc_auc_test <- en_pred_2 %>% 
  roc_auc(wlf, .pred_yes)

# confusion plot of results
cm <- confusionMatrix(en_pred$wlf, en_pred$.pred_class)

cm

save(roc_auc_test, cm, en_pred_2,
     file = "results/en_analysis.rda")

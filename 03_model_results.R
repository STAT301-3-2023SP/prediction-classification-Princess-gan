# Get final model results

library(tidymodels)
library(tidyverse)
library(yardstick)
library(caret)
tidymodels_prefer()

result_files <- list.files("results/","*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

test <- read_csv("data/test.csv") %>%
  janitor::clean_names()
  #select(-id) %>% 
  #mutate_at(cat_var, as.factor)


#---------------------------------------------------
# baseline/null model
null_model <- null_model(mode = "classification") %>% 
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
autoplot(en_tune, metric = "roc_auc") # want to find the lowest one

en_tune %>% 
  show_best(metric = "roc_auc")
# predict(en_results, new_data = test)

mars_tune %>% 
  show_best(metric = "roc_auc")

mlp_tune %>% 
  show_best(metric = "roc_auc")

knn_tune %>% 
  show_best(metric = "roc_auc")

rf_tune %>% 
  show_best(metric = "roc_auc")
#-----------------------------------------------
# put all our tune_grids together 
model_set <- as_workflow_set(
#  "elastic_net" = en_tune, 
  "rand_forest" = rf_tune,
  "knn" = knn_tune,
#  "boosted_tree" = bt_tune,
  "mlp" = mlp_tune, # neural network
  #"svm_poly" = svm_poly_tune,
#  "svm_rbf" = svm_rbf_tune,
  "mars" = mars_tune
)


## plot of our results
model_set %>% 
  autoplot(metric = "roc_auc")
  # can see the tuning results but cannot clearly see the best

## plot just the best
model_set_image <- model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal()+
  geom_text(aes(y = mean-0.02, label = wflow_id), angle = 90, hjust = 0.5) +
                         # wflow_id shown in "model_set"
  #ylim(c(0.7, 0.9)) +
  theme(legend.position = "none")
# will want this in report! either save image or will need to include this code
model_set_image
save(model_set_image, file = "images/model_set_image.rda")
## table of our results
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc"), n = 1) %>% # with n=1, can only show top results
  select(wflow_id, best) %>% 
# show_best gives more info than select_best function!
  unnest(cols = c(best)) #%>% 
  #slice_max(mean)
model_results


## computation time
model_times <- bind_rows(#en_tictoc,
                         #bt_tictoc, 
                         rf_tictoc,
                         knn_tictoc,
                         mlp_tictoc,
                         #svm_poly_tictoc,
                         #svm_rbf_tictoc,
                         mars_tictoc) %>%
  mutate(wflow_id = c(#"elastic_net" ,
                       "rand_forest",
                       "knn" ,
                       #"boosted_tree",
                       "mlp", # neural network
                       #"svm_poly" ,
                       #"svm_rbf",
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
save(result_table, result_table_filtered, file = "table/result_table.rda")
save(model_times, file = "table/model_times.rda")

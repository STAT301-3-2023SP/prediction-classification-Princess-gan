# fit the best model to training set and predict testing set 
library(tidymodels)
library(tidyverse)
library(yardstick)
library(caret)
tidymodels_prefer()


result_files <- list.files("results/","*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

# load data
train <- read_csv("data/train.csv") %>%
  janitor::clean_names() %>% 
  select(-id) %>% 
  mutate_at(cat_var, as.factor)

test <- read_csv("data/test.csv") %>%
  mutate(    x005 = as.factor(x005),
             x034 = as.factor(x034),
             x043 = as.factor(x043),
             x084 = as.factor(x084),
             x100 = as.factor(x100),
             x120 = as.factor(x120),
             x202 = as.factor(x202),
             x215 = as.factor(x215),
             x261 = as.factor(x261),
             x291 = as.factor(x291),
             x337 = as.factor(x337),
             x346 = as.factor(x346),
             x364 = as.factor(x364),
             x483 = as.factor(x483),
             x494 = as.factor(x494),
             x533 = as.factor(x533),
             x554 = as.factor(x554),
             x560 = as.factor(x560),
             x586 = as.factor(x586),
             x622 = as.factor(x622),
             x630 = as.factor(x630),
             x689 = as.factor(x689),
             x698 = as.factor(x698),
             x707 = as.factor(x707),
             x720 = as.factor(x720),
             x749 = as.factor(x749),
             x761 = as.factor(x761))
  # janitor::clean_names() %>%
  # mutate_at(vars(-one_of(cat_var, "y")), as.factor)  
  #as.numeric(as.character(data$x001))

#-------------------------------------------------
# mars

# finalize  the workflow
mars_workflow_tuned <- mars_workflow %>% 
  finalize_workflow(select_best(mars_tune, metric = "roc_auc"))

mars_workflow_tuned
# fit testing data to final workflow
mars_fit <- fit(mars_workflow_tuned, train)

# predict the testing data 
mars_pred <- test %>% 
  select(id) %>% 
  bind_cols(predict(mars_fit, new_data = test)) %>% 
  rename(y = .pred_class)

# save results ----
write_csv(mars_pred, file = "attempts/mars_pred.csv")
#-------------------------------------------------
#elastic net

# finalize  the workflow
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tune, metric = "roc_auc"))

en_workflow_tuned
# fit testing data to final workflow
en_fit <- fit(en_workflow_tuned, train)

# predict the testing data 
en_pred <- test %>% 
  select(id) %>% 
  bind_cols(predict(en_fit, new_data = test)) %>% 
  rename(y = .pred)

# save results ----
save(result_table, file = "results/1st_results_table.rda")

write_csv(en_pred, file = "attempt1/en_pred.csv")

#-------------------------------------------------
# random forest 
# finalize  the workflow
rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "roc_auc"))

rf_workflow_tuned
# fit testing data to final workflow
rf_fit <- fit(rf_workflow_tuned, train)

# predict the testing data 
rf_pred <- test %>% 
  select(id) %>% 
  bind_cols(predict(rf_fit, new_data = test)) %>% 
  rename(y = .pred_class)

# save results ----
write_csv(rf_pred, file = "attempts/rf_pred.csv")

#-------------------------------------------------
# mlp
# random forest 
# finalize  the workflow
mlp_workflow_tuned <- mlp_workflow %>% 
  finalize_workflow(select_best(mlp_tune, metric = "roc_auc"))

mlp_workflow_tuned
# fit testing data to final workflow
mlp_fit <- fit(mlp_workflow_tuned, train)

# predict the testing data 
mlp_pred <- test %>% 
  select(id) %>% 
  bind_cols(predict(mlp_fit, new_data = test)) %>% 
  rename(y = .pred_class)

# save results ----
write_csv(mlp_pred, file = "attempts/mlp_pred.csv")

#-------------------------------------------------
#knn

# finalize  the workflow
knn_workflow_tuned <- knn_workflow %>% 
  finalize_workflow(select_best(knn_tune, metric = "roc_auc"))

knn_workflow_tuned
# fit testing data to final workflow
knn_fit <- fit(knn_workflow_tuned, train)

# predict the testing data 
knn_pred <- test %>% 
  select(id) %>% 
  bind_cols(predict(knn_fit, new_data = test)) %>% 
  rename(y = .pred_class)

# save results ----
write_csv(knn_pred, file = "attempts/knn_pred.csv")


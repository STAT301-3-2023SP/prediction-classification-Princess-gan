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
  janitor::clean_names() %>% 
  mutate_at(cat_var, as.factor)
#-------------------------------------------------
#elastic net

# finalize  the workflow
en_workflow_tuned <- en_workflow %>% 
  finalize_workflow(select_best(en_tune, metric = "rmse"))

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
  finalize_workflow(select_best(rf_tune, metric = "rmse"))

rf_workflow_tuned
# fit testing data to final workflow
rf_fit <- fit(rf_workflow_tuned, train)

# predict the testing data 
rf_pred <- test %>% 
  select(id) %>% 
  bind_cols(predict(rf_fit, new_data = test)) %>% 
  rename(y = .pred)

# save results ----
write_csv(rf_pred, file = "attempt1/rf_pred.csv")
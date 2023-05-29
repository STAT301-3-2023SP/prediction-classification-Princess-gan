library(tidymodels)
library(tidyverse)
library(doMC)

load(file = "results/var_info.rda")
load(file = "results/lasso_var.rda")

train <- read_csv("data/train.csv") %>%
  janitor::clean_names() %>% 
  select(-id) %>% 
  mutate_at(cat_var, as.factor)

train_log <- train %>% 
  mutate(y = log(y))

train %>% view()
# test <- read_csv("data/test.csv") %>% 
#   janitor::clean_names()
# can split the training set and do eda on the splitted
# set.seed(1)
# my_split <- initial_split(train, prop = 0.75, strata = y)
# 
# train_data <- training(my_split)
# test_data <- testing(my_split)

#-----------------------------------------
# v-fold cross validation
my_fold <- train %>% 
  vfold_cv(v = 5, repeats = 3, strata = y)

# recipe
load("data/lasso_var.rda")
 #remove "ylog"
# selected_vars <-  selected_vars[selected_vars != "ylog"]
# 
# selected_vars
lasso_vars <- lasso_vars %>% 
  stringr::str_split("_", simplify = TRUE)
lasso_vars <- lasso_vars[,1] %>% unique()

# the union of the lasso, rf, mars variables
 # intersect_vars <- Reduce(intersect, list(lasso_vars, var_select_rf_vars, var_select_mars_vars))
 
# union_vars <- Reduce(union, list(lasso_vars, var_select_rf_vars, var_select_mars_vars))

# select variables to remove
removed_vars <- setdiff(colnames(train), c(lasso_vars, "y"))
removed_vars

union_removed_vars <- setdiff(colnames(reg_train), c(union_vars, "y"))

my_recipe <- recipe(y ~ ., data = train) %>%
  step_rm(!!removed_vars) %>% 
  # step_novel(all_nominal_predictors()) %>% 
  # option to accept new levels in testing dataset(just in case one level doesnt show up in training)
  step_nzv(all_predictors()) %>% 
  step_corr(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) 
  # step_select(any_of(lasso_vars))

my_recipe %>% 
  prep(train) %>% 
  bake(new_data = NULL) %>% 
  view()
  
# some sort of variable reduction
## done in var_selection
# load("data/lasso_var.rda")

lasso_vars

save(my_fold, my_recipe, train_data, test_data,
  file = "results/tuning_setup.rda")

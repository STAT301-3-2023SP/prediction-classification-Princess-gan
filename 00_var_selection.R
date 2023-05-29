# Choosing variables using lasso
# 1.setup an initial recipe (kitchen sink, nzv, normalize, impute)
# 2.set up a tuning lasso workflow, maybe tune penalty but set mixture = 1 to save time
# 3.tune the model
# 4.select the best workflow with the lowest RMSE and fit it to the entire training data
# 5.pipe on extract_fit_engine() and tidy() which will extract a tibble of variables and their coefficients. 
# from here, some variables' coefficients will go to 0, which you can filter out!

library(tidymodels)
library(tidyverse)
library(doMC)

train <- read_csv("data/train.csv") %>% 
  mutate(y = log(y))
#transform? log? logit?

# eda is on a random sample of train to prevent overfitting
# for lower comp times fold on a split dataset
set.seed(3013)
# use either full train or the split train --> doesn't matter
# this prevents overfitting by subsetting into training and testing
fold_var_select <- vfold_cv(train, folds = 5, repeats = 1, strata = y)

init_recipe <- recipe(y ~ ., data = train) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_corr(all_predictors())


### tune a lasso model
lasso_mod <- linear_reg(
  mode = "regression",
  penalty = tune(),
  mixture = 1) %>% 
  set_engine("glmnet")

lasso_params <- extract_parameter_set_dials(lasso_mod) %>% 
  update(penalty = penalty(range = c(0.01, 0.1), trans = NULL))

lasso_grid <- grid_regular(lasso_params, levels = 5)

lasso_workflow <- workflow() %>% 
  add_recipe(init_recipe) %>% 
  add_model(lasso_mod)
 
# set up parallel processing 
registerDoMC(cores = 8)

lass_tune <- tune_grid(
  lasso_workflow,
  resamples = fold_var_select,
  grid = lasso_grid,
  control = control_grid(parallel_over = "everything"),
  metric = metric_set(rmse)
)
  
lasso_final_wflow <- lasso_workflow %>% 
  finalize_workflow(select_best(lass_tune, metric = "rmse"))

# fit final workflow to the entire training data
lasso_fit <- lasso_final_wflow %>% fit(train)

# # get coefficients
# load("data/var_reduct_lasso.rda")
# 
# # mars of rf
# mars_fit %>% 
#   extract_fit_parsnip() %>% 
#   vip::vi()

lasso_var <- lasso_fit %>% 
  tidy()

get_var <- lasso_var %>% 
  filter(estimate !=0, term != "(Intercept)") %>% 
  pull(term)

save(get_var, file = "data/lasso_var.rda")

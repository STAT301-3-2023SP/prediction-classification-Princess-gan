# bt tuning
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

# Handle conflicts
tidymodels_prefer()

# set seed
set.seed(0410)

load("results/tuning_setups.rda")

#-----------------------------------------------------
# set up parallel processing
library(doMC)
detectCores(all.tests = FALSE, logical = TRUE)
registerDoMC(cores = 8)

# define model ---------------------------------------
bt_model <- boost_tree(
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost", importance = "impurity") #let's use variable importance

# check tuning parameters
parameters(bt_model)

# set up tuning grid ----
bt_params <- hardhat::extract_parameter_set_dials(bt_model) %>% 
  update(mtry = mtry(range = c(1, 15)))

bt_grid <- grid_regular(bt_params, levels = 5)

# Set up workflow -------------------------------------
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(my_recipe)

# Tune grid
# clear and start timer 
tic.clearlog()
tic("Boosted tree")

bt_tune <- tune_grid(
  bt_workflow,
  resamples = my_fold,
  grid = bt_grid,
  control = control_grid(save_pred = TRUE, #create an extra column for each prediction
                         save_workflow = TRUE, # let's use extract_workflow
                         parallel_over = "everything"),
  metrics = metric_set(rmse))

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

bt_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)
#runtime is (toc - tic) value

# stopCluster(cl)

save(bt_tune, bt_tictoc, 
     file = "results/tuned_bt.rda")

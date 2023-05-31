# rf tuning
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

# Handle conflicts
tidymodels_prefer()

# set seed
set.seed(0410)

load("results/tuning_setup.rda")

#-----------------------------------------------------
# set up parallel processing
library(doMC)
registerDoMC(cores = 8)

# define model ---------------------------------------
rf_model <- rand_forest(
  mode = "classification",
  mtry = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger", importance = "impurity") #let's use variable importance

# check tuning parameters
parameters(rf_model)

# set up tuning grid ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(range = c(1, 15)))

rf_grid <- grid_regular(rf_params, levels = 5)

# Set up workflow -------------------------------------
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(my_recipe)

# Tune grid
# clear and start timer 
tic.clearlog()
tic("Random forest")

rf_tune <- tune_grid(
  rf_workflow,
  resamples = my_fold,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE, #create an extra column for each prediction
                         save_workflow = TRUE, # let's use extract_workflow
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(model = time_log[[1]]$msg,
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)
#runtime is (toc - tic) value

# stopCluster(cl)

save(rf_workflow, rf_tune, rf_tictoc, 
     file = "results/tuned_rf.rda")


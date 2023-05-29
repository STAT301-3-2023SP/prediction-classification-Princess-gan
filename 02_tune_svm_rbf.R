# svm tuning
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

# Handle conflicts
tidymodels_prefer()

# set seed
set.seed(0420)

load("results/tuning_setup.rda")

#-----------------------------------------------------
# set up parallel processing
library(doMC)
registerDoMC(cores = 8)

# define model ---------------------------------------
svm_model <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>% 
  set_mode("regression") %>% 
  set_engine("kernlab", importance = "impurity") #let's use variable importance

# check tuning parameters
parameters(svm_model)

# set up tuning grid ----
svm_params <- hardhat::extract_parameter_set_dials(svm_model)

svm_grid <- grid_regular(svm_params, levels = 5)

# Set up workflow -------------------------------------
svm_workflow <- workflow() %>% 
  add_model(svm_model) %>% 
  add_recipe(my_recipe)

# Tune grid
# clear and start timer 
tic.clearlog()
tic("Support vector machine_rbf")

svm_rbf_tune <- tune_grid(
  svm_workflow,
  resamples = my_fold,
  grid = svm_grid,
  control = control_grid(save_pred = TRUE, #create an extra column for each prediction
                         save_workflow = TRUE, # let's use extract_workflow
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

svm_rbf_tictoc <- tibble(model = time_log[[1]]$msg,
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)
#runtime is (toc - tic) value

# stopCluster(cl)

save(svm_rbf_tune, svm_rbf_tictoc, 
     file = "results/tuned_svm_rbf.rda")

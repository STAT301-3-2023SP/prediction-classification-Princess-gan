# mlp tuning:
# Single Layer Neural Network (multilayer perceptron)
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
mlp_model <- mlp(
  mode = "classification",
  hidden_units = tune(),
  penalty = tune(),
) %>% 
  set_engine("nnet", importance = "impurity") #let's use variable importance

# check tuning parameters
parameters(mlp_model)

# set up tuning grid ----
mlp_params <- hardhat::extract_parameter_set_dials(mlp_model)

mlp_grid <- grid_regular(mlp_params, levels = 5)

# Set up workflow -------------------------------------
mlp_workflow <- workflow() %>% 
  add_model(mlp_model) %>% 
  add_recipe(my_recipe)

# Tune grid
# clear and start timer 
tic.clearlog()
tic("Single Layer Neural Network (multilayer perceptron)")

mlp_tune <- tune_grid(
  mlp_workflow,
  resamples = my_fold,
  grid = mlp_grid,
  control = control_grid(save_pred = TRUE, #create an extra column for each prediction
                         save_workflow = TRUE, # let's use extract_workflow
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)

toc(log = TRUE)

time_log <- tic.log(format = FALSE)

mlp_tictoc <- tibble(model = time_log[[1]]$msg,
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)
#runtime is (toc - tic) value

# stopCluster(cl)

save(mlp_workflow, mlp_tune, mlp_tictoc, 
     file = "results/tuned_mlp.rda")

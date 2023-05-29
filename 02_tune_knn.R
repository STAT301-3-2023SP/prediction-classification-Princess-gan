# Knn Tuning

library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

tidymodels_prefer()

load("results/tuning_setup.rda")

#######################################################################################
# set up parallel processing
parallel::detectCores()
registerDoMC(cores = 8) # put in each r script

#######################################################################################
# define model engine and workflow

knn_model <-
  nearest_neighbor(neighbors = tune() 
                   #weight_func = tune(), dist_power = tune()
                   ) %>%
  set_engine('kknn') %>%
  set_mode('regression')


knn_params <- extract_parameter_set_dials(knn_model)

knn_grid <- grid_regular(knn_params, levels = 5)

knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(my_recipe)
  
#######################################################################################
# tune grid

knn_tune <- tune_grid(
  knn_workflow, 
  resamples = my_fold,
  grid = knn_grid,
  control = control_grid(save_pred = TRUE, 
                         # create an extra column for each prediction, not necessary, for graphic purpose
                         save_workflow = TRUE,
                         # let's tiy yse extract_workflow later on
                         parallel_over = "everything"),
  metrics = metric_set(rmse)
)


save(knn_tune, file = "results/tuned_knn.rda")

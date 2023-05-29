# mars tuning

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

#-------------------------------
# define model engine and workflow
mars_model <- mars(
  num_terms = tune(),
  prod_degree = tune()
) %>% 
  set_mode("regression") %>% 
  set_engine("earth")

num_terms

mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 23)))

mars_grid <- grid_regular(mars_params, levels = 5)
# Set up workflow -------------------------------------
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(my_recipe)

# Tune grid
# clear and start timer 
 tic.clearlog()
 tic("Multivariate adaptive regression splines ")

mars_tune <- tune_grid(
  mars_workflow,
  resamples = my_fold,
  grid = mars_grid,
  control = control_grid(save_pred = TRUE, #create an extra column for each prediction
                         save_workflow = TRUE, # let's use extract_workflow
                         parallel_over = "everything"),
  metrics = metric_set(rmse))

 toc(log = TRUE)

 time_log <- tic.log(format = FALSE)

 mars_tictoc <- tibble(model = time_log[[1]]$msg,
                      runtime = time_log[[1]]$toc - time_log[[1]]$tic)
#runtime is (toc - tic) value

stopCluster(cl)

save(mars_tune, mars_tictoc, 
     file = "results/tuned_mars.rda")

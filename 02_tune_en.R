# Elastic net tuning

library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

tidymodels_prefer()

load("results/tuning_setup.rda")

######################################
# set up parallel processing
library(doMC)
registerDoMC(cores = 8)

#######################################

# define model engine and workflow
en_model <- linear_reg(penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet") 

# set up tuning grid --------------------------------------------------  
en_params <- extract_parameter_set_dials(en_model)

en_grid <- grid_regular(en_params, levels = 5)

# Set up workflow 
en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(my_recipe)

##########################################

# Tune grid
# # clear and start timer 
# tic.clearlog()
# tic("Elastic Net")

en_tune <- tune_grid(
  en_workflow,
  resamples = my_fold,
  grid = en_grid,
  control = control_grid(save_pred = TRUE, #create an extra column for each prediction
                         save_workflow = TRUE, # let's use extract_workflow
                         parallel_over = "everything"),
  metrics = metric_set(rmse))

# toc(log = TRUE)
# time_log <- tic.log(format = FALSE)
# en_tictoc <- tibble(model = time_log[[1]]$msg,
#                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)
# #runtime is (toc - tic) value

# stopCluster(cl)

save(en_workflow, en_tune,
     file = "results/tuned_en.rda")



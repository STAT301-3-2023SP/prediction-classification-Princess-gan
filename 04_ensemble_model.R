# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)
library(gt)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("model_info/knn_res.rda")
load("model_info/svm_res.rda")
load("model_info/lin_reg_res.rda")

# Load split data object & get testing data
load("data/wildfires_split.rda")

wildfires_test <- wildfires_split %>% testing()

# Create data stack ----
stacks

wildfire_data_stack <- stacks() %>% 
  add_candidates(knn_res) %>% 
  add_candidates(svm_res) %>% 
  add_candidates(lin_reg_res) # can only run after adding "control" arg in fit_resample
as_tibble(wildfire_data_stack)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2) # higher penalty value, going to force more coefficient values to 0


# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)
wildfire_stack_blend <- wildfire_data_stack %>% 
  blend_predictions(penalty = blend_penalty) 

# wildfire_stack_blend <- wildfire_stack_blend %>% 
#   as_tibble() %>%
#   knitr::kable()

# Save blended model stack for reproducibility & easy reference (Rmd report)
save(wildfire_stack_blend, file = "model_info/wildfires_stack_blend.rda")

# Explore the blended model stack
autoplot(wildfire_stack_blend) # shows that higher penalty reduces rmse

autoplot(wildfire_stack_blend, type = "members") #

autoplot(wildfire_stack_blend, type = "weights") # svm picks 2 of many models with diff tuning params

# fit to ensemble to entire training set ----
wildfire_model_fit <- wildfire_stack_blend %>% 
  fit_members()

wildfire_model_fit %>% knitr::kable()
# Identify which model configurations were assigned what stacking coefficients
collect_parameters(wildfire_model_fit, "svm_res") %>% 
  print(n = 25)

# Save trained ensemble model for reproducibility & easy reference (Rmd report)
pred <- wildfires_test %>%
  select(burned) %>% 
  bind_cols(predict(wildfire_model_fit, wildfires_test))

pred
# Explore and assess trained ensemble model
pred_members <- wildfires_test %>%
  select(burned) %>% 
  bind_cols(predict(wildfire_model_fit, wildfires_test, members = TRUE)) %>% # shows predictions of each individual model
  rename(ensemble = .pred)
pred_members
   #table
pred_members_table <- pred_members %>% 
  slice(1:10) %>% 
  knitr::kable()

pred_members_table


plot <- ggplot(pred_members) +
  aes(x = burned, 
      y = ensemble) +
  geom_point() + 
  coord_obs_pred()+
  geom_abline()

# rmse
pred_rmse<- pred_members %>% 
  map_df(rmse, truth = burned, data = pred_members) %>% 
  mutate(member = colnames(pred_members)) %>% 
  filter(member != "burned") %>% 
  arrange(.estimate) # rmse lowest to highest

 #table
pred_rmse_table <- pred_rmse %>% 
  #slice(1:10) %>% 
  knitr::kable()

pred_rmse_table

save(pred_members_table, file = "members.rda")
save(pred_rmse_table, file = "rmse.rda")

#save the weights in ensemble model
weights <- tibble(
  member = c("lin_reg_res_1_1", "svm_res_1_17", "svm_res_1_16", "svm_res_1_19", "knn_res_1_09", "knn_res_1_10", "knn_res_1_11"),
  type = c("linear_reg", "svm_rbf", "svm_rbf", "svm_rbf", "nearest_neighbor", "nearest_neighbor", "nearest_neighbor"),
  weight = c(0.871, 0.326, 0.108, 0.0639, 0.0304, 0.0199, 0.00304)
) %>% 
  knitr::kable()

save(weights, file = "weights.rda")
# gtsave(pred_members_table, file = "members.html")
# gtsave(pred_rmse_table, file = "rmse.html")

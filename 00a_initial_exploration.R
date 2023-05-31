# initial exploration
library(tidymodels)
library(tidyverse)
tidymodels_prefer()
library(visdat)
library(naniar)
library(doMC)

# parallel processing
detectCores(all.tests = FALSE, logical = TRUE)
registerDoMC(cores = 12)

## lasso/rf to reduce variables, corr test, remove highly correlated variables 

train <- read_csv("data/train.csv") %>% 
  janitor::clean_names()
test <- read_csv("data/test.csv") %>% 
  janitor::clean_names()
# can split the training set and do eda on the splitted
set.seed(1)
my_split <- initial_split(train, prop = 0.75, strata = y)

train_data <- training(my_split)
test_data <- testing(my_split)

#-------------------------------------
# create fold 
# distribution of y 
train %>% 
#   ggplot(mapping = aes(x=log10(y)))+ 
#   geom_density()
# 
# MASS::boxcox(lm(y~1, train_data))
# recommending a potential log transform

# left skewed, applied log10 transformation
train <- train %>% 
  mutate(ylog = log10(y))

# ---------------------------------
# function for exploration
boxplot_fun <- function(var = NULL){
  ggplot(train_data, aes(x = var, y = y)) +
    geom_boxplot()
}

boxplot_log_fun <- function(var = NULL){
  ggplot(train_data, aes(x = factor(!!sym(var)), y = log(y))) +
    geom_boxplot()
}

#----------------------------------
# explore y
ggplot(train_data, aes(x = y)) + 
  geom_histogram()

MASS::boxcox(lm(y ~1, train_data))
# recommending a potential log transform 

train %>% 
  ggplot(mapping = aes(x=log10(y)))+ 
  geom_density()

# left skewed, applied log10 transformation
train <- train %>% 
  mutate(ylog = log10(y))

# --------------------------------------
# missingness 
missing_lst <- list()

for(var in colnames(train_data)){
  missing_lst[var] <- train_data %>% 
    select(any_of(var)) %>% 
    filter(is.na(!!sym(var))) %>% 
    summarize(num_missing = n())
}


missing_tb1 <- enframe(unlist(missing_lst))

# calculate prop of missingness
missing_tb1 %>% 
  mutate(pct = value/4034) %>% 
  arrange(desc(pct))

#---------------------------------------
# remove zero_var
# step_zv() will also do this
var_lst <- list()

for(var in colnames(train_data)){
  var_lst[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(sd = sd(!!sym(var), na.rm = TRUE))
}

var_tbl <- enframe(unlist(var_lst))

# remove zero variance
# mental note that high variance might benefit from a transformation 

zero_var <- var_tbl %>% 
  filter(value ==0) %>% 
  pull(name) # tidyverse version of data$variable

# update training data to remove unwanted variables
train_data <- train_data %>% 
  select(!all_of(zero_var))

#---------------------------------------
# high correlation 
# step_corr could potentially do the same thing 

#------------------------------------------
# miscoded categorical variables
cat_lst <- list()

for(var in colnames(train_data)){
  cat_lst[var] <- train_data %>% 
    select(any_of(var)) %>% 
    # count of unique values in variable
    summarize(unique = length(unique(!!sym(var))))
}

cat_tbl <- enframe(unlist(cat_lst))
view(cat_tbl)


cat_var <- cat_tbl %>% 
  filter(value <= 10) %>% 
  pull(name)

map(cat_var, boxplot_fun)

# -------------------------------------
# # create fold
# fold1 <- vfold_cv(train, v=5, repeats = 3, strata = y)

boxplot_fun(var = "x025")

boxplot_log_fun(var = "x025")

map(cat_var, boxplot_fun)

map(cat_var, boxplot_log_fun)

# choose if any look like they have a relationship
# turn to factor with mutate
# do so in both training and testing 

# could write a function for histogram to explore variable
# could write a function for scatterplots to explore relation with y

# if I decide to transform 
train_data <- train_data %>% 
  mutate(x742 = factor(x742))

test_data <- test_data %>% 
  mutate(x742 = factor(x742))
# from here save out my "clean datasets"
# consider a variable reduction technique such as lasso or random forest

## written in 01_var_selection
save(zero_var, cat_var, file = "results/var_info.rda")
write_rds(train_data, "data/train_data.rds")


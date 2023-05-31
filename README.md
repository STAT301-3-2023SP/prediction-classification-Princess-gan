## Classification Activity

Hi! In this repository, the R scripts can reproduce all the tuning model results and predictions. All the files are named in Arabic order.

To recreate, you can:
- start with 00a and 00b, which are EDA, the initial exploration, and the variable selection process before recipe. 
- 01 provides the essential initial setups for tuning. 
- All the 02 R scripts can reproduce the tuning results for models (stored in `results` folder). 
- 03 R script stores the RMSE metric applied to the tuned model with the training dataset
- 04 R scipts are tuned ensemble model candidates and ensemble model. 
- 05 R script stores all the final predictions for the test dataset, which are all written up as `.csv` files in `attempts` folder. These are results posted to Kaggle! 

The `attempts` folders hold all the predictions, `results` folder hold all the tuned results, `table` folder holds the material for memo. 

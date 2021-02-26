#Elle Loveseth
#script to identify best forecast parameters for individual forecasts at scale using the prophet forecasting package.

#load required packages
library(prophet) #forecasting https://facebook.github.io/prophet/
library(dplyr) #data manipulation
library(purrr) #apply functions to lists

##choose forecast accuracy metric to identify best parameters
metric <- 'rmse'  #options: 'mse' | 'rmse'

##chose forecast length in weeks (prophet also allows daily data)
forecast_length <- 52

#summarise forecasting variable by week in training dataframe
#prophet reads ds as date and y as the forecasting variable
x <- x %>%
 group_by(uniqueid, ds) %>%
 summarise(y = sum(y)) %>%
 ungroup()

#add binary covid indicator (can also add additional regressors at this stage)
#note: all regressors must have future values forecasted
x$covid <- ifelse(x$ds >= '2020-03-01', 1, 0)

#split training set into individual lists by uniqueid
x <- x %>%
  split(.$uniqueid)

#create a grid with each combination of hyperparameters
#this can be used to cast a wide net with wide ranges of parameters
#once general best parameters are identified, less variation can be considered
param_grid <- expand.grid(weekly_seasonality = FALSE,
                          yearly_seasonality = c(5, 10),
                          growth = 'linear',
                          seasonality_mode = 'multiplicative',
                          seasonality_prior_scale = c(2, 5, 10),
                          changepoint_range = c(.6, .8),
                          changepoint_prior_scale = c(.03, .1, .5),
                          holidays = .holidays,
                          holidays_prior_scale = c(2, 5, 10))

#duplicate the grid for each individual forecast in list form
l_param_grid <- rep(list(param_grid), length(x))
names(l_param_grid) <- names(x)

print("Model Specification Grid Complete.")

#create empty lists to append results
results <- data.frame(matrix(ncol = 2))
results <- vector(mode = 'numeric')
results <- rep(list(results), length(x))
names(results) <- names(x)

#run forecast through loop
for (i in seq_len(nrow(param_grid))) {
  #store parameters for each loop
  parameters <- param_grid[i, ]
  
  #fit base model
  m <- prophet(weekly.seasonality = parameters$weekly_seasonality,
               yearly.seasonality = parameters$yearly_seasonality,
               growth = parameters$growth,
               seasonality.mode = parameters$seasonality_mode,
               seasonality.prior.scale = parameters$seasonality_prior_scale,
               changepoint.range = parameters$changepoint_range,
               changepoint.prior.scale = parameters$changepoint_prior_scale,
               holidays = parameters$holidays,
               holidays.prior.scale = parameters$holidays_prior_scale)
  
  #add regressors to model
  m <- add_regressor(m, name = 'covid', prior.scale = 3, standardize = 'TRUE', mode = 'additive')

  #create replications of model for each uniqueid in x
  m <- rep(list(m), length(x))
  names(m) <- names(x)

  #fit model to each list in x using the map2 function in purrr
  #map and map2 allows for faster processing by applying the function to each list
  m <- map2(m, x, fit.prophet)

  #create future lists for each forecast
  future <- map(m, make_future_dataframe, periods = forecast_length, freq = 'week')
  
  #flatten future lists into a dataframe to merge regressors
  future <- bind_rows(future, .id = 'uniqueid')

  future$ds <- as.Date(future$ds)

  #merge future values of regressors
  future$covid <- ifelse(future$ds >= '2020-03-01', 1, 0)

  #split dataframe back into lists
  future <- future %>%
    split(.$uniqueid)

  #generate the forecast 
  forecast <- map2(m, future, predict)

  #we'll use built in performance metrics to test performance
  #this runs multiple scenarios holding back set amounts of data to calculate performance
  m.cv <- map(m, cross_validation, initial = 120, period = 20, horizon = 53, units = 'weeks')
  m.p <- map(m.cv, performance_metrics)
  m.p <- map(m.p, summarise, rmse = mean(rmse), mse = mean(mse))
  
  #bind performance metrics to results table
  results <- map2(results, m.p, rbind)
}

#bind results with corresponding parameters, then flatten into a dataframe
best_params <- map2(l_param_grid, results, cbind)
best_params <- bind_rows(best_params, .id = 'uniqueid')

#use chosen metric to identify best parameters and filter dataframe
if (metric == 'rmse') {
  best_params <- best_params %>%
    group_by(id) %>%
    slice(which.min(rmse))
} else {
  best_params <- best_params %>%
    group_by(id) %>%
    slice(which.min(mape))
}

print('Best forecast parameters identified.')

#you are now left with a table that contains the forecast parameters with best fit for each uniqueid
#assemble another grid using these results to fine tune a forecast, or use the table to run your final forecast scenario!

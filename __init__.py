# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:40:17 2021

@author: Michael Hauptman

This module returns change points in a time series, using Kalman filters and EVT as described in https://www.robots.ox.ac.uk/~sjrob/Pubs/LeeRoberts_EVT.pdf

This module first prepares a Kalman state-space of a time series as a random walk plus linear change, where the linear change factor (aka "Trend) is a hidden variable and so dynamic

In order to get a Kalman filter of a time series, plus change points in the trend, do the following:

1. Prepare your time series as data plus Panda dates
2. Create  the necessary Kalman representation by creating a "session" object by calling the ChangePoint class, e.g.:
	Session=ChangePointDetector.ChangePointDetectorSession(data,dates). 'SeasonalityPeriods' is an optional input, e.g 12 = calendar month seasonality
3. Determine the changepoints by running the ChangePointDetectorFunction on your "session", e.g.
	Results=Session.ChangePointDetectorFunction()
4. This will return a "Results" object that contains the following:
	- ChangePoints.  This is a list of 0s and 1s the length of the data, where 1s represent changepoints
	- Prediction.  This is the Kalman smoothed actuals, plus a 3 period forecast.  Note no forecast will be made if there is a changepoint in the last 3 		dates
	- PredictionVariance
	- ExtendedDates.  These are the original dates plus 3 exta for the forecast (if a forecast has been made)
	- Trend.  This is the linear change factor
	- TrendVariance.  Variance of the trend



"""




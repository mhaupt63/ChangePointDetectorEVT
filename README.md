# ChangePointDetector
 This module takes a time series and returns:  (a) the piecewise underlaying linear trend, and (b) the times where there is a change in the underlying trend
 
 We use a Kalman Filter to find the piecewise underlying linear trend, with a state space representation of seasonality and an underlying linear trend.  We initialise the parameters of the Kalman filter using a Least Squares estimator out of "statsmodels'
 
 For the change detector on the underlying trend we use another Kalman filter, this time a single period autoregression.  We then consider the Malhalanobis distance between that filter output, using a Gumbel distribution to decide where the increase in distance likely indicates a change in trend.  This implements the approach described by Lee & Roberts at
  https://www.robots.ox.ac.uk/~sjrob/Pubs/LeeRoberts_EVT.pdf

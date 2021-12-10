# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:56:11 2021
v0.8 28Oct21

@author: Michael Hauptman

CHANGE POINT DETECTOR

This module takes a time series and returns the uunderlying linear trend and changpoints. It uses EVT as described in https://www.robots.ox.ac.uk/~sjrob/Pubs/LeeRoberts_EVT.pdf

INSTRUCTIONS:

1. from ChangePointDetector import ChangePointDetector 
2. Prepare your time series as data plus Panda dates
3. Create  the necessary Kalman representation by creating a "session" object by calling the ChangePoint class, e.g.:
	Session=ChangePointDetector.ChangePointDetectorSession(data,dates). 'SeasonalityPeriods' is an optional input, e.g 12 = calendar month seasonality
4. Determine the changepoints by running the ChangePointDetectorFunction on your "session", e.g.
	Results=Session.ChangePointDetectorFunction()
5. This will return a "Results" object that contains the following:
	- ChangePoints.  This is a list of 0s and 1s the length of the data, where 1s represent changepoints
	- Prediction.  This is the Kalman smoothed actuals, plus a 3 period forecast.  Note no forecast will be made if there is a changepoint in the last 3 		dates
	- PredictionVariance
	- ExtendedDates.  These are the original dates plus 3 exta for the forecast (if a forecast has been made)
	- Trend.  This is the linear change factor
	- TrendVariance.  Variance of the trend


"""

import numpy as np
from numpy.linalg import inv
from statsmodels.tsa.statespace.mlemodel import MLEModel
# import math
from math import sqrt, log, exp,pi
from scipy.stats.distributions import chi2
from dateutil.relativedelta import relativedelta

class ModuleResults:
    def __init__(self,Trend,TrendVariance,ChangePoints,Prediction,PredictionVariance,ExtendedDates):
            self.Trend=Trend
            self.TrendVariance=TrendVariance
            self.ChangePoints=ChangePoints
            self.Prediction=Prediction
            self.ExtendedDates=ExtendedDates
            self.PredictionVariance=PredictionVariance

class TimeSeriesData:
        def __init__(self,data,dates):
            self.data=data
            self.dates=dates
            #Determine periodicity
            Delta=(dates[1]-dates[0]).days
            if Delta in range(27,32):
                Period = "months"
            elif Delta ==7:
                Period = 'weeks'
            else:
                Period ="days"
            self.Period=Period

            
class SeasonalStateArrays:
    def __init__(self,endog,SeasonalityPeriods=1):
        if SeasonalityPeriods ==0:
            A=np.diag(np.ones(1))
            H=np.ones(1)
        elif SeasonalityPeriods ==1:
            A=np.array([[1,1],[0,1]])
            H=np.array([1,0])
        else:
            B=np.zeros((SeasonalityPeriods,1))
            C=-1*np.ones(SeasonalityPeriods+1)
            C[0]=0
            C[-1]=0
            A=np.hstack([A,B])
            A[0,-1]=1
            D=np.zeros(SeasonalityPeriods+1)
            D[-1]=1
            A[-1,:]=D
            A=np.vstack([A[0,:],C,A[1:,:]]).astype('float')
            H=np.zeros((SeasonalityPeriods+1))
            H[0]=1
            H[1]=1
        P= 0.01*np.eye(SeasonalityPeriods+1)
        Mu=np.zeros(SeasonalityPeriods+1)
        Mu[0]=endog[0]

        self.transition=A
        self.design=np.array([H])
        self.Mu=Mu
        self.P=P
        
class KalmanModel(MLEModel):
    def __init__(self, endog, transition,design,**dates):
        super().__init__(endog, k_states = transition.shape[0], **dates)

        self['transition']=transition
        self['design']=design
        self['selection']=np.eye(transition.shape[0])
        self.initialize_approximate_diffuse()

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self['state_cov']=np.eye(self.k_states)
        for i in range(self.k_states):
            self['state_cov'][i,i]=params[0]
        for j in range(self['design'].shape[0]):
            self['obs_cov',j, j] = params[1] ** 2
            
def kgain(P, H, R):
    IS =  H @ P @ H.T + R # the Covariance of predictive mean of Y
    K = P @ H.T @ inv(IS) #State covariance in observation space divided by  Covariance of predictive mean of Y
    return K

def MDCalc(X,Xold,P):
        return(
        sqrt((X-Xold).T@inv(P)@(X-Xold))  #Mahalanobis distance
        )
    
def MD_ProbCalc(MD,DF):
    return(
    chi2.cdf(MD, df=DF)
    )

class MyKalmanFilter:
    def __init__(self, TS_Data, SeasonalityPeriods=1):
        OriginalData=TS_Data.data
        NormalisedData=OriginalData/np.linalg.norm(OriginalData)  
        self.OriginalData=OriginalData
        self.data=NormalisedData
        self.dates=TS_Data.dates
        self.SeasonalityPeriods=SeasonalityPeriods
        self.StateArrays=SeasonalStateArrays(self.data,SeasonalityPeriods)
        
    def InitialiseFilter(self):
        NormalisedData=self.data
        kf_model=KalmanModel(NormalisedData,transition=self.StateArrays.transition,design=self.StateArrays.design)
        
        ObservationVar=0.1
        start_params = [np.var(NormalisedData)**3, ObservationVar] #State transition convariance and observation covariance. These settings will underfit the curve
        kf_model_fit = kf_model.fit(start_params,maxiter = 20,method = 'bfgs', hessian= 'true')
        Growth=[kf_model_fit.filtered_state[0,x] for x in range(kf_model_fit.filtered_state.shape[1])]
        Base=np.mean(abs(NormalisedData-Growth))

        Success=False
        i=3
        while Success ==False and i>0:  #Increae state covariance until fitting improves over underfit base case against target
            start_params = [np.var(NormalisedData)**(i), ObservationVar]
            Target=0.3
            kf_model_fit = kf_model.fit(start_params,maxiter = 20,method = 'bfgs', hessian= 'true')
            Growth=[kf_model_fit.filtered_state[0,x] for x in range(kf_model_fit.filtered_state.shape[1])]
            if np.mean(abs(NormalisedData-Growth))/Base<Target:  
                Success =True
            i-=0.05
        
        #Determine kalman gain
        K_gain=np.zeros(self.SeasonalityPeriods+1)
        for i in range(kf_model_fit.filtered_state.shape[1]):
            P=kf_model_fit.filtered_state_cov[:,:,i]
            KGain=kgain(P, kf_model['design'], kf_model['obs_cov'])
            K_gain=np.vstack([K_gain,KGain.reshape([1,self.SeasonalityPeriods+1])[0]])
            
        self.KalmanModel=kf_model
        self.KalmanModelFit=kf_model_fit
        self.K_gain=K_gain
        

   
class ChangePointDetectorSession:
    def __init__(self, data,dates, SeasonalityPeriods=1):
        self.ts=TimeSeriesData(data,dates)
        self.TimeseriesModel = MyKalmanFilter(self.ts)
        self.TimeseriesModel.InitialiseFilter()
        self.SeasonalityPeriods=SeasonalityPeriods
    
    def ChangePointDetectorFunction(self):
    #This is the main function    
        gs_orig=self.TimeseriesModel.KalmanModel.smooth(self.ts.data).smoothed_state[-1,:]  #Get underlying trend
        gs=TimeSeriesData(gs_orig,self.ts.dates)
        OnePeriodRegressionModel=MyKalmanFilter(gs,SeasonalityPeriods=0)
        OnePeriodRegressionModel.InitialiseFilter()
        
        #Capture variance
        
        
        MDout,MD_Prob = [],[]
        DF=1
        for i in range(OnePeriodRegressionModel.KalmanModelFit.filtered_state.shape[1]):
        
            X=OnePeriodRegressionModel.KalmanModelFit.filtered_state[:,i]
            P=OnePeriodRegressionModel.KalmanModelFit.filtered_state_cov[:,:,i]
            if i==0:
                Xold=X
            else:
                Xold=OnePeriodRegressionModel.KalmanModelFit.filtered_state[:,i-1]
            try:
                MD=MDCalc(X,Xold,P)
            except:
                MD=0
            MDPr=MD_ProbCalc(MD,DF)
            MDout.append(MD)
            MD_Prob.append(MDPr)
        
        #Find changes in underlying trend
        EVMT=[]
        for i in range(self.SeasonalityPeriods+1):
            EVMT.append(0)
        ChangePoints=[]
        ChangePoint=0
        PrLmOld=0
        for t in range(self.SeasonalityPeriods+2,len(MDout)+1):
            EV=0
            EV_Old=1  #Prl1=1 when t=1
            #Iterate over a window expanding back to the last change point 
            for m in range(1,t-ChangePoint+1):
        
                loc=sqrt(2*log(m+1))-(2*pi+log(log(m+1)))/(2*sqrt(log(m+1)))
                scale=1/sqrt(2*log(m+1))
                if m<=2:
                    EV_conditionalm=exp(-exp(-(MDout[t-1])))
                else:
                    EV_conditionalm=exp(-exp(-(MDout[t-1]-loc)/scale))
        
                if m==1:
                    PrLm=EV_Old
                else:
                    PrLm=(1-EV_Old)*PrLmOld
        
                EV+=EV_conditionalm*PrLm
        
        #         print(f't{ t},m{ m},MD {MDout[t-m]}, EV_conditionalm {EV_conditionalm},PrLmOld {PrLmOld},PrLm, {PrLm},EV {EV}') 
                PrLmOld=PrLm
                EV_Old=EV
            #Changepoint if likelihood > 95% and no changepoints in last 2 periods
            if   EV>0.95 and (t-(ChangePoint+1))>2:  
                EV=1
                ChangePoint=t-1
                PrLmOld=0
                EV_Old=1
        #         print('Change at ',t,MD_Prob[t-1])
                ChangePoints.append(ChangePoint)
        
            EVMT.append(EV)
            
        ExtendedDates=list(self.ts.dates)
        #Do forecast if no change points in last 3 periods
        if  not sum([i > len(self.ts.data)-3 for i in ChangePoints]):
            #Start with historical data
            Prediction=self.TimeseriesModel.KalmanModelFit.predict(0,len(self.ts.data))
            
            #Take average of last 3 growth factors
            Trend=[self.TimeseriesModel.KalmanModelFit.filtered_state[1,x] for x in range(self.TimeseriesModel.KalmanModelFit.filtered_state.shape[1])]
            if abs(Trend[-1])<abs(sum(Trend[-3:])/3):
                AverageGrowth=Trend[-1]
            else:
                AverageGrowth=sum(Trend[-3:])/3                 
            
            #Add forecast to historical data
            for i in range(3):
                Prediction=np.append(Prediction,Prediction[-1]+AverageGrowth)
            
            #Prediction=self.TimeseriesModel.KalmanModelFit.predict(0,len(self.ts.data)+3)
            #add extra dates
            
            Period= self.ts.Period
            for i in range(0,3):
                if Period =="months":
                    NextDate=self.ts.dates[-1]+relativedelta(months=i+1)
                elif Period == 'weeks':
                    NextDate=self.ts.dates[-1]+relativedelta(weeks=i+1)
                else:
                    NextDate=self.ts.dates[-1]+relativedelta(days=i+1)
                ExtendedDates.append(NextDate)
                
            PredictionVariance=self.TimeseriesModel.KalmanModelFit.get_prediction(0,len(self.ts.data)+3).var_pred_mean
        else:
            Prediction=self.TimeseriesModel.KalmanModelFit.predict(0,len(self.ts.data))
            PredictionVariance=self.TimeseriesModel.KalmanModelFit.get_prediction(0,len(self.ts.data)).var_pred_mean
            ExtendedDates=self.ts.dates
        
        Prediction=Prediction*np.linalg.norm(self.ts.data)
        PredictionVariance=PredictionVariance*(np.linalg.norm(self.ts.data)**2)
#         PredictionVariance=PredictionVariance*Prediction
        
        Trend=[self.TimeseriesModel.KalmanModelFit.filtered_state[1,x] for x in range(self.TimeseriesModel.KalmanModelFit.filtered_state.shape[1])]
        Trend=[x*np.linalg.norm(self.ts.data) for x in Trend]
        
        TrendVariance=[self.TimeseriesModel.KalmanModelFit.filtered_state_cov[-1,-1,x] for x in range(self.TimeseriesModel.KalmanModelFit.filtered_state_cov.shape[2])]
        TrendVariance=[x*np.linalg.norm(self.ts.data) for x in TrendVariance]
        
        ChangePoints=[1 if x in ChangePoints else 0 for x in range(len(gs_orig))]
        
        Results= ModuleResults(Trend=Trend,TrendVariance=TrendVariance,ChangePoints=ChangePoints,Prediction=Prediction,\
                              PredictionVariance=PredictionVariance,ExtendedDates=ExtendedDates)
        
        return Results
    
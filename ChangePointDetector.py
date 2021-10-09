# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:56:11 2021

@author: Michael Hauptman

CHANGE POINT DETECTOR

This module takes a time series and returns:

the underlaying linear trend
the times where there is a change in the time series, either a one-off spike or change in underlying trend

It uses EVT as described in https://www.robots.ox.ac.uk/~sjrob/Pubs/LeeRoberts_EVT.pdf


"""

import numpy as np
from numpy.linalg import inv
from statsmodels.tsa.statespace.mlemodel import MLEModel
# import math
from math import sqrt, log, exp,pi
from scipy.stats.distributions import chi2
from dateutil.relativedelta import relativedelta

class ModuleResults:
    def __init__(self,Trend,TrendVariance,ChangePoints,Prediction,ExtendedDates):
            self.Trend=Trend
            self.TrendVariance=TrendVariance
            self.ChangePoints=ChangePoints
            self.Prediction=Prediction
            self.ExtendedDates=ExtendedDates

class TimeSeriesData:
        def __init__(self,data,dates):
            self.data=data
            self.dates=dates
            #Determine periodicity
            Delta=(dates[1]-dates[0])
            if Delta in range(27,32):
                Period = "months"
            elif Delta ==1:
                Period = 'days'
            elif Delta ==7:
                Period = 'weeks'
            else:
                Period ="months"
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
#         self.initialize_known(Mu, P)
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
    K = P @ H.T @ inv(IS)
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
        
        start_params = [np.var(NormalisedData)**3, 0.1]  ##These settings will underfit the curve
        kf_model_fit = kf_model.fit(start_params,maxiter = 20,method = 'bfgs', hessian= 'true')
        Growth=[kf_model_fit.filtered_state[0,x] for x in range(kf_model_fit.filtered_state.shape[1])]
        Base=np.mean((NormalisedData-Growth))

        Success=False
        i=3
        while Success ==False and i>0:  #Iterate until fitting improves over underfit base case
            start_params = [np.var(NormalisedData)**(i), 0.1]
            kf_model_fit = kf_model.fit(start_params,maxiter = 20,method = 'bfgs', hessian= 'true')
            Growth=[kf_model_fit.filtered_state[0,x] for x in range(kf_model_fit.filtered_state.shape[1])]
            if np.mean((NormalisedData-Growth))/Base<.05:  #.08
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
        

class ChangePointDetector:
    def __init__(self, data,dates, SeasonalityPeriods=1):
        self.ts=TimeSeriesData(data,dates)
        self.TimeseriesModel = MyKalmanFilter(self.ts)
        self.TimeseriesModel.InitialiseFilter()
        self.SeasonalityPeriods=SeasonalityPeriods
    
    def ChangePointDetectorFunction(self):
    #This is the main function    
        gs_orig=self.TimeseriesModel.KalmanModelFit.filtered_state[-1,:]  #Get underlying linear trend
        gs=TimeSeriesData(gs_orig,self.ts.dates)
        OnePeriodRegressionModel=MyKalmanFilter(gs,SeasonalityPeriods=0) #Single period autoregression of linear trend
        OnePeriodRegressionModel.InitialiseFilter()
        
        #Get Mahalanobis distance between successive points in underlying trend
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
                    EV_conditionalm=exp(-exp(-(MDout[t-1])))  #Gumbel cdf
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
            if   EV>0.95:  
                EV=1
                ChangePoint=t
                PrLmOld=0
                EV_Old=1
        #         print('Change at ',t,MD_Prob[t-1])
                ChangePoints.append(ChangePoint)
        
            EVMT.append(EV)
            
        ExtendedDates=list(self.ts.dates)
        if self.TimeseriesModel.K_gain[-1,0]<0.65 and not sum([i > len(self.ts.data)-4 for i in ChangePoints]):
            Prediction=self.TimeseriesModel.KalmanModelFit.predict(0,len(self.ts.data)+3)
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
        else:
            Prediction=self.TimeseriesModel.KalmanModelFit.predict(0,len(self.ts.data))
            ExtendedDates=self.ts.dates
        
        Prediction=Prediction*np.linalg.norm(self.ts.data)
        
        Trend=[self.TimeseriesModel.KalmanModelFit.filtered_state[1,x] for x in range(self.TimeseriesModel.KalmanModelFit.filtered_state.shape[1])]
        Trend=[x*np.linalg.norm(self.ts.data) for x in Trend]
        
        TrendVariance=[self.TimeseriesModel.KalmanModelFit.filtered_state_cov[-1,-1,x] for x in range(self.TimeseriesModel.KalmanModelFit.filtered_state_cov.shape[2])]
        TrendVariance=[x*np.linalg.norm(self.ts.data) for x in TrendVariance]
        
        ChangePoints=[1 if x in ChangePoints else 0 for x in range(len(gs_orig))]
        
        Results= ModuleResults(Trend=Trend,TrendVariance=TrendVariance,ChangePoints=ChangePoints,Prediction=Prediction,\
                              ExtendedDates=ExtendedDates)
        
        return Results
    
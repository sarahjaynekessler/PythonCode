import pandas as pd
import numpy as np

def rollingmedainXY(df,X,bins):
    df2 = df.select_dtypes(include=np.number)
    data_cut = pd.cut(df2[X],bins)
    grp = df2.groupby(by = data_cut)
    ret = grp.aggregate(np.nanmedian)
    return(ret)

def rollingstdXY(df,X,bins):
    df2 = df.select_dtypes(include=np.number)
    data_cut = pd.cut(df2[X],bins)           #we cut the data following the bins
    grp = df2.groupby(by = data_cut)        #we group the data by the cut
    ret = grp.aggregate(np.nanstd)         #we produce an aggregate representation (std) of each bin
    return(ret)

def rollingMADZscoreXY(df,X,bins):
    df2 = df.select_dtypes(include=np.number)
    data_cut = pd.cut(df2[X],bins)           #we cut the data following the bins
    grp = df2.groupby(by = data_cut)        #we group the data by the cut
    ret = grp.aggregate(MADzscore)         #we produce an aggregate MAD per bin
    return(ret)

def rollingMADXY(df,X,bins):
    df2 = df.select_dtypes(include=np.number)
    data_cut = pd.cut(df2[X],bins)           #we cut the data following the bins
    grp = df2.groupby(by = data_cut)        #we group the data by the cut
    ret = grp.aggregate(myMAD)         #we produce an aggregate MAD per bin
    return(ret)

def myMAD(x): #my function for MAD that ignored nans
    med = np.nanmedian(x) #calculate median of bin
    x   = abs(x-med) #fins absolute difference
    MAD = np.nanmedian(x) #take median of the differences per bin
    return(MAD)

def MADzscore(x):
    med = np.nanmedian(x) #calculate median of bin
    x   = abs(x-med) #fins absolute difference
    MAD = np.nanmedian(x) #take median of the differences per bin
    zscore = (0.6745*x)/MAD
    return(zscore)

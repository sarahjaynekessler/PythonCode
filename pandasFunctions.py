import pandas as pd

def rollingmedainXY(df,X,Y):
	data_cut = pd.cut(df.X,bins)           #we cut the data following the bins
	grp = df.groupby(by = data_cut)        #we group the data by the cut
	ret = grp.aggregate(np.median)         #we produce an aggregate representation (median) of each bin
	return(ret)

def rollingstdXY(df,X,Y):
	data_cut = pd.cut(df.X,bins)           #we cut the data following the bins
	grp = df.groupby(by = data_cut)        #we group the data by the cut
	ret = grp.aggregate(np.std)         #we produce an aggregate representation (std) of each bin
	return(ret)

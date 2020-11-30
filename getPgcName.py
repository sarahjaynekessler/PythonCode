import pandas as pd
import numpy as np

def makePGCName(names):
	pklfile = '/Users/kessler.363/pythonPrograms/superset_alias.pkl'
	how = 'left'
	df_dict = pd.read_pickle(pklfile)
	df_names = pd.DataFrame(np.atleast_1d(names),columns=['alias'])
	df_pgc = pd.DataFrame(pd.merge(df_names, df_dict,on='alias', how=how)['PGC'])
	df_pgc['PGC'] = 'PGC' + df_pgc['PGC'].astype(str)
	return(df_pgc['PGC'].astype('str'))

def formatName(name):
	if name[:3] == 'ESO':
		if name[3:4] == ' ':
			splitname = name.split()
			suf = splitname[1].split('-')
			if len(suf[0]) ==1:
				suf[0] = '00'+suf[0]
			elif len(suf[0]) == 2:
				suf[0] = '0'+suf[0]
			if len(suf[1]) == 1:
				suf[1] = '00'+suf[1]
			elif len(suf[1]) == 2:
				suf[1] = '0'+suf[1]
			esoname = splitname[0]+suf[0]+'-'+suf[1]
		else:
			suf = name[3:].split('-')
			if len(suf[0]) ==1:
				suf[0] = '00'+suf[0]
			elif len(suf[0]) == 2:
				suf[0] = '0'+suf[0]
			if len(suf[1]) == 1:
				suf[1] = '00'+suf[1]
			elif len(suf[1]) == 2:
				suf[1] = '0'+suf[1]
			esoname = 'ESO'+suf[0]+'-'+suf[1]
	return(esoname)

def getButaGalNames(table,pgc=None,returntable=None):
	#takes in astropy table
	#if you want to convert to pgc use the variable
	#if you want a pandas df back with pgc names call return table
	if returntable is not None:

		for i in np.arange(len(table['Name'])):
			if 'ESO' in table[i]['Name']:
				try:		
					table[i]['Name'] = formatName(table[i]['Name'])
				except:
					print(table[i]['Name'])
			else:
				table[i]['Name']= table[i]['Name'].replace(' ','')
		if pgc is None:		
			return(table)	
		else:

			table['pgcnames'] = makePGCName(table['Name'].tolist())
			tabledf = table.to_pandas()
			tabledf['pgcnames'] = tabledf['pgcnames'].astype(str).str[:-2]

			return(tabledf)
	else:
		butanames = []
		for x in table['Name']:
			if 'ESO' in x:
				butanames.append(formatName(x))
			else:
				butanames.append(x.replace(' ',''))
		if pgc is None:
			return(butanames)
		else:
			return(pgcnames.tolist())

		

def getNameFromFiles(dirlist):
	namelist = []
	for i in dirlist:
		name = i.split('/')[-1].split('.')[0]
		namelist.append(name)
	return(namelist)

def returnNotMatches(a, b):
    return([[x for x in a if x not in b], [x for x in b if x not in a]])	


from glob import glob
from astropy.io import fits
from UsefulAstroData import getPixelSize
from astropy import wcs
from UsefulAstroData import radec_maps,deproject
from reproject import reproject_interp
import numpy as np 
import astropy.units as u
import astropy.constants as const
import pandas as pd

def makeAngularR25Aperture(df,i,radii_list):
    angradii = []
    for rad in radii_list:
        ap = rad*df.iloc[i].R25_DEG
        angradii.append(ap)
    andradii.append(df.iloc[i].RA2/3600.)
    return(angradii)

def makeAngularREFFAperture(df,i,radii_list):
    angradii = []
    for rad in radii_list:
        ap = rad*df.iloc[i].REFF
        angradii.append(ap)
    andradii.append(df.iloc[i].RA2/3600.)
    return(angradii)


def makePhysicalAperture(df,i,radii_list):
    kpcradii = []
    for rad in radii_list:
        ap = ((rad)/(df.iloc[i].DIST_MPC*1e3))*u.radian.to(u.deg)
        kpcradii.append(ap)
    kpcradii.append(df.iloc[i].RA2/3600.)
    return(kpcradii)

def makeBarAperture(df,i,radii_list):
    barradii=[] 
    for rad in radii_list:
        ap = ((rad*df.iloc[i].RA2)/3600.)
        barradii.append(ap)
    barradii.append(df.iloc[i].RA2/3600.)
    return(barradii)

def runApertureLoop(path,df,radii_list,bands,suffix,typeofAp,makedf=True,ind=None):

    pgcnames = ['PGC'+ str(i) for i in df.PGC.astype('int')]
    if ind==None:
        ind = len(pgcnames)
    else:
        ind=ind
    ra2s = [float(i) for i in df.RA2_kpc]
    bands = [i.lower() for i in bands]
    radii_str = [str(i).replace('.','p') for i in radii_list]
    radii_str.append('RA2')
    listnames = []
    for i in bands:
        for j in radii_str:
            listnames.append(i.upper()+'_'+j+suffix)

    listnames = [i.replace('1'+suffix,suffix) for i in listnames]

    dictionary = {key:[] for key in listnames}
    if typeofAp=='angular':
        radiinames = ['_'+j+suffix for j in radii_str]
        radiinames = [i.replace('1'+suffix,suffix) for i in listnames]
        radiisuffix = ['_'+i+suffix for i in radii_str]
        radiisuffix = [i.replace('_1','_') for i in radiisuffix]
    else:
        radiisuffix = ['_'+i+suffix for i in radii_str] 

    wavesum = {'fuv':1540*1e-4,'nuv':2310*1e-4,'w1':3.4,'w2':4.6,'w3':12}

    for i in np.arange(len(pgcnames[:ind])):
        for band in bands:
            file = glob(path+pgcnames[i]+'_*'+band+'*_*fits')
            stars = glob(path+pgcnames[i]+'_*'+band+'*_*.fits')
            if len(file) == 0:
                for rd in radiisuffix:
                    dictionary[band.upper()+rd].append(np.nan)

            else:
                if len(stars) == 0:
                    centerApertures(file[0],0,df,i,band,wavesum[band],radiisuffix,radii_list,dictionary,typeofAp)
                else:
                    centerApertures(file[0],stars[0],df,i,band,wavesum[band],radiisuffix,radii_list,dictionary,typeofAp)

            if (len(pgcnames)-i)%50 == 0:
                print(len(pgcnames)-i,'Files to go')
                
    dictionary['PGC'] = pgcnames[:ind]
    dictionary['RA2_kpc']=ra2s[:ind]
    if makedf==True:
        newdf = makeDataFrame(dictionary,radiisuffix,bands)
        return(newdf)
    else:

        return(dictionary,radiisuffix,bands)

def makeDataFrame(dictionary,radiisuffix,bands):
    newdf = pd.DataFrame(dictionary)

    cuv = 10**(-43.42)
    cw3 = 10**(-42.79)
    cw4 = 10**(-42.73)
    for r in radiisuffix:
        newdf['SFRFUVW3'+r] = cuv*newdf['FUV'+r]+ cw3*newdf['W3'+r]
        if 'w4' in bands:
            newdf['SFRFUVW4'+r] = cuv*newdf['FUV'+r]+ cw4*newdf['W4'+r]
        newdf['M/Msun'+r] =  -0.04+1.12*np.log10(newdf['W1'+r]/3.839e33) #from Wen et al '13,  https://doi.org/10.1093/mnras/stt939
        newdf['mag_W1'+r] = (-2.5*np.log10((newdf['W1'+r]*1e-7)/3.0128e28)) 
        newdf['mag_W2'+r] = (-2.5*np.log10((newdf['W2'+r]*1e-7)/3.0128e28))    
        newdf['SFR/M*'+r] = newdf['SFRFUVW3'+r] / newdf['M/Msun'+r]

    newdf.PGC = newdf.PGC.apply(lambda x: x[3:])
    return(newdf)

def centerApertures(f,stars,df,i,band,wavlength,radiisuffix,radii_list,dictionary,typeofAp):
    hdulist = fits.open(f)[0]
    data = hdulist.data
    w = wcs.WCS(hdulist.header)
    ster = getPixelSize(w)
    data[np.isnan(data)] = 0
    data=data*ster*1e6*1e-23*(const.c.to(u.micron/u.s)/(wavlength*u.micron)).value #convert from sr to pixel and then to erg/s/cm^2/Hz
    if stars != 0:
        starmask = fits.open(stars)[0].data
        data[stars==1] = np.nan
    else:
        pass
    
    racen,deccen = df.iloc[i].RA_DEG,df.iloc[i].DEC_DEG
    pa,incl = df.iloc[i].POSANG_DEG,df.iloc[i].INCL_DEG
    
    ra,dec = radec_maps(hdulist.header)
    rgrid,tgrid = deproject(ra,dec,racen,deccen,pa,incl)
    
    medback = np.nanmedian(data[np.logical_and(rgrid>1.5*df.iloc[i].R25_DEG,
                                rgrid<2*df.iloc[i].R25_DEG)])
    data-=medback

    if typeofAp=='angularR25':
        radii = makeAngularR25Aperture(df,i,radii_list) 
    elif typeofAp=='angularREFF':
        radii = makeAngularREFFAperture(df,i,radii_list) 
    elif typeofAp == 'physical':
        radii = makePhysicalAperture(df,i,radii_list)
    elif typeofAp == 'bar':
        radii = makeBarAperture(df,i,radii_list)

    
    for r in np.arange(len(radii)):

        mask = rgrid<=radii[r]
        maskeddata = data[mask]

        if len(maskeddata)==0:
            dictionary[band.upper()+radiisuffix[r]].append(np.nan)
            
        elif (len(maskeddata[np.isnan(maskeddata)])/len(maskeddata))>0.15:
            print(f,i,len(maskeddata[np.isnan(maskeddata)])/len(maskeddata))
            dictionary[band.upper()+radiisuffix[r]].append(np.nan)

        else:
            apsum = np.nansum(maskeddata)*4*np.pi*(df.iloc[i].DIST_MPC*u.megaparsec.to(u.cm))**2
            dictionary[band.upper()+radiisuffix[r]].append(apsum)



import numpy as np
import aplpy
from astropy.io import fits
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
from glob import glob


def plot5x5(gals,stars,namelist,ax):             
        for i in np.arange(25):
                ax[i].clear()                        
                norm = simple_norm(fits.getdata(gals[i]),'log')
                ax[i].imshow(fits.getdata(gals[i]),aspect = 'auto',cmap = 'gray',norm = norm)        
                ax[i].contour(fits.getdata(stars[i]),colors = 'red',linewidths = .8)
                ax[i].set_title(str(namelist[i]))
                plt.setp(ax, xticks=[], yticks=[])
                plt.tight_layout()
        plt.draw()
        plt.pause(.01)

def getfilenames(namelist,band):
    gals = []
    stars = []
    for i in np.arange(len(namelist)):
        try:
            galpath = glob('fitsfiles/'+str(namelist[i])+'_*'+str(band)+'*_gauss15.fits')[0]
            starpath = glob('fitsfiles/'+str(namelist[i])+'_*'+str(band)+'*stars.fits')[0]
            gals.append(galpath)
            stars.append(starpath)
        except:
            pass

    return(gals,stars)

def runloop(galsfull,starsfull,start,namelist):
        bounds = np.arange(start,len(galsfull),25)
        f,ax = plt.subplots(5,5,figsize = (15,15))
        ax = ax.ravel()
        plt.show(block=False)
        for b in np.arange(start,len(bounds),1):        
                if b+1<len(bounds):    
                        gals = galsfull[bounds[b]:bounds[b+1]]
                        stars = starsfull[bounds[b]:bounds[b+1]]
                        namescut = namelist[bounds[b]:bounds[b+1]]
                        plot5x5(gals,stars,namescut,ax)                                            
                        
                        print(bounds[b+1],(b+1),len(bounds) - (b+1))
        plt.close()

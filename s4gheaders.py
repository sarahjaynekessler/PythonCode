from astropy.io import fits
from astropy.table import Table
import numpy as np
from reproject import reproject_interp
import astropy.units as u
import sliceCube
import aplpy
import gal_data
from getPgcName import makePGCName
from glob import glob
import os
import warnings
import logging

def reproject(f,makejpgfile=None):
        #takes file name (str), desired beam size, original beam size as arguments. Has the option to desample the image and scale the arcsecond resolution per beam to multilple pixels per beam. 

        #convolve to specified resolution in ". Applies convolved nan mask
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        failednames = []
        hdulist = fits.open(f)[0]
        if len(hdulist.data.shape) > 2:
                hdulist = sliceCube.sliceCube(f)
        try:
            galname = f.split('/')[-1].split('.')[0]
        except:
            failednames.append(f)
        print(galname)
        pgcname = makePGCName(galname)[0]
        hdr = hdulist.header.copy()

        galbase = gal_data.gal_data(names = galname)

        #create header
        res = np.float64(0.75*u.arcsec.to(u.deg)).round(decimals=4)
        galsize = galbase['R25_DEG'][0]*1.5
        length = (galsize/res).round(decimals=3)
        naxis = int(length*2)
        ra = galbase['RA_DEG'][0].round(decimals=6)
        dec = galbase['DEC_DEG'][0].round(decimals=6)
        
        hdr['CRVAL1'] = ra
        hdr['CRVAL2'] = dec
        hdr['NAXIS1'] = naxis
        hdr['NAXIS2'] = naxis
        hdr['CRPIX1'] = length
        hdr['CRPIX2'] = length

        #apply mask
        try:
            maskfile = glob('/Users/kessler.363/Desktop/S4G/S4Gmask2/*'+galname+'*.fits')[0]
            maskdata= fits.open(maskfile)[0].data
            hdulist.data[maskdata!=0] = np.nan
            hdulist.data[np.isnan(hdulist.data)] = np.nanmedian(hdulist.data)
            fits.writeto('temp.fits',hdulist.data,header = hdulist.header,overwrite=True)
            

            #reproject

            rep,foot = reproject_interp('temp.fits',hdr)
            fits.writeto('temp.fits',rep,header=hdr,overwrite=True)
            if makejpgfile != None:
                    makejpg(pgcname)
        except:
            failednames.append(f)
        return(failednames)
def makejpg(pgcname):
        gc = aplpy.FITSFigure('temp.fits')
        gc.show_grayscale()
        gc.set_nan_color('black')
        gc.axis_labels.hide()
        gc.ticks.hide()
        gc.tick_labels.hide()
        gc.save('/Users/kessler.363/Desktop/S4G/jpg2/'+pgcname+'.jpg')
        gc.close()


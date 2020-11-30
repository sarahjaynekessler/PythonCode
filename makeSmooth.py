from astropy.io import fits
from astropy import wcs
import numpy as np
from astropy.convolution import convolve,Gaussian2DKernel
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import sliceCube

def makeSmoothResKnown(f,beam,origbeam,desample=None,scaleArcSecRes = False,
    applynanmask=False):
        #takes file name (str), desired beam size, original beam size as arguments. Has the option to desample the image and scale the arcsecond resolution per beam to multilple pixels per beam. 

        #convolve to specified resolution in ". Applies convolved nan mask


        hdulist = fits.open(f)[0]
        if len(hdulist.data.shape) > 2:
            hdulist = sliceCube.sliceCube(f)
        nanmask = np.isnan(hdulist.data)

        w = wcs.WCS(hdulist.header)

        pixcrd = np.array([[0,0],[0,1]])
        world = w.wcs_pix2world(pixcrd,1)
        print('The coordiantes of the first two pixels are '+str(world[0])+' and '+str(world[1]))

        c1 = SkyCoord(ra = world[0][0],dec = world[0][1],frame ='fk5',unit = 'deg')
        c2 = SkyCoord(ra = world[1][0],dec = world[1][1],frame ='fk5',unit = 'deg')


        sep = c1.separation(c2)


        
        if scaleArcSecRes !=False:
            arcsecRes = np.round(sep.to(u.arcsec).value,decimals=1)*scaleArcSecRes
        else:
            arcsecRes = np.round(sep.to(u.arcsec).value,decimals=1)

        print('One pixel is currently '+str(arcsecRes))
        
        print('Resolution is currently '+str(origbeam)+' arcseconds')


        fwhm = np.sqrt(beam**2-origbeam**2)*(1./arcsecRes)

        print('To get to '+ str(beam)+' arcsec resolution we need to convolve to '+str(fwhm)+' pixels.')

        image = hdulist.data
        header = hdulist.header
        stddev =  fwhm/(np.sqrt(8*np.log(2)))
        krnl = Gaussian2DKernel(stddev)
        smoothed = convolve(image,krnl)
        if applynanmask!=False:
            smoothed[nanmask] = np.nan
        else:
            pass
        header['COMMENT'] = 'SK Convolved to '+ str(beam)+ ' arcsec'
        hdulist.header['BMIN'] = (beam*u.arcsec).to(u.degree).value
        hdulist.header['BMAJ'] = (beam*u.arcsec).to(u.degree).value

        if desample == True:
            fits.writeto('temp.fits',smoothed,header=hdulist.header,overwrite=True)

            pixscale = beam/3.
            print(arcsecRes/pixscale)
            print(int(hdulist.header['NAXIS1'] *(arcsecRes/pixscale)))
            hdulist.header['NAXIS1'] = int(hdulist.header['NAXIS1'] *(arcsecRes/pixscale))
            hdulist.header['NAXIS2'] = int(hdulist.header['NAXIS2']*(arcsecRes/pixscale))
            hdulist.header['CRPIX1'] *= arcsecRes/pixscale
            hdulist.header['CRPIX2'] *= arcsecRes/pixscale
            delta = float(pixscale*u.arcsec.to(u.degree))
            hdulist.header['CD1_1'] = -delta
            hdulist.header['CD2_2'] = delta
            hdulist.header['BMIN'] = (beam*u.arcsec).to(u.degree).value
            hdulist.header['BMAJ'] = (beam*u.arcsec).to(u.degree).value
            
            
        
            temp = fits.open('temp.fits')[0]
            smoothedrep,foot = reproject_interp(temp,hdulist.header)
            
            os.system('rm temp.fits')
            return(smoothedrep,hdulist.header)

        else:
            return(smoothed,hdulist.header)

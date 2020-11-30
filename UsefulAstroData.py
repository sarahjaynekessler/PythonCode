from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp
from matplotlib import rc
from astropy.table import Table,join,Column,vstack
from photutils import CircularAperture,CircularAnnulus
import aplpy
from photutils import aperture_photometry,SkyCircularAperture,SkyCircularAnnulus
from astropy.visualization import hist
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pyneb as pn
from dust_extinction.parameter_averages import F99,CCM89
from astropy import constants as const
from reproject import reproject_interp
from astropy.stats import sigma_clipped_stats
import os
import random
from glob import glob
import pandas as pd
from scipy.optimize import curve_fit
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from matplotlib.pyplot import cm
import numpy.ma as ma  

def radec_maps(hdr, shape=None):
    """
    Build RA and Dec arrays to go with the provided header.
    """

    w = wcs.WCS(hdr, naxis=[1,2])

    if shape is None:
        size_y = hdr['NAXIS2']
        size_x = hdr['NAXIS1']
    else:
        size_y = shape[0]
        size_x = shape[1]
    
    ra = np.zeros((size_y,size_x))
    dec = np.zeros_like(ra)

    for ii in range(size_y):
        for jj in range(size_x):
            this_ra, this_dec = w.all_pix2world(jj, ii, 0.)
            ra[ii,jj] = this_ra
            dec[ii,jj] = this_dec

    return ra, dec

def deproject(
    ra, dec, ra_ctr, dec_ctr, pa, incl,
    ):
    """
    Convert RA, Dec into cylindrical galactocentric coordinates r and
    theta.
    """

    # Calculate offset in RA, Dec
    x_off = (ra - ra_ctr)*np.cos(np.deg2rad(dec_ctr))
    y_off = (dec - dec_ctr)
    
    # Rotate so that x is along the 
    rot_ang = -1.0*(np.deg2rad(pa) - np.pi/2.0)
    deproj_x = x_off * np.cos(rot_ang) + y_off * np.sin(rot_ang)
    deproj_y = y_off * np.cos(rot_ang) - x_off * np.sin(rot_ang)
    
    deproj_y = deproj_y / np.cos(np.deg2rad(incl))
    
    rgrid = np.sqrt(deproj_x**2 + deproj_y**2)
    tgrid = np.arctan2(deproj_y, deproj_x)

    return rgrid, tgrid

def getridofstars11arcsec(filepath):

        f = fits.open(filepath)[0]
        data = f.data
        hdr = f.header
        mask = data.copy()
        mask[~np.isnan(mask)] = 1.
        regions = np.loadtxt('reproj/convolved/desampled/backsub/ds9_11arc_m51_stars_pixcoords.reg',delimiter=',')
        for i in np.arange(len(regions)):
                x = int(regions[i][0])
                y = int(regions[i][1])
                r = int(regions[i][2])
        
                data[y-r:y+r,x-r:x+r] = np.nan

        data = data*mask
        fits.writeto(filepath[:-5] + '_starsnanned.fits' ,data,header=hdr,overwrite=True)


def desample(f,orgres,newres,fname):
        #desamplesize is desired res / actual res (example 3" / 0.1"))
        hdulist = fits.open(f)[0]
        w = wcs.WCS(hdulist.header)

        header = fits.getheader('reproj/convolved/11.0_arcsec_ngc5194_0298_med_mosaic_center030.fits')
        desamplesize = newres/orgres
        
        data = hdulist.data
        print(data.shape)
        #header = hdulist.header
        
        indY = np.arange(0,data.shape[0],desamplesize,dtype='int')
        indX = np.arange(0,data.shape[1],desamplesize,dtype='int')
        
        reduceY = np.take(data,indY,axis=0)
        print(reduceY.shape)
        reducefinal = np.take(reduceY,indX,axis=1)
        print(reducefinal.shape)

        crpix1 = len(indY)/2. -1
        crpix2 =  len(indX)/2. +5
        print(int(len(indY)/2.))
        pixcrd = np.array([[0,0],[indY[int(len(indY)/2.)],indX[int(len(indX)/2.)]]])
        world = w.wcs_pix2world(pixcrd,1)[1]
        crval1 = world[0]
        crval2 = world[1]
        cdelt1 = np.float(-3.*u.arcsec.to(u.degree))
        cdelt2 = np.float(3.*u.arcsec.to(u.degree))     

        header['CRPIX1'] = crpix1
        header['CRPIX2'] = crpix2
        header['CDELT1'] = cdelt1
        header['CDELT2'] = cdelt2
        header['CRVAL1'] = crval1
        header['CRVAL2'] = crval2

        fits.writeto('reproj/convolved/desampled/' +str(fname[17:-10])+'_desampled.fits',reducefinal,header=header,overwrite=True)

def findPixelSize(file):
        hdulist = fits.open(file)
        w = wcs.WCS(hdulist[0].header)
        print(w.wcs.name)

        pixcrd = np.array([[0,0],[0,1]])
        world = w.wcs_pix2world(pixcrd,1)
        print(world[0],world[1])

        c1 = SkyCoord(ra = world[0][0],dec = world[0][1],frame ='fk5',unit = 'deg')
        c2 = SkyCoord(ra = world[1][0],dec = world[1][1],frame ='fk5',unit = 'deg')


        sep = c1.separation(c2)
        print(sep)
        print(sep.to(u.arcsec))
        rad = sep.to(u.radian)


        strad = rad**2.
        print('The distance of one pixel is ')
        print(strad)



        return(np.float(np.array(strad)))


def Mjyster_to_ergs(data,lam_um):
        data = (data*(1.e-17)*(3.e14))/ (lam_um)
        return(data)

def rollingmean(filepath,threshhold,name):
        f = fits.open(filepath)[0]

        

        hdr = fits.getheader(filepath)
        data= f.data
        ideal = 0.0

        ha_reproj = fits.getdata('reproj/convolved/desampled/8.0_arcsec_ngc5194_vngs_pacs70_ergs_reproj.fits')

        ha_reproj[ha_reproj > 2.e-5] = 20.
        ha_reproj[ha_reproj == 0.] = 1.
        ha_reproj[np.isnan(ha_reproj)] = 1.
        ha_reproj[ha_reproj == 20.] = np.nan
        ha_reproj[~np.isnan(ha_reproj)] = 1.
        nonha = data*ha_reproj
        test = data*nonha
        
        #if masked == 'pab':
        #       ha_reproj[35:45,50:60] = ha_reproj[50:60,80:90] = ha_reproj[18:28,112:]= np.nan
                #ha_reproj[70:80,95:110] = ha_reproj[80:90,95:110] = np.nan
        nonha = data*ha_reproj
        test = data*nonha
        fits.writeto('test.fits',test,overwrite=True)   
        normalback = np.nanmedian(nonha)
        print(normalback)
        testimage = data.copy()
        halfbox = threshhold/2.
        print(ideal)
        print('starting for loop')

        for y in np.arange(0,nonha.shape[0]):
                for x in np.arange(0,nonha.shape[1]):
                        if (y+halfbox) > (nonha.shape[0]):
                                ymin = (nonha.shape[0]) - halfbox*2
                                ymax = (nonha.shape[0]) 

                        if (x+halfbox) > (nonha.shape[1]):
                                xmin = (nonha.shape[1]) - halfbox*2
                                xmax = (nonha.shape[1]) 
                        else:
                                xmin = x-halfbox
                                xmax = x+halfbox
                                ymin = y-halfbox
                                ymax = y+halfbox
                        med = np.nanmean(nonha[ymin:ymax,xmin:xmax])

                        diff = med - ideal

                        testimage[y,x] = data[y,x] - diff
                        if ~np.isnan(med):
                                oldmed = med
        if name == 'hashift':
                testimage[:,145:] = np.nan
        fits.writeto('reproj/convolved/desampled/backsub/' + str(name) + 'mean_backsub.fits',testimage,hdr,overwrite=True)

def makeSquareGridApertures(f,aperturesize,threshhold):
        #aperturesize is in pixels
        #aperturesize is radius of circle
        squaresize = aperturesize*2 - 20
        nonnanpix = []


        f.data[f.data < threshhold] = np.nan
        xint = 0
        yint = 0
        for x in np.arange(0,f.data.shape[1]+ squaresize/2,squaresize):
                for y in np.arange(0,f.data.shape[0]+ squaresize/2,squaresize):
                        lennans = np.float(np.count_nonzero(np.isnan(f.data[yint:y,xint:x])))
                        lennonnans = np.float(len(np.isnan(f.data[yint:y,xint:x].flat)))
                        
                        if np.isnan(f.data[yint:y,xint:x]).all():
                                pass
                        elif (1.- (lennans/lennonnans)) < 0.75: 
                                        pass    
                        else:
                                nonnanpix.append([x - squaresize/2,y- squaresize/2])
                        #print(yint,y,xint,x)
                        #print(ahamask.data[yint:y,xint:x])
                        if y == np.arange(0,f.data.shape[0]+ squaresize/2,squaresize)[-1]:
                                yint = 0
                        else: 
                                yint = y
                if x == np.arange(0,f.data.shape[1]+ squaresize/2,squaresize)[-1]:
                        xint = 0
                else: 
                        xint = x

        apertures = CircularAperture(nonnanpix,r = aperturesize)

        plt.imshow(f.data)
        apertures.plot()
        plt.show()
        w = wcs.WCS(f.header)
        world = w.wcs_pix2world(nonnanpix,0)
        ras =  Column(data = [world[j][0] for j in np.arange(len(world))],name = 'ra')
        decs = Column(data = [world[k][1] for k in np.arange(len(world))],name = 'dec')
        t = Table([ras,decs])
        t.write('NGC5194_Aperture_Locs.csv',format = 'ascii',delimiter = ',',overwrite=True)
        
        return(nonnanpix,world)


def makeCube2D(data):

        w = wcs.WCS(data.header)
        print(w)
        data.header.remove('PC1_1')
        data.header.remove('PC1_2')
        data.header.remove('PC1_3')
        data.header.remove('PC1_4')
        data.header.remove('PC2_1')
        data.header.remove('PC2_2')
        data.header.remove('PC2_3')
        data.header.remove('PC2_4')
        data.header.remove('PC3_1')
        data.header.remove('PC3_2')
        data.header.remove('PC3_3')
        data.header.remove('PC3_4')
        data.header.remove('PC4_1')
        data.header.remove('PC4_2')
        data.header.remove('PC4_3')
        data.header.remove('PC4_4')

        data.header.remove('CTYPE3')
        data.header.remove('CRVAL3')
        data.header.remove('CDELT3')
        data.header.remove('CRPIX3')
        data.header.remove('CUNIT3')

        data.header.remove('CTYPE4')
        data.header.remove('CRVAL4')
        data.header.remove('CDELT4')
        data.header.remove('CRPIX4')
        data.header.remove('CUNIT4')


        w = wcs.WCS(data.header)
        print(w)

        return(data)

def getPixelSize(w):

        pixcrd = np.array([[0,0],[0,1]])
        world = w.wcs_pix2world(pixcrd,1)

        c1 = SkyCoord(ra = world[0][0],dec = world[0][1],frame ='fk5',unit = 'deg')
        c2 = SkyCoord(ra = world[1][0],dec = world[1][1],frame ='fk5',unit = 'deg')


        sep = c1.separation(c2)

        rad = sep.to(u.radian)


        strad = rad**2.



        return(strad.value)

def converthmsToDeg(t2):
    RAS = []
    DECS = []
    for i in np.arange(len(t2)):
        ra = str(str(t2[i]['RAh'])+'h'+str(t2[i]['RAm'])+'m'+str(t2[i]['RAs'])+'s') 
        dec = str('+'+str(t2[i]['DEd'])+'d'+str(t2[i]['DEm'])+'m'+str(t2[i]['DEs'])+'s')
        RAS.append(ra)
        DECS.append(dec)
    
    return(RAS,DECS)

def blancbalmerapertures():
    import warnings
    warnings.simplefilter("ignore")
    t =  Table.read('/Users/kessler.363/Desktop/PaBeta/Blanc09.txt',format = 'ascii') 
    ha = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/working_ngc5194_ha_arc4.fits')[0]
    freefree = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/freefree_3arc.fits')[0]
    freefreeimg = freefree.data
 
    f = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/convolvedmosaic/arc4.fits')[0]
    f.data/=2.
    w = wcs.WCS(f.header)
    ster = getPixelSize(w) 
    f.data*=ster
    nanmask = f.data.copy()
    nanmask[~np.isnan(nanmask)] = 1.

    rgrid = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/rgrid.fits')[0]
   
    haimg = ha.data
    #convert to per pixel instead of per ster and apply nanmask
    haimg *=nanmask*ster 

    rgridimg = rgrid.data    
    freefreeimg*=nanmask*np.pi*(((2./2.)**2.)/(1.1331*(2.94*(np.pi/180.)/3600.)*(2.67*(np.pi/180.)/3600.)))*1e-23*(3e18/(9.085e7)**2)

    haerr = 2.772485559909721e+35
    paberr = 2.3849392462793174e+35
    freefreeerr =  3.198812487256924e+23*np.pi*(((2./2.)**2.)/(1.1331*(2.94*(np.pi/180.)/3600.)*(2.67*(np.pi/180.)/3600.)))*1e-23*(3e18/(9.085e7)**2)  

    t2 = Table.read('/Users/kessler.363/Desktop/PaBeta/M51/analysis/Kennicutt07M51PaAlpha.txt',format = 'ascii')
    ras,decs = converthmsToDeg(t2)

    kenaps = SkyCircularAperture(SkyCoord(ras,decs,frame = 'fk5'),r = (13*u.arcsec)/2.)
    kenans = SkyCircularAnnulus(SkyCoord(ras,decs,frame = 'fk5'),r_in = 3.19*u.arcsec,r_out = 47.*u.arcsec)
    kenaps = kenaps.to_pixel(w)
    kenans = kenans.to_pixel(w)
    ken_masks = kenans.to_mask(method='center')


    RAS,DECs = converthmsToDeg(t)
    world = [[RAS[j],DECs[j]] for j in np.arange(len(RAS))]
    c = SkyCoord(world) 

    apertures = SkyCircularAperture(c,r = 4.3*u.arcsec/2.).to_pixel(w)
    annulus_apertures = SkyCircularAnnulus(c, r_in=5.2/2.*u.arcsec, r_out=49/2.*u.arcsec).to_pixel(w)
    annulus_masks = annulus_apertures.to_mask(method='center')

    #paB apertures
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(f.data)
        try:
            annulus_data_1d = annulus_data[mask.data>0]
        except:
            pass
        _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    pab_bkg_median = np.array(bkg_median)

    pab_phot_table = aperture_photometry(f.data , apertures)
    pab_phot_table['annulus_median'] = pab_bkg_median
    pab_phot_table['aper_bkg'] = pab_bkg_median*apertures.area
    pab_phot_table['aper_sum_bkgsub'] = pab_phot_table['aperture_sum'] - pab_phot_table['aper_bkg']
    pab_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
    pab_phot_table.rename_column('aper_sum_bkgsub','pab_aperture_sum')


    #paB apertures (kenn)
    bkg_median = []
    for mask in ken_masks:
        annulus_data = mask.multiply(f.data)
        try:
            annulus_data_1d = annulus_data[mask.data>0]
        except:
            pass
        _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    pab_bkg_median_ken = np.array(bkg_median)

    pab_phot_table_ken = aperture_photometry(f.data , kenaps)
    pab_phot_table_ken['annulus_median'] = pab_bkg_median_ken
    pab_phot_table_ken['aper_bkg'] = pab_bkg_median_ken*kenaps.area()
    pab_phot_table_ken['aper_sum_bkgsub'] = pab_phot_table_ken['aperture_sum'] - pab_phot_table_ken['aper_bkg']
    pab_phot_table_ken['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
    pab_phot_table_ken.rename_column('aper_sum_bkgsub','pab_aperture_sum')

    #rgrid
    rgridimg *= (np.pi/180.)*8580.
    rgrid_phot_table = aperture_photometry(rgridimg, apertures)
    rgrid_phot_table['aperture_sum']/= apertures.area
    rgrid_phot_table.rename_column('aperture_sum','r_kpc')   

    #rgrid
    rgrid_phot_table_ken = aperture_photometry(rgridimg,kenaps)
    rgrid_phot_table_ken['aperture_sum']/= kenaps.area()
    rgrid_phot_table_ken.rename_column('aperture_sum','r_kpc')   

    #ha apertures
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(haimg)
        try:
            annulus_data_1d = annulus_data[mask.data>0]
        except:
            pass
        _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    ha_bkg_median = np.array(bkg_median)

    ha_phot_table = aperture_photometry(haimg , apertures)
    ha_phot_table['annulus_median'] = ha_bkg_median
    ha_phot_table['aper_bkg'] = ha_bkg_median*apertures.area
    ha_phot_table['aper_sum_bkgsub'] = ha_phot_table['aperture_sum'] - ha_phot_table['aper_bkg']
    ha_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
    ha_phot_table.rename_column('aper_sum_bkgsub','ha_aperture_sum')

    #ha apertures (kenn)
    bkg_median = []
    for mask in ken_masks:
        annulus_data = mask.multiply(haimg)
        try:
            annulus_data_1d = annulus_data[mask.data>0]
        except:
            pass
        _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    ha_bkg_median_ken = np.array(bkg_median)

    ha_phot_table_ken = aperture_photometry(haimg , kenaps)
    ha_phot_table_ken['annulus_median'] = ha_bkg_median_ken
    ha_phot_table_ken['aper_bkg'] = ha_bkg_median_ken*kenaps.area()
    ha_phot_table_ken['aper_sum_bkgsub'] = ha_phot_table_ken['aperture_sum'] - ha_phot_table_ken['aper_bkg']
    ha_phot_table_ken['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
    ha_phot_table_ken.rename_column('aper_sum_bkgsub','ha_aperture_sum')

    #freefree apertures
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(freefreeimg)
        annulus_data_1d = annulus_data[mask.data>0]
        _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    freefree_bkg_median = np.array(bkg_median)

    freefree_phot_table = aperture_photometry(freefreeimg , apertures)
    freefree_phot_table['annulus_median'] = freefree_bkg_median
    freefree_phot_table['aper_bkg'] = freefree_bkg_median*apertures.area
    freefree_phot_table['aper_sum_bkgsub'] = freefree_phot_table['aperture_sum'] - freefree_phot_table['aper_bkg']
    freefree_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
    freefree_phot_table.rename_column('aper_sum_bkgsub','freefree_aperture_sum')




    Aha,extimg,hacorr = HaExtinctionMags(ha_phot_table['ha_aperture_sum'],pab_phot_table['pab_aperture_sum'],'PaB','Cardelli',3.1)
    Ahap,extimg,hacorr = HaExtinctionMags(ha_phot_table['ha_aperture_sum']-2*haerr,pab_phot_table['pab_aperture_sum']+paberr,'PaB','Cardelli',3.1)
    Aham,extimg,hacorr = HaExtinctionMags(ha_phot_table['ha_aperture_sum']+2*haerr,pab_phot_table['pab_aperture_sum']-paberr,'PaB','Cardelli',3.1)


    Ahabalmer,extimgbalmer,hacorrbalmer = HaExtinctionMags(t['Ha'],t['Hb'],'Hbeta','Cardelli',3.1)
    Ahabalmerp,_,_ =  HaExtinctionMags(t['Ha']+ t['e_Ha'],t['Hb']-t['e_Hb'],'Hbeta','Cardelli',3.1)
    Ahabalmerm,_,_ =  HaExtinctionMags(t['Ha']- t['e_Ha'],t['Hb']+t['e_Hb'],'Hbeta','Cardelli',3.1)

    Ahafreefree,_,_ = HaExtinctionMags(ha_phot_table['ha_aperture_sum'],freefree_phot_table['freefree_aperture_sum'],'freefree','Cardelli',3.1)
    Ahafreefreep,_,_ = HaExtinctionMags(ha_phot_table['ha_aperture_sum']-haerr,freefree_phot_table['freefree_aperture_sum'] + freefreeerr, 'freefree','Cardelli',3.1)
    Ahafreefreem,_,_ = HaExtinctionMags(ha_phot_table['ha_aperture_sum']+haerr,freefree_phot_table['freefree_aperture_sum'] - freefreeerr ,'freefree','Cardelli',3.1)

    Ahaken,_,_ = HaExtinctionMags(10**t2['LogLHa'],10**t2['LogLPa'],'Paalpha','Cardelli',3.1)
    mask = ma.getmask(Ahaken)
    Ahaken = ma.getdata(Ahaken)[~mask]
    rgrid_phot_table_ken = rgrid_phot_table_ken[~mask]

    mask = np.logical_and(rgrid_phot_table['r_kpc']<1.3,np.logical_and(Aha>0.,Aha<3.5))
    kenmask = rgrid_phot_table_ken['r_kpc']<1.3

    t = Table.read('NGC5194_full_aperture_2arc.txt',format = 'ascii')
    t1m = Table.read('m51Aha+H.txt',format = 'ascii')                                                          
    errs = t['E_AHa'][t1m['r']<1.3]
    ahaerrs = errs[:len(Aha[mask])]
    balmererrs = [Ahabalmer[mask]-Ahabalmerm[mask],Ahabalmerp[mask]-Ahabalmer[mask]]
    freefreeerrs = [Ahafreefree[mask]-Ahafreefreem[mask],Ahafreefreep[mask]-Ahafreefree[mask]]
    #ahaerrs = [Aha[mask]-Aham[mask],Ahap[mask]-Aha[mask]]
    kenerrs = 0.1*Ahaken[kenmask]
    
    
    return(Aha[mask],Ahabalmer[mask],Ahaken[kenmask],Ahafreefree[mask],rgrid_phot_table['r_kpc'][mask],rgrid_phot_table_ken['r_kpc'][kenmask],balmererrs,freefreeerrs,ahaerrs,kenerrs)


def freefreeapertures():
        import warnings
        warnings.simplefilter("ignore") 
        #apr is aperture diameter in pixels
        fname = '/Users/kessler.363/Desktop/PaBeta/M51/convolvedmosaic/arc3.fits'
        apr = (3./.3)
        ha = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/working_ngc5194_ha_arc3.fits')[0]
        freefree = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/freefree_3arc.fits')[0]
        rgrid = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/rgrid.fits')[0]

        phot_tables = []
        stds = []
        #cmap = plt.get_cmap("rainbow")
        t = Table.read('/Users/kessler.363/Desktop/PaBeta/M51/NGC5194_Aperture_Locs3.csv',format = 'ascii',delimiter = ',')
        #t = Table.read('NGC5194_Aperture_Locs2back.csv',format = 'ascii',delimiter = ',')


        f = fits.open(fname)[0]
        w = wcs.WCS(f.header)
        ster = getPixelSize(w) 
        f.data*=ster
        nanmask = f.data.copy()
        nanmask[~np.isnan(nanmask)] = 1.


        
        haimg = ha.data
        rgridimg = rgrid.data
        freefreeimg = freefree.data
        #convert to per pixel instead of per ster and apply nanmask
        haimg *=nanmask*ster 
        rgridimg *=nanmask
        #freefreeimg*=nanmask*np.pi*((1.5**2.)/(1.1331*(2.94*(np.pi/180.)/3600.)*(2.67*(np.pi/180.)/3600.)))*1e-23*ster
        freefreeimg*=nanmask*np.pi*(((1.5/2.)**2.)/(1.1331*(2.94*(np.pi/180.)/3600.)*(2.67*(np.pi/180.)/3600.)))*1e-23*(3e18/(9.085e7)**2)

        haerr = 6.772485559909721e+35
        paberr = 6.3849392462793174e+35
        #freefreeerr =  3.198812487256924e+23*np.pi*(((3./2.)**2.)/(1.1331*(2.94*(np.pi/180.)/3600.)*(2.67*(np.pi/180.)/3600.)))*1e-23
        freefreeerr =  3.198812487256924e+23*np.pi*(((3./2.)**2.)/(1.1331*(2.94*(np.pi/180.)/3600.)*(2.67*(np.pi/180.)/3600.)))*1e-23*(3e18/(9.085e7)**2)


        world = [ [t['ra'][j],t['dec'][j]] for j in np.arange(len(t['ra']))]
        pixcoords = w.wcs_world2pix(world,0)



        apertures = CircularAperture(pixcoords,r = apr/2.)
        annulus_apertures = CircularAnnulus(pixcoords, r_in=apr/2. + 4, r_out=apr/2.+150)
        annulus_masks = annulus_apertures.to_mask(method='center')
        #rgrid apertures for distance from cetner of gal (np.pi/180.)*dist*1e3
        rgridimg *= (np.pi/180.)*8580.
        rgrid_phot_table = aperture_photometry(rgridimg, apertures)
        rgrid_phot_table['aperture_sum']/= apertures.area
        rgrid_phot_table.rename_column('aperture_sum','r_kpc')

        #paB apertures
        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(f.data)
            annulus_data_1d = annulus_data[mask.data>0]
            _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        pab_bkg_median = np.array(bkg_median)

        pab_phot_table = aperture_photometry(f.data , apertures)
        pab_phot_table['annulus_median'] = pab_bkg_median
        pab_phot_table['aper_bkg'] = pab_bkg_median*apertures.area
        pab_phot_table['aper_sum_bkgsub'] = pab_phot_table['aperture_sum'] - pab_phot_table['aper_bkg']
        pab_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
        pab_phot_table.rename_column('aper_sum_bkgsub','pab_aperture_sum')
        ras =  Column(data = t['ra'],name = 'ra')
        decs = Column(data = t['dec'],name = 'dec')
        pab_phot_table.add_column(ras,index = 1)
        pab_phot_table.add_column(decs,index = 2)
                
        #ha apertures
        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(haimg)
            annulus_data_1d = annulus_data[mask.data>0]
            _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        ha_bkg_median = np.array(bkg_median)

        ha_phot_table = aperture_photometry(haimg , apertures)
        ha_phot_table['annulus_median'] = ha_bkg_median
        ha_phot_table['aper_bkg'] = ha_bkg_median*apertures.area
        ha_phot_table['aper_sum_bkgsub'] = ha_phot_table['aperture_sum'] - ha_phot_table['aper_bkg']
        ha_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
        ha_phot_table.rename_column('aper_sum_bkgsub','ha_aperture_sum')
                

        #freefree apertures
        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(freefreeimg)
            annulus_data_1d = annulus_data[mask.data>0]
            _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        freefree_bkg_median = np.array(bkg_median)

        freefree_phot_table = aperture_photometry(freefreeimg , apertures)
        freefree_phot_table['annulus_median'] = freefree_bkg_median
        freefree_phot_table['aper_bkg'] = freefree_bkg_median*apertures.area
        freefree_phot_table['aper_sum_bkgsub'] = freefree_phot_table['aperture_sum'] - freefree_phot_table['aper_bkg']
        freefree_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
        freefree_phot_table.rename_column('aper_sum_bkgsub','freefree_aperture_sum')

        mask = rgrid_phot_table['r_kpc'] <3.

        #freefree_phot_table['freefree_aperture_sum'][mask]*=0.546
        freefree_phot_table['freefree_aperture_sum'][~mask]*=0.7

        print(len(pab_phot_table),len(ha_phot_table),len(rgrid_phot_table))

        Aha,ext,hacorr = HaExtinctionMags(ha_phot_table['ha_aperture_sum'],pab_phot_table['pab_aperture_sum'],'PaB','Cardelli',3.1)
        Ahap,haextp,hacorrp = HaExtinctionMags(ha_phot_table['ha_aperture_sum']-haerr,pab_phot_table['pab_aperture_sum']+paberr,'PaB','Cardelli',3.1)
        Aham,haextm,hacorrm = HaExtinctionMags(ha_phot_table['ha_aperture_sum']+haerr,pab_phot_table['pab_aperture_sum']-paberr,'PaB','Cardelli',3.1)

        Ahafreefree,_,_ = HaExtinctionMags(ha_phot_table['ha_aperture_sum'],freefree_phot_table['freefree_aperture_sum'],'freefree','Cardelli',3.1)
        Ahafreefreep,_,_ = HaExtinctionMags(ha_phot_table['ha_aperture_sum']-haerr,freefree_phot_table['freefree_aperture_sum'] + freefreeerr, 'freefree','Cardelli',3.1)
        Ahafreefreem,_,_ = HaExtinctionMags(ha_phot_table['ha_aperture_sum']+haerr,freefree_phot_table['freefree_aperture_sum'] - freefreeerr ,'freefree','Cardelli',3.1)


        sfrfreefree = freefreeToSFR(freefree_phot_table['freefree_aperture_sum'])
        sfrfreefree/=5.4e-42

        sfrha = hacorr
        #t2 = Table([sfrfreefree,sfrha,Aha],names = ('SFRff','SFRha','Aha'))
        t2 = Table([ha_phot_table['ha_aperture_sum'],pab_phot_table['pab_aperture_sum'],freefree_phot_table['freefree_aperture_sum'],Aha,sfrfreefree,sfrha],names = ('LHa','LPab','FreeFree', 'AHa', 'FreeFreeSFR','HaSFR'))

        return(pab_phot_table,ha_phot_table,rgrid_phot_table,freefree_phot_table,apertures,annulus_apertures,sfrfreefree,sfrha,Aha,t2,Ahafreefree,Ahafreefreep,Ahafreefreem)


def aperturesnew(fname,apr):
        import warnings
        warnings.simplefilter("ignore") 
            #apr is aperture diameter in pixels
        #ha = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/working_ngc5194_ha_arc2.fits')[0]       
        ha = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ha_2arc_sings.fits')[0]

        ir24 = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ir24.fits')[0]       
        ir70 = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ir70.fits')[0]       
        ir12 = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ir12.fits')[0]       
        ir8 = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ir8.fits')[0]       
        h1 = fits.open('/Users/kessler.363/Desktop/PaBeta/GasData/rep/ngc5194_hi_col_rep.fits')[0]
        h2 = fits.open('/Users/kessler.363/Desktop/PaBeta/GasData/rep/ngc5194_heracles_h2_col_rep.fits')[0]
        h2hires = fits.open('/Users/kessler.363/Desktop/PaBeta/GasData/rep/ngc5194_paws_h2_col_rep.fits')[0] 

        rgrid = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/rgrid.fits')[0]

        phot_tables = []
        stds = []
        #cmap = plt.get_cmap("rainbow")
        t = Table.read('/Users/kessler.363/Desktop/PaBeta/M51/analysis/NGC5194_Aperture_Locs_test.csv',format = 'ascii',delimiter = ',')
        #t = Table.read('NGC5194_Aperture_Locs2back.csv',format = 'ascii',delimiter = ',')


        f = fits.open(fname)[0]
        f.data/=2.
        w = wcs.WCS(f.header)
        ster = getPixelSize(w) 
        f.data*=ster
        nanmask = f.data.copy()
        nanmask[~np.isnan(nanmask)] = 1.


        
        haimg = ha.data
        rgridimg = rgrid.data
        ir24img = ir24.data
        ir70img = ir70.data
        ir8img = ir8.data
        ir12img = ir12.data
        h1img = h1.data
        h2img = h2.data
        h2hiresimg = h2hires.data
        #convert to per pixel instead of per ster and apply nanmask
        haimg *=nanmask*ster
        ir24img*=nanmask*ster
        ir70img*=nanmask*ster
        ir12img*=nanmask*ster
        ir8img*=nanmask*ster
        rgridimg *=nanmask
        h1img*=nanmask
        h2img*=nanmask
        h2hiresimg*=nanmask 

        haerr = 2.772485559909721e+35
        paberr = 2.3849392462793174e+35



        world = [ [t['ra'][j],t['dec'][j]] for j in np.arange(len(t['ra']))]
        pixcoords = w.wcs_world2pix(world,0)



        apertures = CircularAperture(pixcoords,r = apr/2.)
        annulus_apertures = CircularAnnulus(pixcoords, r_in=apr/2. + 4, r_out=apr/2.+150)
        annulus_masks = annulus_apertures.to_mask(method='center')
        #rgrid apertures for distance from cetner of gal (np.pi/180.)*dist*1e3
        rgridimg *= (np.pi/180.)*8580.
        rgrid_phot_table = aperture_photometry(rgridimg, apertures)
        rgrid_phot_table['aperture_sum']/= apertures.area
        rgrid_phot_table.rename_column('aperture_sum','r_kpc')

        #paB apertures
        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(f.data)
            annulus_data_1d = annulus_data[mask.data>0]
            _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        pab_bkg_median = np.array(bkg_median)

        pab_phot_table = aperture_photometry(f.data , apertures)
        pab_phot_table['annulus_median'] = pab_bkg_median
        pab_phot_table['aper_bkg'] = pab_bkg_median*apertures.area
        pab_phot_table['aper_sum_bkgsub'] = pab_phot_table['aperture_sum'] - pab_phot_table['aper_bkg']
        pab_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
        pab_phot_table.rename_column('aper_sum_bkgsub','pab_aperture_sum')
        ras =  Column(data = t['ra'],name = 'ra')
        decs = Column(data = t['dec'],name = 'dec')
        pab_phot_table.add_column(ras,index = 1)
        pab_phot_table.add_column(decs,index = 2)

      
        #ha apertures
        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(haimg)
            annulus_data_1d = annulus_data[mask.data>0]
            _,median_sigclip,_ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
        ha_bkg_median = np.array(bkg_median)

        ha_phot_table = aperture_photometry(haimg , apertures)
        ha_phot_table['avg_aper'] = ha_phot_table['aperture_sum']/apertures.area
        ha_phot_table['annulus_median'] = ha_bkg_median
        ha_phot_table['aper_bkg'] = ha_bkg_median*apertures.area
        ha_phot_table['aper_sum_bkgsub'] = ha_phot_table['aperture_sum'] - ha_phot_table['aper_bkg']
        ha_phot_table['aper_sum_bkgsub']*=(4*np.pi*(2.65e25)**2)
        ha_phot_table.rename_column('aper_sum_bkgsub','ha_aperture_sum')
                

        #24 apertures
        ir24_phot_table = aperture_photometry(ir24img , apertures)
        ir24_phot_table['aperture_sum']/=apertures.area
        ir24_phot_table['aperture_sum']*=const.c.to(u.micron/u.s).value/24. 
        ir24_phot_table['aperture_sum'] = ha_phot_table['avg_aper']/ir24_phot_table['aperture_sum']
        ir24_phot_table.rename_column('aperture_sum','ir24_aperture_sum')

        #70 apertures
        ir70_phot_table = aperture_photometry(ir70img , apertures)
        ir70_phot_table['aperture_sum']/=apertures.area
        ir70_phot_table['aperture_sum']*=const.c.to(u.micron/u.s).value/70.  
        ir70_phot_table['aperture_sum'] = ha_phot_table['avg_aper']/ir70_phot_table['aperture_sum']        
        ir70_phot_table.rename_column('aperture_sum','ir70_aperture_sum')


        #12 apertures

        ir12_phot_table = aperture_photometry(ir12img , apertures)
        ir12_phot_table['aperture_sum']/=apertures.area
        ir12_phot_table['aperture_sum']*=const.c.to(u.micron/u.s).value/12.
        ir12_phot_table['aperture_sum'] = ha_phot_table['avg_aper']/ir12_phot_table['aperture_sum']
        
        ir12_phot_table.rename_column('aperture_sum','ir12_aperture_sum')

        #8 apertures
        ir8_phot_table = aperture_photometry(ir8img , apertures)
        ir8_phot_table['aperture_sum']/=apertures.area
        ir8_phot_table['aperture_sum']*=const.c.to(u.micron/u.s).value/8. 
        ir8_phot_table['aperture_sum'] = ha_phot_table['avg_aper']/ir8_phot_table['aperture_sum']
        
        ir8_phot_table.rename_column('aperture_sum','ir8_aperture_sum')
                
        #h1 apertures
        h1_phot_table = aperture_photometry(h1img , apertures)
        h1_phot_table['aperture_sum']/=apertures.area
        h1_phot_table.rename_column('aperture_sum','h1_aperture_sum')


        #h2 apertures
        h2_phot_table = aperture_photometry(h2img , apertures)
        h2_phot_table['aperture_sum']/=apertures.area
        h2_phot_table.rename_column('aperture_sum','h2_aperture_sum')
                
        #h2hires apertures
        h2hires_phot_table = aperture_photometry(h2hiresimg , apertures)
        h2hires_phot_table['aperture_sum']/=apertures.area
        h2hires_phot_table.rename_column('aperture_sum','h2_aperture_sum')
                

        print(len(pab_phot_table),len(ha_phot_table),len(rgrid_phot_table))

        phottable = join(pab_phot_table,ha_phot_table)
     

        print(len(pab_phot_table),len(ha_phot_table),len(rgrid_phot_table))

        Aha,extimg,hacorr = HaExtinctionMags(ha_phot_table['ha_aperture_sum'],pab_phot_table['pab_aperture_sum'],'PaB','Cardelli',3.1)

        mask = np.logical_and(Aha>0.,np.logical_and(pab_phot_table['pab_aperture_sum']>0.,ha_phot_table['ha_aperture_sum']>0.))

        merrs,perrs = getAhaerrs(pab_phot_table['pab_aperture_sum'][mask],ha_phot_table['ha_aperture_sum'][mask])

        t2 = Table([t['ra'][mask],t['dec'][mask],ha_phot_table['ha_aperture_sum'][mask],pab_phot_table['pab_aperture_sum'][mask],Aha[mask]],names = ('RA','Dec','LHa','LPab','AHa'))

        t3 = Table([t['ra'][mask],t['dec'][mask],ir24_phot_table['ir24_aperture_sum'][mask],ir70_phot_table['ir70_aperture_sum'][mask],ir8_phot_table['ir8_aperture_sum'][mask],ir12_phot_table['ir12_aperture_sum'][mask]],names = ('RA','Dec','ir24','ir70','ir8','ir12'))


        t3.write('IRTable5194.txt',format = 'ascii',overwrite=True)
        return(pab_phot_table[mask],ha_phot_table[mask],rgrid_phot_table[mask],ir24_phot_table[mask],ir70_phot_table[mask],ir8_phot_table[mask],ir12_phot_table[mask],h1_phot_table[mask],h2_phot_table[mask],h2hires_phot_table[mask],apertures,annulus_apertures,Aha[mask],hacorr[mask],extimg[mask],merrs,perrs,t['ra'][mask],t['dec'][mask],mask,t2)

def getAhaerrs(pab,ha):

    haerr = 2.772485559909721e+35
    paberr = 2.3849392462793174e+35

    Aha,haext,hacorr = HaExtinctionMags(ha,pab,'PaB','Cardelli',3.1)
    Ahap,haextp,hacorrp = HaExtinctionMags(ha-haerr,pab+paberr,'PaB','Cardelli',3.1)
    Aham,haextm,hacorrm = HaExtinctionMags(ha+haerr,pab-paberr,'PaB','Cardelli',3.1)

    merrs = Aham
    perrs = Ahap

    return(merrs,perrs)



def freefreeToSFR(data,T=None):
        #data in units of erg/s/Hz
        #default T = 10^4
        if T== None:
                T = 6300.

        SFR = 4.6e-28* (T/(1.e4))**(-0.45) * (33.)**(0.1)  * (data)

        return(SFR)


def dustModel(wavelength):
        from scipy.interpolate import CubicSpline

        #wavelength in angstroms
        #uses anchor points from Fitzpatrick '99 and a cubic spline interpolator to     return the extintion value (IN MAGS) at a given wavelength (in the optical and IR)
        #for R=3.1 which is typical for MW      

        anchorLambdaA = np.array([np.inf,26500.,12200.,6000.,5470.,4670.,4110.,2700.,2600.])
        anchorInvLambdaMicron = np.array([0.000,0.377,0.820,1.667,1.828,2.141,2.433,3.704,3.846])
        anchorRval = np.array([0.000,0.265,0.829,2.688,3.055,3.806,4.315,6.265,6.591])

        rCurve = CubicSpline(anchorInvLambdaMicron,anchorRval)

        wavelengthInvMicron = 1./(wavelength*0.0001)

        extinction = np.around(rCurve(wavelengthInvMicron),decimals = 3)

        return(float(extinction))

def deReddenModel(flux,wavelength):
        #returns dereddened flux
        ext = dustModel(wavelength)
        tauLam = np.log(10**(ext/2.5))
        deredFlux = flux/np.exp(-tauLam)

        return(deredFlux,ext)

def deRedden(f,ext):
        #returns dereddened flux
        flux,hdr = fits.getdata(f,header=True)
        tauLam = np.log(10**(ext/2.5))
        deredFlux = flux/np.exp(-tauLam)

        fits.writeto('reproj/convolved/dered/'+str(fname)+'.fits',deredFlux,header=hdr,overwrite=True)


def getFluxgivenExt(Aha,lam,intrinsic_ratio):
        #lam in anfstroms
        
        kratio = dustModel(6562.8)/dustModel(lam)

        #Aha =  (dustModel(6562.8)/(dustModel(1.284e4)-dustModel(6562.8)))*2.5*np.log10((Fha/Fpab)/intrinsic_ratio)
        F_ratio = 10**((Aha/2.5)*((dustModel(lam)-dustModel(6562.8))/dustModel(6562.8)))*intrinsic_ratio

        return(F_ratio)

def getFluxImagegivenAha(data,Aha):
        intrinsic_ratio = 2.85/0.162
        kratio = dustModel(6562.8)/dustModel(1.284e4)

        #Aha =  (dustModel(6562.8)/(dustModel(1.284e4)-dustModel(6562.8)))*2.5*np.log10((Fha/Fpab)/intrinsic_ratio)
        #F_ratio = 10**((Aha/2.5)*((dustModel(1.284e4)-dustModel(6562.8))/dustModel(6562.8)))*intrinsic_ratio
        
        extimg = ((10**(Aha/2.5)) -1.)*data

        return(extimg)

def getEBminusV(Fha,Fpab):
        intrinsic_ratio = 2.85/0.162
        Fpab+=.6e-6

        EBV = ((dustModel(6562.8) - (dustModel(1.284e4))/2.5)*np.log10(intrinsic_ratio*(Fpab/Fha)))

        from astropy.visualization import hist
        from scipy.stats import norm
        import matplotlib.mlab as mlab
        import matplotlib.pyplot as plt

        EBV = np.log10(EBV)
        n,bins,other = hist(EBV[np.isfinite(EBV)],bins = 'knuth',normed = 1)
        (mu, sigma) = norm.fit(EBV[np.isfinite(EBV)])
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)

        plt.xlabel(r'A(H$\alpha$)')
        plt.ylabel('N')
        plt.tight_layout()
        plt.show()
        return(10**mu)


def HaExtinctionMags(hadata,otherdata,whichpaschen,model,Rv):
        
        H1 = pn.RecAtom('H',1)
        Halpha = H1.getEmissivity(tem = 6300., den = 1e3, lev_i = 3, lev_j = 2)
        Paalpha = H1.getEmissivity(tem = 6300., den = 1e3, lev_i = 4, lev_j = 3)
        Pabeta = H1.getEmissivity(tem = 6300., den = 1e3, lev_i = 5, lev_j = 3)
        Hbeta = H1.getEmissivity(tem = 6300,den = 1e3,lev_i =4,lev_j = 2)

        exCardelli = CCM89(Rv = Rv)
        exF99 = F99(Rv = Rv)

        halam = 6265.8*u.AA
        if whichpaschen == 'PaB':
                lam = 1.284*u.micron
        elif whichpaschen == 'Paalpha':
                lam = 1.875*u.micron
        else:
                lam = (486.14*u.nm).to(u.micron)

        if model == 'Fitzpatrick':
                kHa =  exF99(halam)*Rv
                kOther = exF99(lam)*Rv
        else:
                kHa =  exCardelli(halam)*Rv
                kOther = exCardelli(lam)*Rv     
        


        if whichpaschen == 'PaB':
                Aha =   (kHa/(kOther-kHa))*2.5*np.log10((hadata/otherdata)/(Halpha/Pabeta))
        elif whichpaschen == 'Paalpha':
                Aha =   (kHa/(kOther-kHa))*2.5*np.log10((hadata/otherdata)/(Halpha/Paalpha))

        elif  whichpaschen == 'freefree':
                Aha =   2.5*np.log10(otherdata/hadata)
        else:

                Aha =   (kHa/(kOther-kHa))*2.5*np.log10((hadata/otherdata)/(Halpha/Hbeta))
        
        extimg = ((10**(Aha/2.5) -1.))*hadata
        hacorr = (10**(Aha/2.5))*hadata


        return(Aha,extimg,hacorr)

def getAlphaFactor(apsize): 
    return((((2*np.pi)/(360*3600.))**2)*1e6*1e-23*4*np.pi*(apsize/2.)**2)

def getAhaIRcurve(xrang,alpha):
    return(2.5*np.log10(1+alpha*(1./xrang)))

def getRatiofromAHa(Aha,whichpaschen):
    Rv = 3.1
    H1 = pn.RecAtom('H',1)
    Halpha = H1.getEmissivity(tem = 6300., den = 1e3, lev_i = 3, lev_j = 2)
    Paalpha = H1.getEmissivity(tem = 6300., den = 1e3, lev_i = 4, lev_j = 3)
    Pabeta = H1.getEmissivity(tem = 6300., den = 1e3, lev_i = 5, lev_j = 3)
    Hbeta = H1.getEmissivity(tem = 6300,den = 1e3,lev_i =4,lev_j = 2)
    if whichpaschen == 'PaB':
            lam = 1.284*u.micron
    else:
            lam = (486.14*u.nm).to(u.micron)



    halam = 6265.8*u.AA

    exCardelli = CCM89(Rv = Rv)
    
    kHa =  exCardelli(halam)*Rv
    kOther = exCardelli(lam)*Rv     
    if whichpaschen == 'PaB':
            print(kHa,kOther)
            print(10**((Aha/2.5)*((kOther-kHa)/kHa))*(Halpha/Pabeta))
            return(10**((Aha/2.5)*((kOther-kHa)/kHa))*(Halpha/Pabeta))
    else:
            print(kHa,kOther)
            print(10**((Aha/2.5)*((kOther-kHa)/kHa))*(Halpha/Hbeta))
            return(10**((Aha/2.5)*((kOther-kHa)/kHa))*(Halpha/Hbeta))



def makeMixtureModel(Aharange,Lharange):
    Rv = 3.1
    H1 = pn.RecAtom('H',1)
    Halpha = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 3, lev_j = 2)
    Paalpha = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 4, lev_j = 3)
    Pabeta = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i= 5, lev_j = 3)
    Hbeta = H1.getEmissivity(tem = 1e4,den = 1e3,lev_i =4,lev_j = 2)
   
    exCardelli = CCM89(Rv = Rv)
    kHa =  exCardelli(6265.8*u.AA)*Rv
    kHB = exCardelli(486.14*u.nm)*Rv  
    kPab = exCardelli(1.284*u.micron)*Rv  

    Ahb = Aharange*(kHB/kHa)
    Apab = Aharange*(kPab/kHa)
 
    Lhbrange = Lharange/((Halpha/Hbeta))
    Lpabrange = Lharange/((Halpha/Pabeta))

    Foha = 10**(-Aharange/2.5) *Lharange 
    Fohb =  10**(-Ahb/2.5)*Lhbrange
    Fopab = 10**(-Apab/2.5)*Lpabrange


    Foha = np.sum(i for i in Foha[np.isfinite(Foha)])
    Fohb = np.sum(i for i in Fohb[np.isfinite(Fohb)])
    Fopab = np.sum(i for i in Fopab[np.isfinite(Fopab)])
    
    #print(Foha/Fohb,Foha/Fopab)


    return((Foha/Fohb),(Foha/Fopab))

def makeMixtureModelLha(Aharange,mixture=None):
    Rv = 3.1
    H1 = pn.RecAtom('H',1)
    Halpha = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 3, lev_j = 2)
    Pabeta = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 5, lev_j = 3) 

    exCardelli = CCM89(Rv = Rv)

    Lharange = np.zeros_like(Aharange,dtype = 'f8')

    if mixture!=None:
        Ahaobs = []
        HaHbs = []
        HaPabs = []
        for i in np.arange(len(Aharange)):
            Lharange = np.zeros_like(Aharange,dtype = 'f8')            
            Lharange[:i+1] = np.round(1./(i+1),decimals =2)
            #get observed line ratios
            HaHb,HaPab = makeMixtureModel(Aharange,Lharange)
            HaHbs.append(HaHb)
            HaPabs.append(HaPab)
            #calculate observed Aha
            halam = 6265.8*u.AA
            lam = 1.284*u.micron
            kHa =  exCardelli(halam)*Rv
            kOther = exCardelli(lam)*Rv     
            Ahaob =   (kHa/(kOther-kHa))*2.5*np.log10((HaPab)/(Halpha/Pabeta))
            print(len(Aharange) - i)
            Ahaobs.append(Ahaob)
    else:
        Ahaobs = []
        HaHbs = []
        HaPabs = []
        for i in np.arange(len(Aharange)):
            Lharange = np.zeros_like(Aharange,dtype = 'f8')
            Lharange[i] = 1.
            #print(Lharange)
            #get observed line ratios
            HaHb,HaPab = makeMixtureModel(Aharange,Lharange)
            HaHbs.append(HaHb)
            HaPabs.append(1./HaPab)

            #calculate observed Aha
            halam = 6265.8*u.AA
            lam = 1.284*u.micron
            kHa =  exCardelli(halam)*Rv
            kOther = exCardelli(lam)*Rv     
            Ahaob =   (kHa/(kOther-kHa))*2.5*np.log10((HaPab)/(Halpha/Pabeta))
            print(len(Aharange) - i)
            Ahaobs.append(Ahaob)

        
    return(HaHbs,HaPabs,Ahaobs)


def makeMixtureModelGas(numrange,mixture=None,factor = None):
    Rv = 3.1
    H1 = pn.RecAtom('H',1)
    Halpha = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 3, lev_j = 2)
    Pabeta = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 5, lev_j = 3) 

    exCardelli = CCM89(Rv = Rv)

    if factor == None:
        Gasrange = numrange.copy()
    else:
        Gasrange = numrange.copy()*factor

    Avrange = Gasrange/1.9e21
    Aharange =  Avrange*exCardelli(6265.8*u.AA)
    Lharange = np.zeros_like(Aharange,dtype = 'f8')

    if mixture!=None:
        Ahaobs = []
        for i in np.arange(len(Gasrange)):
            Lharange = np.zeros_like(Aharange,dtype = 'f8')            
            Lharange[:i+1] = np.round(1./(i+1),decimals =2)
            #get observed line ratios
            HaHb,HaPab = makeMixtureModel(Aharange,Lharange)

            #calculate observed Aha
            halam = 6265.8*u.AA
            lam = 1.284*u.micron
            kHa =  exCardelli(halam)*Rv
            kOther = exCardelli(lam)*Rv     
            Ahaob =   (kHa/(kOther-kHa))*2.5*np.log10((HaPab)/(Halpha/Pabeta))
            print(Ahaob)
            print(len(Gasrange) - i)
            Ahaobs.append(Ahaob)
    else:
        Ahaobs = Aharange.copy()
    return(Ahaobs,Gasrange)

def SetupMonteCarlo(hatrue,haerr,pabtrue,paberr,temp,temperr,roerr,gainerr,n):

    #Normally distrubuted values of ha and Pab
    
    #add noise to data
    ha_sim = hatrue+haerr*np.random.normal()
    pab_sim = pabtrue+paberr*np.random.normal()

    #randomly choose temp
    temp_sim = np.random.normal(temp,temperr)

    #randomly choose density
    ro_sim = roerr

    #add gain error
    ha_sim*=gainerr
    pab_sim*=gainerr

    return(ha_sim,pab_sim,temp_sim,ro_sim)

def AHaMonteCarlo(hadata,otherdata,T,ro,model):
        
        H1 = pn.RecAtom('H',1)
        Halpha = H1.getEmissivity(tem = T, den = ro, lev_i = 3, lev_j = 2)
        Pabeta = H1.getEmissivity(tem = T, den = ro, lev_i = 5, lev_j = 3)

        Rv = 3.1
        halam = 6265.8*u.AA
        lam = 1.284*u.micron

        exCardelli = CCM89(Rv = Rv)
        exF99 = F99(Rv = Rv)

        if model == 1:
                kHa =  exF99(halam)*Rv
                kOther = exF99(lam)*Rv
        else:
                kHa =  exCardelli(halam)*Rv
                kOther = exCardelli(lam)*Rv   


        Aha =   (kHa/(kOther-kHa))*2.5*np.log10((hadata/otherdata)/(Halpha/Pabeta))
        
        return(Aha)

def RunMonteCarlo(T=None,ro = None,ropert=None,temppert=None,kpert=None,errpert=None,n = None,tstatic= None,verbose=None):

    
    #tables = ['NGC5194_backpert_02_arc.txt','NGC5194_backpert_12_arc.txt','NGC5194_backpert_22_arc.txt','NGC5194_backpert_32_arc.txt','NGC5194_backpert_42_arc.txt','NGC5194_backpert_52_arc.txt','NGC5194_Aperture_Table_2arc.txt']

    tables = ['NGC5194_backpert_ff_02_arc.txt','NGC5194_backpert_ff_12_arc.txt','NGC5194_backpert_ff_22_arc.txt','NGC5194_backpert_ff_32_arc.txt','NGC5194_backpert_ff_42_arc.txt','NGC5194_backpert_ff_52_arc.txt','NGC5194_Aperture_Table_3arc.txt']

        
    #remove old tables in directory
    #os.system('rm MonteCarloTables/*')
    os.system('rm MonteCarloTablesFF/*')


    #previously calculated rms errors
    if errpert != None:
        haerr = 0
        paberr = 0
    else:
        haerr = 8.65e34
        paberr = 9.06e34
    #gain errors



    #set # of times to run. Default is 1e3 times.
    if n !=None:
        N = n
    else:
        N = 1000

    #Setup Temperature. Default is 1e4, sigma is 1000K
    if T!=None:
        temp = T
    else:
        temp = 10000.
    temperr = 1000.

    if ro!= None:
        Ro = ro
    else:
        Ro = 1e3

    Ahascatter = []
    for i in np.arange(N):

        #randomly choose map
        if tstatic == None:
            ch = random.choice(tables)
            t = Table.read(ch,format = 'ascii')
        else:
            #t = Table.read('NGC5194_Aperture_Table_2arc.txt',format = 'ascii')
            t = Table.read('NGC5194_Aperture_Table_3arc.txt',format = 'ascii')
        
        #read in data
        hatrue = t['LHa']
        pabtrue = t['LPab']
        ahatrue = t['AHa']

        roerr = random.choice(np.linspace(1e2,1e4)) 
        gainerr = 1 - random.choice([.05,.06,.07])*random.choice([-1,1])
        
        ha_sim,pab_sim,temp_sim,ro_sim = SetupMonteCarlo(hatrue,haerr,pabtrue,paberr,temp,temperr,roerr,gainerr,N)

        if ropert != None:
            ro_sim = Ro
        else:
            pass
        if temppert != None:
            temp_sim = temp
        else:
            pass
        if kpert !=None:
            randint = 2
        else:
            randint = random.choice([1,2])
        if verbose!=None:
            print(ch,ro_sim,temp_sim,randint,np.nanmedian(ha_sim),np.nanmedian(pab_sim))
        else:
            pass

        Aha_sim = AHaMonteCarlo(ha_sim,pab_sim,temp_sim,ro_sim,randint)
        
        Ahascatter.append(np.nanmedian(Aha_sim))
        temp_sim = np.full_like(ha_sim,temp_sim).tolist()
        t = Table([ha_sim,pab_sim,temp_sim,Aha_sim],names = ('LHa','LPab','T','AHa'))
        #t.write('MonteCarloTables/table'+str(int(i))+'.txt',format = 'ascii',overwrite=True)
        t.write('MonteCarloTablesFF/table'+str(int(i))+'.txt',format = 'ascii',overwrite=True)

    print(np.nanstd(Ahascatter))

def getValuesfromMCTables(t):
    #tables = glob('MonteCarloTables/*.txt')
    tables = glob('MonteCarloTablesFF/*.txt')

    dfLha = pd.DataFrame()
    dfLpab = pd.DataFrame()
    dfAha = pd.DataFrame()

    for i in np.arange(len(tables)):
        tab = Table.read(tables[i],format = 'ascii')
        df = tab.to_pandas()
        dfLha['LHa'+str(i)] = df['LHa']
        dfLpab['LPab'+str(i)] = df['LPab']
        dfAha['AHa'+str(i)] = df['AHa']

    t['E_LHa'] = dfLha.std(axis=1)
    t['E_Pab'] = dfLpab.std(axis=1)
    t['E_AHa'] = dfAha.std(axis=1)

    return(t)


def irFunc(x,factor):
    return(2.5*np.log10(1+factor/x)) 

def fitIR(data,Aha,band,verbose=None,factor=None):
    popt, pcov = curve_fit(irFunc, data*const.c.to(u.micron/u.s).value/band,Aha)
    if factor != None:
        factor1 = factor
    else:
        factor1 = np.round(popt[0],decimals=3)
    if verbose==True:
        print(factor1)
    xr = np.linspace(1e-5,.5,len(Aha))

    y = 2.5*np.log10(1+factor1/xr)

    return(xr,y)
    

def makeIRPlots(galaxy):
    if galaxy == 'NGC6946':
        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  

        irt = Table.read('../../NGC6946/analysis/IRTable6946.txt',format = 'ascii')
        ir8 = irt['ir8']
        ir12 = irt['ir12']
        ir24 = irt['ir24']
        ir70 = irt['ir70']
        ir100 = irt['ir100']
        Aha = t['AHa']
        factor2 = [0.011,0,0.038,0.011,0]
        factor3 = [0,0,0.031,0,0]
        citations2 = ['Kennicutt 09','','Kennicutt 07','Li 13','']
        citations3 = ['','','Calzetti 07','','','']
        irbands = [ir8,ir12,ir24,ir70,ir100]
        bands = [8,12,24,70,100]
        err = t['E_AHa']
        r = t1['r']    
        mask = r>1.


    elif galaxy == 'NGC5194':
        t = Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1 = Table.read('m51Aha+H.txt',format = 'ascii') 
        irt = Table.read('IRTable5194.txt',format = 'ascii')

        ir8 = irt['ir8']
        ir12 = irt['ir12']
        ir24 = irt['ir24']
        ir70 = irt['ir70']
        Aha = t['AHa']
        factor2 = [0.011,0,0.038,0.011]
        factor3 = [0,0,0.031,0]
        citations2 = ['Kennicutt 09','','Kennicutt 07','Li 13']
        citations3 = ['','','Calzetti 07','','']
        irbands = [ir8,ir12,ir24,ir70]
        bands = [8,12,24,70]
        err = t['E_AHa']
        r = t1['r'] 
        mask = r>2.
        xlims = [[-3,-1],[-2.5,-.5],[-2.8,-.25],[-3.5,-1.25]]

    else:

        t6946 = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t16946 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  

        irt6946 = Table.read('../../NGC6946/analysis/IRTable6946.txt',format = 'ascii')

        m51t = Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        m51t1 = Table.read('m51Aha+H.txt',format = 'ascii') 
        m51irt = Table.read('IRTable5194.txt',format = 'ascii')

        t = vstack([t6946,m51t])
        t1 = vstack([t16946,m51t1])
        irt = vstack([irt6946,m51irt])
        print(len(t))

        ir8 = irt['ir8']
        ir12 = irt['ir12']
        ir24 = irt['ir24']
        ir70 = irt['ir70']
        ir100 = irt['ir100']
        Aha = t['AHa']
        factor2 = [0.011,0,0.038,0.011,0]
        factor3 = [0,0,0.031,0,0]
        citations2 = ['Kennicutt 09','','Kennicutt 07','Li 13','']
        citations3 = ['','','Calzetti 07','','','']
        irbands = [ir8,ir12,ir24,ir70,ir100]
        bands = [8,12,24,70,100]
        err = t['E_AHa']
        r = t1['r']    
        mask = r>2.
    for i in np.arange(len(irbands)):
        data = irbands[i]
        band = bands[i]

        if factor2[i]!=0:
            popt, pcov = curve_fit(irFunc, data[mask]*const.c.to(u.micron/u.s).value/band,Aha[mask],bounds = [factor2[i]-0.3*factor2[i],0.3*factor2[i]+factor2[i]])
        else:
            popt, pcov = curve_fit(irFunc, data[mask]*const.c.to(u.micron/u.s).value/band,Aha[mask])

        factor1 = popt[0]
        print(band)
        print(np.round(popt[0],decimals=3))
        print(np.sqrt(np.diag(pcov)))
        print(0.3*np.round(popt[0],decimals=3))
        xr = np.linspace(1e-5,.5)

        plt.figure()
        plt.errorbar(np.log10(data*const.c.to(u.micron/u.s).value/band),Aha,yerr=err,alpha = 0.5,color = 'gray',fmt = 'none',zorder=0,label = '')
        plt.scatter(np.log10(data*const.c.to(u.micron/u.s).value/band),Aha,c = r,cmap = 'plasma',s=6,label = '',zorder=1)
        plt.plot(np.log10(xr),2.5*np.log10(1+factor1/xr),'k-',lw=2,label = 'This Work')
        plt.plot(np.log10(xr),2.5*np.log10(1+factor1/xr)+2.5*np.log10(1+factor1/xr)*.3,'c:',lw=2,label = '')
        plt.plot(np.log10(xr),2.5*np.log10(1+factor1/xr)-2.5*np.log10(1+factor1/xr)*.3,'c:',lw=2,label = '')

        if factor2[i]!=0:
            label2 = citations2[i]
            print(label2)
            plt.plot(np.log10(xr),2.5*np.log10(1+factor2[i]/xr),'k--',lw=2,label = label2)
        if factor3[i]!=0:
            label3 = citations3[i]

            plt.plot(np.log10(xr),2.5*np.log10(1+factor3[i]/xr),'k-.',lw=2,label = label3)

        plt.grid(linestyle = ':',zorder=0,color = 'gray')
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)
        plt.xlabel(r'log(H$\alpha$/'+str(band)+'$\mu$m)',fontsize=16)

        if np.logical_or(band==8,np.logical_or(band==70,band==100)):
            plt.title(galaxy,fontsize=16)

        cb =plt.colorbar() 
        cb.set_label('r [kpc]',fontsize=16)
        plt.legend(fontsize=16,loc = 'upper right')
        plt.ylim(-2,12)
        plt.tight_layout()
        if galaxy == 'NGC6946':
            plt.savefig('/Users/kessler.363/Desktop/AHavHa'+str(band)+'_6946.png')
            plt.close()
        elif galaxy == 'NGC5194':
            plt.xlim(xlims[i][0],xlims[i][1])
            plt.savefig('/Users/kessler.363/Desktop/AHavHa'+str(band)+'_5194.png')
            plt.close()

        elif galaxy == 'both':
            plt.close()

def makeScatterTable(galaxy):
    if galaxy == 'NGC6946':
        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  

        irt = Table.read('../../NGC6946/analysis/IRTable6946.txt',format = 'ascii')


        
        ir8 = irt['ir8']
        ir12 = irt['ir12']
        ir24 = irt['ir24']
        ir70 = irt['ir70']
        ir100 = irt['ir100']
        
        x8,y8 = fitIR(ir8,t['AHa'],8.,factor = 0.014)
        x12,y12 = fitIR(ir12,t['AHa'],12.,factor = 0.089)
        x24,y24 = fitIR(ir24,t['AHa'],24.,factor = 0.049)
        x70,y70 = fitIR(ir70,t['AHa'],70.,factor = 0.014)
        x100,y100 = fitIR(ir100,t['AHa'],100.,factor = 0.018)


        xr = np.linspace(1e-5,.5,9)

        iridx = np.digitize(t['AHa'],xr)


        running_std_ir8 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y8[iridx==k])) for k in np.arange(len(xr))]) 
        running_std_ir12 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y12[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir24 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y24[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir70 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y70[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir100 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y100[iridx==k])) for k in np.arange(len(xr))])


        running_std_ir8_log = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y8[iridx==k]))) for k in np.arange(len(xr))]) 
        running_std_ir12_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y12[iridx==k]))) for k in np.arange(len(xr))])
        running_std_ir24_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y24[iridx==k]))) for k in np.arange(len(xr))])
        running_std_ir70_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y70[iridx==k])))for k in np.arange(len(xr))])
        running_std_ir100_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y100[iridx==k])))for k in np.arange(len(xr))])


        print(np.nanmedian(running_std_ir8),np.median(running_std_ir12),np.nanmedian(running_std_ir24),np.nanmedian(running_std_ir70))

        numt = Table.read('Screen+MixtureTable6946.txt',format = 'ascii')
        nr = numt['nr']
        screen = numt['screen']
        quarterscreen = numt['quarterscreen']
        halfscreen = numt['halfscreen']
        nrmix = numt['nr_mix']
        mixture = numt['mixture']

        numbins = np.arange(19,23.5,.5)

        idx = np.digitize(np.log10(t1['H']),numbins)

        running_std_screen = np.array([np.std(np.abs(t['AHa'][idx==k]-screen[idx==k])) for k in np.arange(len(numbins))]) 
        running_std_mixture = np.array([np.std(np.abs(t['AHa'][idx==k]-mixture[idx==k])) for k in np.arange(len(numbins))]) 

        running_std_halfscreen = np.array([np.std(np.abs(t['AHa'][idx==k]-halfscreen[idx==k])) for k in np.arange(len(numbins))]) 
        running_std_quarterscreen = np.array([np.std(np.abs(t['AHa'][idx==k]-quarterscreen[idx==k])) for k in np.arange(len(numbins))]) 
        
        running_std_screen_log = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(screen[idx==k]))) for k in np.arange(len(numbins))]) 
        running_std_mixture_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(mixture[idx==k]))) for k in np.arange(len(numbins))]) 

        running_std_halfscreen_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(halfscreen[idx==k]))) for k in np.arange(len(numbins))]) 
        running_std_quarterscreen_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(quarterscreen[idx==k]))) for k in np.arange(len(numbins))]) 


        radialbins = np.linspace(0,12,9)
        idxradial = np.digitize(t1['r'],radialbins)

        popt,pcov = curve_fit(exponential_func,t1['r'],t['AHa'])
        print(popt)
        running_std_radial = np.array([np.std(t['AHa'][idxradial==k]-exponential_func(t1['r'][idxradial==k],*popt)) for k in np.arange(len(radialbins))]) 
        running_std_radial_log = np.array([np.std(np.log10(np.abs(t['AHa'][idxradial==k]))-np.log10(np.abs(t['AHa'][idxradial==k]-exponential_func(t1['r'][idxradial==k],*popt)))) for k in np.arange(len(radialbins))]) 
        

        


        T = Table([xr,running_std_ir8,running_std_ir12,running_std_ir24,running_std_ir70,running_std_ir100,numbins,running_std_screen,running_std_halfscreen,running_std_quarterscreen ,running_std_mixture,radialbins,running_std_radial,running_std_ir8_log,running_std_ir12_log,running_std_ir24_log,running_std_ir70_log,running_std_ir100_log,running_std_screen_log,running_std_halfscreen_log,running_std_quarterscreen_log,running_std_mixture_log,running_std_radial_log],names = ('Ha/IR','ir8','ir12','ir24','ir70','ir100','Gas','screen','halfscreen','quarterscreen','mixture','r','radial_std','logir8','logir12','logir24','logir70','logir100','logscreen','loghalfscreen','logquarterscreen','logmixture','logradial_std'))
        T.write('NGC6946_binned_STDs.txt',format = 'ascii',overwrite=True)

    elif galaxy == 'NGC5194':
        t = Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1 = Table.read('m51Aha+H.txt',format = 'ascii') 
        irt = Table.read('IRTable5194.txt',format = 'ascii')

        ir8 = irt['ir8']
        ir12 = irt['ir12']
        ir24 = irt['ir24']
        ir70 = irt['ir70']

        x8,y8 = fitIR(ir8,t['AHa'],8.,factor = 0.009)
        x12,y12 = fitIR(ir12,t['AHa'],12.,factor = 0.025)
        x24,y24 = fitIR(ir24,t['AHa'],24.,factor = 0.029)
        x70,y70 = fitIR(ir70,t['AHa'],70.,factor = 0.014)


        xr = np.linspace(1e-5,.5,9)

        iridx = np.digitize(t['AHa'],xr)


        running_std_ir8 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y8[iridx==k])) for k in np.arange(len(xr))]) 
        running_std_ir12 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y12[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir24 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y24[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir70 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y70[iridx==k])) for k in np.arange(len(xr))])

        running_std_ir8_log = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y8[iridx==k]))) for k in np.arange(len(xr))]) 
        running_std_ir12_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y12[iridx==k]))) for k in np.arange(len(xr))])
        running_std_ir24_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y24[iridx==k]))) for k in np.arange(len(xr))])
        running_std_ir70_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y70[iridx==k])))for k in np.arange(len(xr))])

        print(np.nanmedian(running_std_ir8),np.median(running_std_ir12),np.nanmedian(running_std_ir24),np.nanmedian(running_std_ir70))

        numt = Table.read('Screen+MixtureTable5194.txt',format = 'ascii')
        nr = numt['nr']
        screen = numt['screen']
        quarterscreen = numt['quarterscreen']
        halfscreen = numt['halfscreen']
        nrmix = numt['nr_mix']
        mixture = numt['mixture']

        numbins = np.arange(19,23.5,.5)

        idx = np.digitize(np.log10(t1['H']),numbins)

        running_std_screen = np.array([np.std(np.abs(t['AHa'][idx==k]-screen[idx==k])) for k in np.arange(len(numbins))]) 
        running_std_mixture = np.array([np.std(np.abs(t['AHa'][idx==k]-mixture[idx==k])) for k in np.arange(len(numbins))]) 

        running_std_halfscreen = np.array([np.std(np.abs(t['AHa'][idx==k]-halfscreen[idx==k])) for k in np.arange(len(numbins))]) 
        running_std_quarterscreen = np.array([np.std(np.abs(t['AHa'][idx==k]-quarterscreen[idx==k])) for k in np.arange(len(numbins))]) 
        
        running_std_screen_log = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(screen[idx==k]))) for k in np.arange(len(numbins))]) 
        running_std_mixture_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(mixture[idx==k]))) for k in np.arange(len(numbins))]) 

        running_std_halfscreen_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(halfscreen[idx==k]))) for k in np.arange(len(numbins))]) 
        running_std_quarterscreen_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(quarterscreen[idx==k]))) for k in np.arange(len(numbins))]) 


        radialbins = np.linspace(0,12,9)
        idxradial = np.digitize(t1['r'],radialbins)

        popt,pcov = curve_fit(exponential_func,t1['r'],t['AHa'])
        print(popt)
        running_std_radial = np.array([np.std(t['AHa'][idxradial==k]-exponential_func(t1['r'][idxradial==k],*popt)) for k in np.arange(len(radialbins))]) 
        running_std_radial_log = np.array([np.std(np.log10(np.abs(t['AHa'][idxradial==k]))-np.log10(np.abs(t['AHa'][idxradial==k]-exponential_func(t1['r'][idxradial==k],*popt)))) for k in np.arange(len(radialbins))]) 
        

        


        T = Table([xr,running_std_ir8,running_std_ir12,running_std_ir24,running_std_ir70,numbins,running_std_screen,running_std_halfscreen,running_std_quarterscreen ,running_std_mixture,radialbins,running_std_radial,running_std_ir8_log,running_std_ir12_log,running_std_ir24_log,running_std_ir70_log,running_std_screen_log,running_std_halfscreen_log,running_std_quarterscreen_log,running_std_mixture_log,running_std_radial_log],names = ('Ha/IR','ir8','ir12','ir24','ir70','Gas','screen','halfscreen','quarterscreen','mixture','r','radial_std','logir8','logir12','logir24','logir70','logscreen','loghalfscreen','logquarterscreen','logmixture','logradial_std'))
        T.write('NGC5194_binned_STDs.txt',format = 'ascii',overwrite=True)

    else:
            
        t6946 = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t16946 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  

        t5194 =  Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t15194 = Table.read('m51Aha+H.txt',format = 'ascii') 
        
        irt5194 = Table.read('IRTable5194.txt',format = 'ascii')
        irt6946 = Table.read('../../NGC6946/analysis/IRTable6946.txt',format = 'ascii')

        t = vstack([t6946,t5194],join_type = 'inner')
        t1 = vstack([t16946,t15194],join_type = 'inner')
        irt = vstack([irt6946,irt5194],join_type = 'inner')


        
        ir8 = irt['ir8']
        ir12 = irt['ir12']
        ir24 = irt['ir24']
        ir70 = irt['ir70']

        x8,y8 = fitIR(ir8,t['AHa'],8.)
        x12,y12 = fitIR(ir12,t['AHa'],12.)
        x24,y24 = fitIR(ir24,t['AHa'],24.,factor = 0.031)
        x70,y70 = fitIR(ir70,t['AHa'],70.)

        xr = np.linspace(1e-5,.5,9)

        iridx = np.digitize(t['AHa'],xr)

        running_std_ir8 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y8[iridx==k])) for k in np.arange(len(xr))]) 
        running_std_ir12 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y12[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir24 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y24[iridx==k])) for k in np.arange(len(xr))])
        running_std_ir70 = np.array([np.std(np.abs(t['AHa'][iridx==k]-y70[iridx==k])) for k in np.arange(len(xr))])

        running_std_ir8_log = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y8[iridx==k]))) for k in np.arange(len(xr))]) 
        running_std_ir12_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y12[iridx==k]))) for k in np.arange(len(xr))])
        running_std_ir24_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y24[iridx==k]))) for k in np.arange(len(xr))])
        running_std_ir70_log  = np.array([np.std(np.log10(np.abs(t['AHa'][iridx==k]))-np.log10(np.abs(y70[iridx==k])))for k in np.arange(len(xr))])

        print(np.nanmedian(running_std_ir8),np.median(running_std_ir12),np.nanmedian(running_std_ir24),np.nanmedian(running_std_ir70))

        numt = Table.read('Screen+MixtureTableBoth.txt',format = 'ascii')
        nr = numt['nr']
        screen = numt['screen']
        quarterscreen = numt['quarterscreen']
        halfscreen = numt['halfscreen']
        nrmix = numt['nr_mix']
        mixture = numt['mixture']

        numbins = np.arange(19,23.5,.5)

        idx = np.digitize(np.log10(t1['H']),numbins)

        running_std_screen = np.array([np.std(np.abs(t['AHa'][idx==k]-screen[idx==k])) for k in np.arange(len(numbins))]) 
        running_std_mixture = np.array([np.std(np.abs(t['AHa'][idx==k]-mixture[idx==k])) for k in np.arange(len(numbins))]) 

        running_std_halfscreen = np.array([np.std(np.abs(t['AHa'][idx==k]-halfscreen[idx==k])) for k in np.arange(len(numbins))]) 
        running_std_quarterscreen = np.array([np.std(np.abs(t['AHa'][idx==k]-quarterscreen[idx==k])) for k in np.arange(len(numbins))]) 
        
        running_std_screen_log = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(screen[idx==k]))) for k in np.arange(len(numbins))]) 
        running_std_mixture_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(mixture[idx==k]))) for k in np.arange(len(numbins))]) 

        running_std_halfscreen_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(halfscreen[idx==k]))) for k in np.arange(len(numbins))]) 
        running_std_quarterscreen_log  = np.array([np.std(np.log10(np.abs(t['AHa'][idx==k]))-np.log10(np.abs(quarterscreen[idx==k]))) for k in np.arange(len(numbins))]) 


        radialbins = np.linspace(0,12,9)
        idxradial = np.digitize(t1['r'],radialbins)


        popt,pcov = curve_fit(exponential_func,t1['r'],t['AHa'])
        print(popt)
        running_std_radial = np.array([np.std(t['AHa'][idxradial==k]-exponential_func(t1['r'][idxradial==k],*popt)) for k in np.arange(len(radialbins))]) 
        running_std_radial_log = np.array([np.std(np.log10(np.abs(t['AHa'][idxradial==k]))-np.log10(np.abs(t['AHa'][idxradial==k]-exponential_func(t1['r'][idxradial==k],*popt)))) for k in np.arange(len(radialbins))]) 
        

        
        

        


        T = Table([xr,running_std_ir8,running_std_ir12,running_std_ir24,running_std_ir70,numbins,running_std_screen,running_std_halfscreen,running_std_quarterscreen ,running_std_mixture,radialbins,running_std_radial,running_std_ir8_log,running_std_ir12_log,running_std_ir24_log,running_std_ir70_log,running_std_screen_log,running_std_halfscreen_log,running_std_quarterscreen_log,running_std_mixture_log,running_std_radial_log],names = ('Ha/IR','ir8','ir12','ir24','ir70','Gas','screen','halfscreen','quarterscreen','mixture','r','radial_std','logir8','logir12','logir24','logir70','logscreen','loghalfscreen','logquarterscreen','logmixture','logradial_std'))

        T.write('BothGals_binned_STDs.txt',format = 'ascii',overwrite=True)

def exponential_func(x, a, b, c):
    return a*np.exp(-b*x)+c

def makeGasPlots(galaxy):
    if galaxy == 'NGC5194':
        t =  Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1 = Table.read('m51Aha+H.txt',format = 'ascii') 

        numt = Table.read('Screen+MixtureTable5194.txt',format = 'ascii')

        plt.figure()
        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t1['H']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t1['H']),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        plt.plot(np.log10(numt['nr']),numt['screen'],'k-',label = 'Screen')
        plt.plot(np.log10(numt['nr']),numt['quarterscreen'],'k--',label = r'0.206$\cdot$Screen')
        plt.plot(np.log10(numt['nr']),numt['mixture'],'k:',label = 'Mixture')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(N(H)) 500pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.legend(loc = 'upper left',fontsize=16)
        plt.title('NGC5194',fontsize=16)
        plt.tight_layout()

        plt.xlim(20.8,23)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/NGC5194_AHa_H.png')
        plt.close()

        plt.figure()
        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t1['H']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t1['H']),t['AHa'],c = np.log10(t1['h2']/t1['h1']),cmap = 'viridis',alpha = 0.7,zorder=1,label = '')

        cb = plt.colorbar()
        cb.set_label(r'log(H$_2$/HI)',fontsize=16) 
        plt.xlabel(r'log(N(H)) 500pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.title('NGC5194',fontsize=16)
        plt.tight_layout()
        plt.xlim(20.8,23)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/NGC5194_AHa_H_ratio.png')
        plt.close()


        plt.figure()
        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t1['Hhires']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t1['Hhires']),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        plt.plot(np.log10(numt['nr']),numt['screen'],'k-',label = 'Screen')
        plt.plot(np.log10(numt['nr']),numt['quarterscreen'],'k--',label = r'0.206$\cdot$Screen')
        plt.plot(np.log10(numt['nr']),numt['mixture'],'k:',label = 'Mixture')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(N(H)) 90pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.legend(loc = 'upper left',fontsize=16)
        plt.title('NGC5194',fontsize=16)
        plt.tight_layout()

        plt.xlim(20.3,23.1)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/NGC5194_AHa_Hhires.png')
        plt.close()




    elif galaxy == 'NGC6946':
        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  


        numt = Table.read('Screen+MixtureTable6946.txt',format = 'ascii')

        plt.figure()
        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t1['H']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t1['H']),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        plt.plot(np.log10(numt['nr']),numt['screen'],'k-',label = 'Screen')
        plt.plot(np.log10(numt['nr']),numt['quarterscreen'],'k--',label = r'0.169$\cdot$Screen')
        plt.plot(np.log10(numt['nr']),numt['mixture'],'k:',label = 'Mixture')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(N(H)) 500pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.legend(loc = 'upper left',fontsize=16)
        plt.title('NGC6946',fontsize=16)
        plt.tight_layout()

        plt.xlim(20.8,23.3)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/NGC6946_AHa_H.png')
        plt.close()

        plt.figure()
        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t1['H']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t1['H']),t['AHa'],c = np.log10(t1['h2']/t1['h1']),cmap = 'viridis',alpha = 0.7,zorder=1,label = '')

        cb = plt.colorbar()
        cb.set_label(r'log(H$_2$/HI)',fontsize=16) 
        plt.xlabel(r'log(N(H)) 500pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.title('NGC6946',fontsize=16)
        plt.tight_layout()
        plt.xlim(20.8,23.3)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/NGC6946_AHa_H_ratio.png')
        plt.close()


        plt.figure()
        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t1['Hhires']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t1['Hhires']),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        plt.plot(np.log10(numt['nr']),numt['screen'],'k-',label = 'Screen')
        plt.plot(np.log10(numt['nr']),numt['quarterscreen'],'k--',label = r'0.169$\cdot$Screen')
        plt.plot(np.log10(numt['nr']),numt['mixture'],'k:',label = 'Mixture')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(N(H)) 200pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.legend(loc = 'upper left',fontsize=16)
        plt.title('NGC6946',fontsize=16)
        plt.tight_layout()

        plt.xlim(20.8,23.5)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/NGC6946_AHa_Hhires.png')
        plt.close()


    else:
        tm =  Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1m = Table.read('m51Aha+H.txt',format = 'ascii') 
        numtm = Table.read('Screen+MixtureTable5194.txt',format = 'ascii')

        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  
        numt = Table.read('Screen+MixtureTable6946.txt',format = 'ascii')


        xr = np.arange(21,24,.08)

        idxm = np.digitize(np.log10(t1m['H']),xr)
        running_stdm = np.array([np.nanstd(tm['AHa'][idxm==k]) for k in np.arange(len(xr))])  
        running_medm = np.array([np.nanmedian(tm['AHa'][idxm==k]) for k in np.arange(len(xr))])

        idx = np.digitize(np.log10(t1['H']),xr)
        running_std = np.array([np.nanstd(t['AHa'][idx==k]) for k in np.arange(len(xr))])  
        running_med = np.array([np.nanmedian(t['AHa'][idx==k]) for k in np.arange(len(xr))])


        plt.figure()
        plt.fill_between(xr,running_med-running_std,running_med+running_std,color = 'forestgreen',alpha = 0.7,label = 'NGC6946')
        plt.plot(xr,running_med,color = 'forestgreen',lw=3,label ='')

        plt.fill_between(xr,running_medm-running_stdm,running_medm+running_stdm,color = 'orange',alpha = 0.7,label = 'NGC5194')
        plt.plot(xr,running_medm,color = 'orange',lw=3,label ='')

        nr = np.logspace(21,24,len(t['AHa'])) 
        screen,_ = screen,gasrange = makeMixtureModelGas(nr,factor=0.2)

        plt.plot(np.log10(nr),screen,'k--',label = r'0.20$\cdot$Screen')

        plt.xlabel(r'log(N(H)) 500pc [cm$^{-2}$]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.legend(loc = 'upper left',fontsize=16)
        plt.tight_layout()

        plt.xlim(21,23.3)
        plt.ylim(0,5)
        plt.savefig('/Users/kessler.363/Desktop/BothHGals_AHa_H.png')
        plt.close()



def makeLuminosityPlots(galaxy):
    if galaxy == 'NGC5194':
        t =  Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1 = Table.read('m51Aha+H.txt',format = 'ascii') 
        

        plt.figure()

        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t['LHa']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t['LHa']),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(L(H$\alpha$)) [erg/s]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.title('NGC5194',fontsize=16)
        plt.tight_layout()

        plt.xlim(36.1,38.9)
        plt.ylim(-2,7)
        plt.savefig('/Users/kessler.363/Desktop/LHavAHa_5194.png')
        plt.close()


        plt.figure()

        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t['LHa']*10**(t['AHa']/2.5)),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t['LHa']*10**(t['AHa']/2.5)),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(L(H$\alpha$) (corr)) [erg/s]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.title('NGC5194',fontsize=16)
        plt.tight_layout()

        plt.xlim(36.3,40.5)
        plt.ylim(-2,7)
        plt.savefig('/Users/kessler.363/Desktop/LHavAHa_5194_corr.png')
        plt.close()

    elif galaxy == 'NGC6946':
        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  
        

        plt.figure()

        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t['LHa']),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t['LHa']),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(L(H$\alpha$)) [erg/s]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.title('NGC6946',fontsize=16)
        plt.tight_layout()

        plt.xlim(34.8,38.9)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/LHavAHa_6946.png')
        plt.close()


        plt.figure()

        plt.grid(linestyle = ':',zorder=0,color = 'gray') 
        plt.errorbar(np.log10(t['LHa']*10**(t['AHa']/2.5)),t['AHa'],yerr = t['E_AHa'],fmt = 'none',color = 'gray',alpha = 0.5,zorder=0,label = '')
        plt.scatter(np.log10(t['LHa']*10**(t['AHa']/2.5)),t['AHa'],c = t1['r'],cmap = 'plasma',alpha = 0.7,zorder=1,label = '')

        cb = plt.colorbar()
        cb.set_label('r [kpc]',fontsize=16) 
        plt.xlabel(r'log(L(H$\alpha$) (corr)) [erg/s]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.title('NGC6946',fontsize=16)
        plt.tight_layout()

        plt.xlim(36.3,40.5)
        plt.ylim(-2,12)
        plt.savefig('/Users/kessler.363/Desktop/LHavAHa_6946_corr.png')
        plt.close()

def makeGalaxyImages():
    filename = '../../NGC6946/Adam/raw/convolved_mosaic_nostars/arc2.fits'
    plt.figure()
    gc = aplpy.FITSFigure(filename)
    gc.show_colorscale(cmap = 'viridis',vmin = 0,vmax =9e-6) 
    plt.xlabel(r'R.A. (J2000)',fontsize=16)
    plt.ylabel(r'Dec. (J2000)',fontsize=16)
    gc.add_colorbar()
    plt.ylabel(r'I(Pa$\beta$) [erg/s/cm$^2$/sr]',fontsize=16)
    #plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/NGC6946pab.png')
    plt.close()
    
    filename = '../../NGC6946/Adam/raw/ha_aplpyimage.fits'
    plt.figure()
    gc = aplpy.FITSFigure(filename)
    gc.show_colorscale(cmap = 'viridis',vmin = 0,vmax =9e-6) 
    plt.xlabel(r'R.A. (J2000)',fontsize=16)
    plt.ylabel(r'Dec. (J2000)',fontsize=16)
    gc.add_colorbar()
    plt.ylabel(r'I(H$\alpha$) [erg/s/cm$^2$/sr]',fontsize=16)
    #plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/NGC6946ha.png')
    plt.close()

    filename = '../convolvedmosaic/arc2.fits'
    plt.figure()
    gc = aplpy.FITSFigure(filename)
    gc.show_colorscale(cmap = 'viridis',vmin = 0,vmax =9e-6) 
    plt.xlabel(r'R.A. (J2000)',fontsize=16)
    plt.ylabel(r'Dec. (J2000)',fontsize=16)
    gc.add_colorbar()
    plt.ylabel(r'I(Pa$\beta$) [erg/s/cm$^2$/sr]',fontsize=16)
    #plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/NGC5194pab.png')
    plt.close()
    
    filename = '../ha_aplpyimage.fits'
    plt.figure()
    gc = aplpy.FITSFigure(filename)
    gc.show_colorscale(cmap = 'viridis',vmin = 0,vmax =9e-6) 
    plt.xlabel(r'R.A. (J2000)',fontsize=16)
    plt.ylabel(r'Dec. (J2000)',fontsize=16)
    gc.add_colorbar()
    plt.ylabel(r'I(H$\alpha$) [erg/s/cm$^2$/sr]',fontsize=16)
    #plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/NGC5194ha.png')
    plt.close()



def makeApertureImages():
    apr = (2./.3)
    filename = '../convolvedmosaic/arc2.fits'
    t = Table.read('/Users/kessler.363/Desktop/PaBeta/M51/analysis/NGC5194_Aperture_Locs_test.csv',format = 'ascii',delimiter = ',')

    f = fits.open(filename)[0]
    w = wcs.WCS(f.header)
    world = [ [t['ra'][j],t['dec'][j]] for j in np.arange(len(t['ra']))]
    pixcoords = w.wcs_world2pix(world,0)
    aps = CircularAperture(pixcoords,r = apr/2.)

    level = 9.e-07*16.7
    plt.figure()
    gc = aplpy.FITSFigure(filename)
    gc.show_grayscale(vmin = 0,vmax =9e-6)
    gc.show_contour('../ha_2arc_sings.fits',levels = [level],colors = 'yellow') 
    aps.plot(color = 'red',lw=.5)
    plt.xlabel(r'R.A. (J2000)',fontsize=16)
    plt.ylabel(r'Dec. (J2000)',fontsize=16)
    plt.savefig('/Users/kessler.363/Desktop/Threshhold_M51.png')
    plt.close()
    

    filename = '/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/convolved_mosaic_nostars/arc2.fits'
    t = Table.read('/Users/kessler.363/Desktop/PaBeta/NGC6946/analysis/NGC6946_Aperture_Locs_2arc.csv',format = 'ascii',delimiter = ',')
    f = fits.open(filename)[0]
    w = wcs.WCS(f.header)
    world = [ [t['ra'][j],t['dec'][j]] for j in np.arange(len(t['ra']))]
    pixcoords = w.wcs_world2pix(world,0)
    aps = CircularAperture(pixcoords,r = apr/2.)


    sr = 2.11537736210317e-12
    level = 1e-6*sr*17.6
    plt.figure()
    gc = aplpy.FITSFigure(filename)
    gc.show_grayscale(vmin = 0,vmax =9e-6)
    gc.show_contour('/Users/kessler.363/Desktop/PaBeta/NGC6946/analysis/testha2arc.fits',levels = [level],colors = 'yellow') 
    aps.plot(color = 'red',lw=.5)
    plt.xlabel(r'R.A. (J2000)',fontsize=16)
    plt.ylabel(r'Dec. (J2000)',fontsize=16)
    plt.savefig('/Users/kessler.363/Desktop/Threshhold_6946.png')
    plt.close()
    


def make2MASSPlots(before=None):
    
    onorig = sorted(glob('/Users/kessler.363/Desktop/PaBeta/M51/raw/on/rawdir/*.fits')) 
    onshift = sorted(glob('/Users/kessler.363/Desktop/PaBeta/M51/onshift/*.fits')) 
    
    photflamoff = 1.5274129E-20
    photflamon = 4.2779222E-19
    sr =3.866022405704777e-13

    j2mass = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/Adam/working_ngc5194_2mass_j.fits')[0]


    colors=cm.Oranges(np.linspace(0,1,len(onorig)+2))
    plt.figure()
    plt.gca().set_facecolor('gainsboro')
    #plt.gca().set_set_gridlines('')

    for i in np.arange(len(onorig)):
        bounds = np.arange(0.,3.,.1)

        j2massbins = [[] for i in np.arange(len(bounds))]
        onorigbins = [[] for i in np.arange(len(bounds))]
        onshiftbins = [[] for i in np.arange(len(bounds))]

        onorigdata,foot = reproject_interp(fits.open(onorig[i])[0],j2mass.header)
        onshiftdata,foot = reproject_interp(fits.open(onshift[i])[0],j2mass.header)

        onorigdata  = (onorigdata*photflamon* ((1.284e4)**2.) * 3.34e4*1e-6)/sr
        onshiftdata  = (onshiftdata*photflamon* ((1.284e4)**2.) * 3.34e4*1e-6)/sr
        

        for n in np.arange(len(bounds)):
            if n<= (len(bounds) - 2): 
                binmask = np.logical_and(j2mass.data >=bounds[n],j2mass.data<bounds[n+ 1])
                try:
                    j2massbins[n].append(np.nanmedian(j2mass.data[binmask]))
                    onorigbins[n].append(np.nanmedian(onorigdata[binmask]))
                    onshiftbins[n].append(np.nanmedian(onshiftdata[binmask]))
                except:
                    pass
        j2massbins =  np.concatenate([j2massbins])
        onorigbins = np.concatenate([onorigbins])
        onshiftbins = np.concatenate([onshiftbins])
        j2massbins =  np.concatenate(j2massbins)
        onorigbins = np.concatenate(onorigbins)
        onshiftbins = np.concatenate(onshiftbins)
        if before!=None:
            plt.plot(j2massbins,onorigbins,lw = 2,color = colors[i],label = 'Panel '+str(i+1))
        else:
            plt.plot(j2massbins,onshiftbins,lw = 2,color = colors[i],label = 'Panel '+str(i+1))
            
    plt.xlabel('2MASS J Band [MJy/sr]',fontsize=16)
    plt.ylabel('ON [MJy/sr]',fontsize=16)
    if before!=None:
        plt.title('NGC5194, before background offset',fontsize=16)
    else:
        plt.title('NGC5194, after background offset',fontsize=16)


    plt.legend(fontsize=16, ncol=3,facecolor = 'darkgrey',loc = 'upper left')

    plt.grid(linestyle = '-',color = 'white')
    plt.tight_layout()
    plt.ylim(0,2.5)


    if before!=None:
        plt.savefig('/Users/kessler.363/Desktop/2mass_v_ON_before_5194.png')
        plt.close('all')
    else:
        plt.savefig('/Users/kessler.363/Desktop/2mass_v_ON_after_5194.png')
        plt.close('all')


    #NGC6946 now
    onorig = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/ON/drz25arc/*.fits')) 
    onshift = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/onshifttest/arc25/*.fits')) 
    
    photflamoff = 1.5274129E-20
    photflamon = 4.2779222E-19
    sr =3.866022405704777e-13

    j2mass = fits.open('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/working_ngc6946_2mass_j.fits')[0]


    colors=cm.Greens(np.linspace(0,1,len(onorig)+2))
    plt.close('all')
    plt.figure()
    plt.gca().set_facecolor('gainsboro')
    #plt.gca().set_set_gridlines('')

    for i in np.arange(len(onorig)):
        bounds = np.arange(0.,3.,.1)

        j2massbins = [[] for i in np.arange(len(bounds))]
        onorigbins = [[] for i in np.arange(len(bounds))]
        onshiftbins = [[] for i in np.arange(len(bounds))]

        onorigdata,foot = reproject_interp(fits.open(onorig[i])[0],j2mass.header)
        onshiftdata,foot = reproject_interp(fits.open(onshift[i])[0],j2mass.header)

        onorigdata  = (onorigdata*photflamon* ((1.284e4)**2.) * 3.34e4*1e-6)/sr
        onshiftdata  = (onshiftdata*photflamon* ((1.284e4)**2.) * 3.34e4*1e-6)/sr
        

        for n in np.arange(len(bounds)):
            if n<= (len(bounds) - 2): 
                binmask = np.logical_and(j2mass.data >=bounds[n],j2mass.data<bounds[n+ 1])
                try:
                    j2massbins[n].append(np.nanmedian(j2mass.data[binmask]))
                    onorigbins[n].append(np.nanmedian(onorigdata[binmask]))
                    onshiftbins[n].append(np.nanmedian(onshiftdata[binmask]))
                except:
                    pass
        j2massbins =  np.concatenate([j2massbins])
        onorigbins = np.concatenate([onorigbins])
        onshiftbins = np.concatenate([onshiftbins])
        j2massbins =  np.concatenate(j2massbins)
        onorigbins = np.concatenate(onorigbins)
        onshiftbins = np.concatenate(onshiftbins)
        if before!=None:
            plt.plot(j2massbins,onorigbins,lw = 2,color = colors[i],label = 'Panel '+str(i+1))
        else:
            plt.plot(j2massbins,onshiftbins,lw = 2,color = colors[i],label = 'Panel '+str(i+1))
            
    plt.xlabel('2MASS J Band [MJy/sr]',fontsize=16)
    plt.ylabel('ON [MJy/sr]',fontsize=16)
    if before!=None:
        plt.title('NGC6946, before background offset',fontsize=16)
    else:
        plt.title('NGC6946, after background offset',fontsize=16)


    plt.legend(fontsize=16, ncol=3,facecolor = 'darkgrey',loc = 'upper left')

    plt.grid(linestyle = '-',color = 'white')
    plt.tight_layout()
    plt.ylim(0,4.5)


    if before!=None:
        plt.savefig('/Users/kessler.363/Desktop/2mass_v_ON_before_6946.png')
        plt.close('all')
    else:
        plt.savefig('/Users/kessler.363/Desktop/2mass_v_ON_after_6946.png')
        plt.close('all')


def makeRatioPlots():
    onfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/M51/drz1arc/on/*.fits'))
    offfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/M51/drz1arc/off/*.fits'))
    plt.close('all')
    plt.figure()
    plt.gca().set_facecolor('gainsboro')

    colors=cm.Oranges(np.linspace(0,1,len(onfiles)+2))
    
    ha = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ha_2arc_sings.fits')[0]
    
    for i in np.arange(len(onfiles)):
        on = fits.open(onfiles[i])[0]
        ondata = on.data
        offdata,foot = reproject_interp(fits.open(offfiles[i])[0],on.header)
        hadata,foot = reproject_interp(ha,on.header)
        mask = hadata >1.e-5 
        ondata[mask] = np.nan
        offdata[mask] = np.nan
        
        
        ratios = np.arange(0.01,.1,.001)
        
        diffs = []
        for j in ratios:
            d = np.nanmedian(np.abs(ondata-j*offdata))
            diffs.append(d)
            

        plt.plot(ratios,diffs,c = colors[i],lw=2,label = 'Panel '+str(i+1))

    theoretical_ratio = 0.03200043235712699 #from ratio of on/off filter widths
    plt.axvline(x = theoretical_ratio,color = 'k',lw=2,linestyle='--')
    plt.xlabel(r'$\beta$',fontsize=16)
    plt.ylabel(r'median($\mid$ON - $\beta$ $\cdot$ OFF$\mid$) [electron/s/pixel]',fontsize=16)
    plt.title('NGC5194',fontsize=16)
    plt.legend(fontsize=16, ncol=3,facecolor = 'darkgrey',loc = 'upper left')
    plt.grid(linestyle = '-',color = 'white')
    plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/Ratio_find_5194.png')

    #now 6946
    print('---'*5)
    print('Now NGC6946')
    offfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/offshifttest/offset/*fits'))
    onfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/onshifttest/*fits'))

    ha = fits.open('/Users/kessler.363/Desktop/PaBeta/NGC6946/analysis/testha2arc.fits')[0]
    
    plt.close('all')
    plt.figure()
    plt.gca().set_facecolor('gainsboro')

    colors=cm.Greens(np.linspace(0,1,len(onfiles)+2))
    

    for i in np.arange(len(onfiles)):
        on = fits.open(onfiles[i])[0]
        ondata = on.data
        offdata,foot = reproject_interp(fits.open(offfiles[i])[0],on.header)
        hadata,foot = reproject_interp(ha,on.header)
        mask = hadata >2.23075472420634e-17
        ondata[mask] = np.nan
        offdata[mask] = np.nan

        ratios = np.arange(0.01,.1,.001)
        
        diffs = []
        for j in ratios:
            d = np.nanmedian(np.abs(ondata-j*offdata))
            diffs.append(d)

        plt.plot(ratios,diffs,c = colors[i],lw=2,label = 'Panel '+str(i+1))

    theoretical_ratio = 0.03200043235712699 #from ratio of on/off filter widths
    plt.axvline(x = theoretical_ratio,color = 'k',lw=2,linestyle='--')

    plt.xlabel(r'$\beta$',fontsize=16)
    plt.ylabel(r'median($\mid$ON - $\beta$ $\cdot$ OFF$\mid$) [electron/s/pixel]',fontsize=16)
    plt.title('NGC6946',fontsize=16)
    plt.legend(fontsize=16, ncol=3,facecolor = 'darkgrey',loc = 'upper left')
    plt.grid(linestyle = '-',color = 'white')
    plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/Ratio_find_6946.png')




def makeRadialPlots():
        tm =  Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1m = Table.read('m51Aha+H.txt',format = 'ascii') 
        numtm = Table.read('Screen+MixtureTable5194.txt',format = 'ascii')

        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  
        numt = Table.read('Screen+MixtureTable6946.txt',format = 'ascii')


        xr = np.arange(0,11,.25)

        idxm = np.digitize(t1m['r'],xr)
        running_stdm = np.array([np.nanstd(tm['AHa'][idxm==k]) for k in np.arange(len(xr))])  
        running_medm = np.array([np.nanmedian(tm['AHa'][idxm==k]) for k in np.arange(len(xr))])

        idx = np.digitize(t1['r'],xr)
        running_std = np.array([np.nanstd(t['AHa'][idx==k]) for k in np.arange(len(xr))])  
        running_med = np.array([np.nanmedian(t['AHa'][idx==k]) for k in np.arange(len(xr))])

        plt.close('all')
        plt.figure()
        plt.fill_between(xr,running_med-running_std,running_med+running_std,color = 'forestgreen',alpha = 0.7,label = 'NGC6946')
        plt.plot(xr,running_med,color = 'forestgreen',lw=3,label ='')

        plt.fill_between(xr,running_medm-running_stdm,running_medm+running_stdm,color = 'orange',alpha = 0.7,label = 'NGC5194')
        plt.plot(xr,running_medm,color = 'orange',lw=3,label ='')

       
        plt.xlabel(r'r [kpc]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

        plt.legend(loc = 'upper left',fontsize=16)
        plt.tight_layout()
        plt.savefig('/Users/kessler.363/Desktop/Radial_AHa_both.png')
        plt.close()

        plt.figure()
        plt.errorbar(t1['r'],t['AHa'],yerr = t['E_AHa'],color = 'forestgreen',fmt = 'o',alpha = 0.7,zorder=0)
        plt.plot(xr,running_med,color = 'k',lw=3,label ='',zorder=1)

        plt.grid(linestyle = ':',zorder=0)
        plt.xlabel(r'r [kpc]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)
        plt.title('NGC6946',fontsize=16)

        plt.tight_layout()
        plt.savefig('/Users/kessler.363/Desktop/Radial_AHa_6946.png')
        plt.close()

        plt.figure()
        plt.errorbar(t1m['r'],tm['AHa'],yerr = tm['E_AHa'],color = 'orange',fmt = 'o',alpha = 0.7,zorder=0)
        plt.plot(xr,running_medm,color = 'k',lw=3,label ='',zorder=1)

        plt.grid(linestyle = ':',zorder=0)
        plt.xlabel(r'r [kpc]',fontsize=16)
        plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)
        plt.title('NGC5194',fontsize=16)
        plt.tight_layout()
        plt.savefig('/Users/kessler.363/Desktop/Radial_AHa_5194.png')
        plt.close()

def makeHistPlots():
        tm =  Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii') 
        t1m = Table.read('m51Aha+H.txt',format = 'ascii') 

        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        t1 = Table.read('../../NGC6946/analysis/NGC6946Aha+H.txt',format = 'ascii')  

        bins = np.arange(0,11,.2)
        
        plt.close('all')
        plt.figure()
        plt.hist(t['AHa'],bins = bins,color = 'forestgreen',alpha = 0.7,label = 'NGC6946')
        plt.hist(tm['AHa'],bins = bins,color = 'orange',alpha = 0.7,label = 'NGC5194')

        plt.xlabel(r'A(H$\alpha$) [mag]',fontsize=16)
        plt.ylabel('N',fontsize=16)

        plt.legend(loc = 'upper right',fontsize=16)
        plt.tight_layout()
        plt.savefig('/Users/kessler.363/Desktop/AhaHistograms.png')

        plt.close()

        plt.figure()
        plt.hist(t['AHa'],bins = bins,weights = t['LHa']*10**(t['AHa']/2.5),color = 'forestgreen',alpha = 0.7,label = 'NGC6946')
        plt.hist(tm['AHa'],bins = bins,weights = tm['LHa']*10**(tm['AHa']/2.5),color = 'orange',alpha = 0.7,label = 'NGC5194')

        plt.xlabel(r'A(H$\alpha$) [mag]',fontsize=16)
        plt.ylabel(r'$\sum$L(H$\alpha$) (corr) [erg/s]',fontsize=16)

        plt.legend(loc = 'upper right',fontsize=16)
        plt.tight_layout()
        plt.savefig('/Users/kessler.363/Desktop/AHaHistbyFluxLinear.png')
        plt.close()

def makeOnvOffPlots():
    onfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/M51/drz1arc/on/*.fits'))
    offfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/M51/drz1arc/off/*.fits'))
    plt.close('all')
    plt.figure()
    plt.gca().set_facecolor('gainsboro')
    ha = fits.open('/Users/kessler.363/Desktop/PaBeta/M51/ha_2arc_sings.fits')[0]
    
    colors=cm.Oranges(np.linspace(0,1,len(onfiles)+2))
    
    for i in np.arange(len(onfiles)):
        bounds = np.arange(-.2,2.,.2)

        onbins = [[] for i in np.arange(len(bounds))]
        offbins = [[] for i in np.arange(len(bounds))]

        on = fits.open(onfiles[i])[0]
        ondata = on.data
        offdata,foot = reproject_interp(fits.open(offfiles[i])[0],on.header) 
        hadata, foot = reproject_interp(ha,on.header)
        mask = hadata>1.e-5
        ondata[mask] = np.nan
        offdata[mask] = np.nan

        for n in np.arange(len(bounds)):
            if n<= (len(bounds) - 2): 
                binmask = np.logical_and(ondata >=bounds[n],ondata<bounds[n+ 1])
                try:
                    onbins[n].append(np.nanmedian(ondata[binmask]))
                    offbins[n].append(np.nanmedian(offdata[binmask]))
                except:
                    pass
        onbins = np.concatenate([onbins])
        offbins = np.concatenate([offbins])
        onbins = np.concatenate(onbins)
        offbins = np.concatenate(offbins)
        plt.plot(onbins,offbins,lw = 2,color = colors[i],label = 'Panel '+str(i+1))

    plt.xlabel(r'ON [electron/s/pixel]',fontsize=16)
    plt.ylabel(r'OFF [electron/s/pixel]',fontsize=16)
    plt.title('NGC5194, OFF background corrected',fontsize=16)
    plt.legend(fontsize=16, ncol=3,facecolor = 'darkgrey',loc = 'upper left')
    plt.grid(linestyle = '-',color = 'white')
    plt.xlim(xmin = -0.05)
    plt.ylim(ymin = -0.5)
    plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/OnVOff5194.png')
    #now 6946
    print('---'*5)
    print('Now NGC6946')
    offfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/OFF/drz1arc/*.fits'))
    onfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/ON/drz1arc/*.fits'))

    #offfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/offshifttest/offset/*fits'))
    #onfiles = sorted(glob('/Users/kessler.363/Desktop/PaBeta/NGC6946/Adam/raw/onshifttest/*fits'))
    ha = fits.open('/Users/kessler.363/Desktop/PaBeta/NGC6946/analysis/testha2arc.fits')[0]
    
    
    plt.close('all')
    plt.figure()
    plt.gca().set_facecolor('gainsboro')

    colors=cm.Greens(np.linspace(0,1,len(onfiles)+2))

    for i in np.arange(len(onfiles)):
        bounds = np.arange(-.2,2.,.2)

        onbins = [[] for i in np.arange(len(bounds))]
        offbins = [[] for i in np.arange(len(bounds))]

        on = fits.open(onfiles[i])[0]
        ondata = on.data
        offdata,foot = reproject_interp(fits.open(offfiles[i])[0],on.header) 
        hadata,foot = reproject_interp(ha,on.header)
        mask = hadata >2.23075472420634e-17
        ondata[mask] = np.nan
        offdata[mask] = np.nan


        for n in np.arange(len(bounds)):
            if n<= (len(bounds) - 2): 
                binmask = np.logical_and(ondata >=bounds[n],ondata<bounds[n+ 1])
                try:
                    onbins[n].append(np.nanmedian(ondata[binmask]))
                    offbins[n].append(np.nanmedian(offdata[binmask]))
                except:
                    pass
        onbins = np.concatenate([onbins])
        offbins = np.concatenate([offbins])
        onbins = np.concatenate(onbins)
        offbins = np.concatenate(offbins)
        plt.plot(onbins,offbins,lw = 2,color = colors[i],label = 'Panel '+str(i+1))

    plt.xlabel(r'ON [electron/s/pixel]',fontsize=16)
    plt.ylabel(r'OFF [electron/s/pixel]',fontsize=16)
    plt.title('NGC6946, OFF background corrected',fontsize=16)
    plt.legend(fontsize=16, ncol=3,facecolor = 'darkgrey',loc = 'upper left')
    plt.grid(linestyle = '-',color = 'white')
    plt.xlim(xmin = -0.05)
    plt.ylim(ymin = -0.5)
    
    plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/OnVOff6946.png')

    

def makeAHaApertureImages():

    apr = (2./.3)
    filename = '../convolvedmosaic/arc2.fits'
    pab,ha,r,ir24,ir70,ir8,ir12,h1,h2,h2hires,aps,ans,aha,hacorr,extimg,merrs,perrs,ra,dec,mask,t = aperturesnew(filename,apr) 
 
    x,y = zip(*aps.positions)
    x,y = np.asarray(x)[mask],np.asarray(y)[mask]
    plt.close('all')
    gc = aplpy.FITSFigure(filename)
    #gc.show_grayscale()
    plt.gca().set_facecolor('gainsboro')    
    plt.scatter(np.asarray(x),np.asarray(y),c = aha,cmap = 'rainbow',s=10)

    plt.xlabel(r'R.A. (J2000)',fontsize=20)
    plt.ylabel(r'Dec. (J2000)',fontsize=20)
    cb = plt.colorbar()
    cb.set_label(r'A(H$\alpha$) [mag]',fontsize=20)
    plt.tight_layout()
    plt.show()
   # plt.savefig('/Users/kessler.363/Desktop/AHa_spatial_5194.png')
    #plt.close()
    

def makeCompareOtherMeasurePlots():
    Aha,Ahabalmer,Ahaken,Ahafreefree,r,rken,balmererrs,freefreeerrs,ahaerrs,kenerrs= blancbalmerapertures()

    colors=cm.Dark2(np.linspace(0,1,4))

    plt.close('all')
    plt.figure()
    plt.errorbar(r,Aha,yerr = ahaerrs,fmt = 'o',c = colors[0],alpha = 0.8,label = 'This Work')
    plt.errorbar(r,Ahabalmer,yerr = balmererrs,fmt = 'o',c = colors[1],alpha = 0.8,label = 'Blanc09')
    plt.errorbar(rken,Ahaken,yerr = kenerrs,fmt = 'o',c = colors[2],alpha = 0.8,label = 'Kennicutt07')
    plt.errorbar(r,Ahafreefree,yerr = freefreeerrs,fmt = 'o',c = colors[3],alpha = 0.8,label = 'Querejeta19')

    plt.xlabel('r [kpc]',fontsize=16)
    plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

    plt.grid(linestyle = ':',zorder=0)
    plt.legend(fontsize=16,loc = 'lower left',ncol =2)
    plt.ylim(ymin=-2)
    
    plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/CompareToLiterature_5194_0int.png')
    
    
def makeAHafreefreeplot():
    pab_phot_table,ha_phot_table,rgrid_phot_table,freefree_phot_table,apertures,annulus_apertures,sfrfreefree,sfrha,Aha,t2,Ahafreefree,Ahafreefreep,Ahafreefreem = freefreeapertures()
    Ahafreefree*=1.5
    Ahafreefreep*=1.5
    Ahafreefreem*=1.5
    
    r = rgrid_phot_table['r_kpc']
    t = Table.read('NGC5194_Aperture_table_3arc.txt',format = 'ascii')

    colors=cm.viridis(np.linspace(0,1,2))

    plt.close('all')
    plt.figure()
    plt.errorbar(r,t['AHa'],yerr = t['E_AHa'],color = colors[0],alpha = 0.8,fmt = 'o',label = r'Pa$\beta$')
    plt.errorbar(r,Ahafreefree,yerr = [Ahafreefree-Ahafreefreem,Ahafreefreep-Ahafreefree],color = colors[1],alpha = 0.8,fmt = 'o',label = '33GHz')
    plt.xlabel('r [kpc]',fontsize=16)
    plt.ylabel(r'A(H$\alpha$) [mag]',fontsize=16)

    plt.grid(linestyle = ':',zorder=0)
    plt.legend(fontsize=16)
    
    plt.tight_layout()
    plt.savefig('/Users/kessler.363/Desktop/AHa_freefreevPab_5194.png')
    plt.close()

def makeCombTable(galaxy):

    if galaxy == 'NGC6946':
        t = Table.read('../../NGC6946/analysis/NGC6946_Aperture_Table_2arc.txt',format = 'ascii')
        irt = Table.read('../../NGC6946/analysis/IRTable6946.txt',format = 'ascii')
    else:
        t = Table.read('NGC5194_Aperture_table_2arc.txt',format = 'ascii')  
        irt = Table.read('IRTable5194.txt',format = 'ascii')

    irt['ir8']*=const.c.to(u.micron/u.s).value/8                          

    irt['ir12']*=const.c.to(u.micron/u.s).value/12                        

    irt['ir24']*=const.c.to(u.micron/u.s).value/24                        

    irt['ir70']*=const.c.to(u.micron/u.s).value/70

    if galaxy == 'NGC6946':
        irt['ir100']*=const.c.to(u.micron/u.s).value/100 

    for i in np.arange(len(irt.colnames)): 
        if i<=1: 
            irt[irt.colnames[i]] = np.round(irt[irt.colnames[i]],decimals=4) 
        else: 
            irt[irt.colnames[i]].info.format = '1.4e' 

    combt = join(t,irt)                                                   

    combt['RA']*=u.degree                                                 
    combt['Dec']*=u.degree                                                
    combt['LHa']*=(u.erg/u.s)                                             
    combt['E_LHa']*=(u.erg/u.s)                                           
    combt['LPab']*=(u.erg/u.s)                                            
    combt['E_Pab']*=(u.erg/u.s)                                           

    combt['AHa']*=(u.mag)                                                 
    combt['E_AHa']*=(u.mag)                                               
    combt.rename_column('ir8','Ha/8um')                                            
    combt.rename_column('ir12','Ha/12um')                                            
    combt.rename_column('ir24','Ha/24um')                                            
    combt.rename_column('ir70','Ha/70um')                                            

    if galaxy == 'NGC6946':

        combt.rename_column('ir100','Ha/100um')                                            


    for i in combt.colnames: 
        if 'Ha/' in i: 
            combt[i].info.format = '1.4e' 
            
    combt.sort('ID')
    #combt.remove_column('ID')

    if galaxy == 'NGC6946':
        combt.write('NGC6946_Full_Aperture_table_2arc.ecsv',format = 'ascii.ecsv', overwrite=True)
        combt.write('NGC6946_Full_Aperture_table_2arc_latex.txt',format = 'latex',overwrite=True)
    else:
        combt.write('NGC5194_Full_Aperture_table_2arc.ecsv',format = 'ascii.ecsv', overwrite=True)
        combt.remove_columns(['Ha/8um','Ha/12um','Ha/24um','Ha/70um'])  
        combt.write('NGC5194_Full_Aperture_table_2arc_latex.txt',format = 'latex',overwrite=True)
        

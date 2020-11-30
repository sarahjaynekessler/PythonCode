import pyneb as pn
import numpy as np
from dust_extinction.parameter_averages import F99,CCM89
import astropy.units as u


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
    Pabeta = H1.getEmissivity(tem = 1e4, den = 1e3, lev_i = 5, lev_j = 3)
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


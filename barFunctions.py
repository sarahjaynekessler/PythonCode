import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import seaborn as sns

sns.set_style("darkgrid")
sns.set(font_scale=1.5)


def makeBinnedProfiles(df,var,sfrcols,xax,xlabel,normalized,savefig=None,notRadial=None,
    var2 = None,var3 = None):

    testdf = df.copy()
    latexnamedict = {'T':'T','DELTAMS':'$\Delta\ SFMS [dex]$','Mh/M*':'$M_{h}/M_{*}$',
            'Qbbarhcor':'$Q_{Bar,hcorr}$','Ropt_kpc':'$r_{opt}}$ [kpc]','ThirdDist_MPC':'D$_{3rd\ Gal}$ [Mpc]','LOGMASS':r'$\log\ {M_{*}}$','Dist':'Dist [Mpc]'}

    if notRadial!=None:

        binpercents = [0,.25,.5,.75,1.]

        grouped = testdf.groupby(pd.cut(testdf[var],testdf.quantile(binpercents)[var].values))

        percentdf = grouped.quantile(binpercents)

        labels = [r'0-25% '+latexnamedict[var],r'25-50% '+latexnamedict[var],
                  r'50-75% '+latexnamedict[var],r'75-100% '+latexnamedict[var]]

        plt.figure(dpi=100)
        colors = cm.Dark2(np.linspace(0, 1, len(grouped)))
        ind = 0
        
        for i in np.arange(len(grouped)):
            plt.scatter(percentdf[var2].iloc[ind:ind+6],percentdf[var3].iloc[ind:ind+6],label = labels[i],color = colors[i],lw=2,alpha=0.9)
            #plt.plot(percentdf[var2].iloc[ind:ind+6],percentdf[var3].iloc[ind:ind+6],label = labels[i],color = colors[i],lw=2,alpha=0.9)
            ind+=6
            
        plt.legend(  loc='lower right')
        plt.ylabel(latexnamedict[var2])
        plt.xlabel(latexnamedict[var3])
        plt.ylim(ymin=0)
        plt.title('Normalized First')
        plt.tight_layout()
        if savefig!=None:
            plt.savefig(savefig,bbox_inches='tight')
    else:
        if normalized.lower()=='first':
            #normalize first by R25, bin, take medians
            

            cols = sfrcols.copy()
            cols.append(var)

            for num in cols[:-1]:
                testdf[num] = testdf[num].copy()/testdf['SFRFUVW3_R25'].copy()
            testdf = testdf[cols]
            testdf = testdf.dropna()
            

            bins = [np.percentile(testdf[var],0),np.percentile(testdf[var],25),np.percentile(testdf[var],50),
                    np.percentile(testdf[var],75),np.percentile(testdf[var],100)]

            grouped = testdf.groupby(pd.cut(testdf[var], bins))
            meds = grouped.median()
            stds = grouped.std()
            
            labels = [r'0-25% '+latexnamedict[var],r'25-50% '+latexnamedict[var],
                      r'50-75% '+latexnamedict[var],r'75-100% '+latexnamedict[var]]

            yax = meds[sfrcols]
            

            colors = cm.Dark2(np.linspace(0, 1, len(yax)))
            plt.figure(dpi=100)
            
            
            for i in np.arange(len(yax)):
                plt.plot(xax,yax.iloc[i],label = labels[i],color = colors[i],lw=2,alpha=0.9)
                plt.plot(xax,yax.iloc[i],'o',color = colors[i])

            plt.legend(loc='lower right')

            plt.ylabel(r'SFR ['+xlabel+'] / SFR$_{R25}$')
            plt.xlabel(xlabel)

            plt.ylim(ymin=0)
            plt.title('Normalized First')
            plt.tight_layout()
            if savefig!=None:
                plt.savefig(savefig,bbox_inches='tight')
        elif normalized=='last':
            
            cols = sfrcols.copy()
            cols.append(var)
            testdf = testdf[cols]
            testdf = testdf.dropna()


            bins = [np.percentile(testdf[var],0),np.percentile(testdf[var],25),np.percentile(testdf[var],50),
                    np.percentile(testdf[var],75),np.percentile(testdf[var],100)]

            grouped = testdf.groupby(pd.cut(testdf[var], bins))
            meds = grouped.median()
            stds = grouped.std()
            
            for col in meds.columns[:-1]:
                meds[col]/=meds['SFRFUVW3_R25']

            labels = [r'0-25% '+latexnamedict[var],r'25-50% '+latexnamedict[var],
                      r'50-75% '+latexnamedict[var],r'75-100% '+latexnamedict[var]]
            
            yax = meds[sfrcols]

            #xax = [0.15,0.2,0.25,0.3,0.5,1]

                
            colors = cm.Dark2(np.linspace(0, 1, len(yax)))
            plt.figure(dpi=100)
            for i in  np.arange(len(yax)):
                plt.plot(xax,yax.iloc[i],label = labels[i],color = colors[i],lw=2,alpha=0.9)
                plt.plot(xax,yax.iloc[i],'o',color = colors[i])

            plt.legend(  loc='lower right')
            plt.xlabel(xlabel)

            plt.ylabel(r'SFR$_{r/R25}$ / SFR$_{R25}$')
            plt.ylim(ymin=0)
            plt.title('Normalized Last')
            plt.tight_layout()
            if savefig!=None:
                plt.savefig(savefig,bbox_inches='tight')
                
        else:
            cols = sfrcols.copy()
            cols.append(var)
          
            testdf = testdf[cols]
            testdf = testdf.dropna()

            bins = [np.percentile(testdf[var],0),np.percentile(testdf[var],25),np.percentile(testdf[var],50),
                    np.percentile(testdf[var],75),np.percentile(testdf[var],100)]

            grouped = testdf.groupby(pd.cut(testdf[var], bins))
            meds = grouped.median()
            stds = grouped.std()

            labels = [r'0-25% '+latexnamedict[var],r'25-50% '+latexnamedict[var],
                      r'50-75% '+latexnamedict[var],r'75-100% '+latexnamedict[var]]
            
            yax = meds[sfrcols]
            #xax = [0.15,0.2,0.25,0.3,0.5,1]

                
            colors = cm.Dark2(np.linspace(0, 1, len(yax)))
            plt.figure(dpi=100)
            for i in  np.arange(len(yax)):
                plt.plot(xax,yax.iloc[i],label = labels[i],color = colors[i],lw=2,alpha=0.9)
                plt.plot(xax,yax.iloc[i],'o',color = colors[i])

            plt.legend(  loc='lower right')
            plt.ylabel(r'SFR$_{r/R25}$')
            plt.xlabel(xlabel)
            plt.ylim(ymin=0)
            plt.title('Not Normalized')
            plt.tight_layout()
            if savefig!=None:
                plt.savefig(savefig,bbox_inches='tight')
                

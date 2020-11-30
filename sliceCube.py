from astropy.io import fits

#takes a filename and dimension of a cube 
def sliceCube(filename):
        hdulist = fits.open(filename)[0]
        print(hdulist.data.shape)
        dim = len(hdulist.data.shape)
        list = ['CTYPE','CRVAL','CDELT','CRPIX','CROTA'] 

        if dim == 4:
                hdulist.data = hdulist.data[0,0,:,:]
                for i in list:
                        hdulist.header.remove(i+str(3))
                        hdulist.header.remove(i+str(4))
                print(hdulist.data.shape)

        elif dim == 3:
                hdulist.data = hdulist.data[0,:,:]
                for i in list:
                        hdulist.header.remove(i+str(3))
                print(hdulist.data.shape)
        else:
                for i in list:
                    try:
                        hdulist.header.remove(i+str(3))
                    except:
                        pass
                    try:
                        hdulist.header.remove(i+str(4))
                    except:
                        pass



        return(hdulist)



        

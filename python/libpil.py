"""
    An interface (high-level) to Python Image Library.

    @date: March 1 , 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
"""

import pdb
import matplotlib.pyplot as plt
import matplotlib.mpl as mpl

import array
try:
    import Image
except ImportError:
    try:
        import PIL.Image as Image
    except ImportError:
        print "You haven't installed PIL, have you?"

try:
    import numpy
except ImportError:
    print "You haven't installed numpy, have you?"


class ConversionException(Exception):
    pass

def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    if len(arr.shape) == 3:
        mode = 'RGBA'
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
        if len(arr[0]) == 3:
            arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)
    elif len(arr.shape) == 2:
        mode = 'L'
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)
    else:
        raise ConversionException("I don't know how to convert this array.")

def concat_imgs(img1, img2, *args, **argd):
    ori = argd.get('ori', 'v')
    imgs = [img1, img2]
    if args:
        imgs.extend(args)
    if ori == 'h':
        w = sum([i.size[0] for i in imgs])
        h = max([i.size[1] for i in imgs])
    else:
        w = max([i.size[0] for i in imgs])
        h = sum([i.size[1] for i in imgs])

    resultimg = Image.new("RGBA", (w, h))
    x = 0
    for im in imgs:
        offset = (x, 0) if ori=='h' else (0, x)
        resultimg.paste(im, offset)
        incr = im.size[0] if ori=='h' else im.size[1]
        x += incr
    return resultimg

def split3array(arr):
    return arr[:,:,0], arr[:,:,1], arr[:,:,2]

def merge3array(x, y, z):
    arr = numpy.c_[x.flatten(), y.flatten()]
    arr = numpy.c_[arr, z.flatten()]
    return arr.reshape(x.shape[1], x.shape[0], 3)

def hist(grayimg, bins=20, rang=(0,255), plot=False):
    hi, edges =  numpy.histogram(grayimg, bins, rang, normed=False)
    hi = numpy.float32(hi)/sum(hi)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #bincenters = 0.5*(edges[1:]+edges[:-1])
        ax.bar(edges[:-1], hi, edges[1]-edges[0], facecolor='green', alpha=0.75)
        #h2, edges2, patches = ax.hist(grayimg.flatten(), 
        #                bins, rang, normed=True, facecolor='green', alpha=0.75)
        #assert all(edges2 == edges)
        plt.show()
    
    return hi, edges

def concathist(rgbarr):
    r, g, b = split3array(rgbarr)
    hr, edges = hist(r)
    hg = hist(g)[0]
    hb = hist(b)[0]
    return hr, hg, hb

def rgbarr2gray(rgbarr):
    r,g,b = split3array(rgbarr)
    return 0.3*r+0.59*g+0.11*b

import colorsys

def rgbarr2hsvarr(rgbarr):
    r,g,b = split3array(rgbarr)
    if r.dtype.type == numpy.uint8:
        r = numpy.float32(r)
        g = numpy.float32(g)
        b = numpy.float32(b)
        r /= 255.0
        g /= 255.0
        b /= 255.0
    vfunc = numpy.vectorize(colorsys.rgb_to_hsv)
    return vfunc(r,g,b)

def lapacian(grayarr):
    try:
        from scipy.stsci.convolve import convolve2d
    except ImportError:
        print "You probabably didn't install python scipy 0.7.0 correctly."
    kernel = -1*numpy.ones((3,3))
    kernel[1,1] = 8.0
    return convolve2d(grayarr, kernel, fft=True)

import profile

@profile.timing
def rgbhist(rgbarr, rbins=4, gbins=8, bbins=4, blackbin=[]):
    cdef int i, j, w, h
    cdef int r, g, b
    cdef int rbase, gbase, bbase
    cdef int uint8, channel
    uint8 = 256
    rbase = uint8 / rbins
    gbase = uint8 / gbins
    bbase = uint8 / bbins
    w,h,channel = rgbarr.shape
    hist = numpy.zeros((rbins, gbins, bbins))
    for i in xrange(w):
        for j in xrange(h):
            r,g,b = rgbarr[i,j,:]
            hist[r / rbase,
                 g / gbase,
                 b / bbase] += 1
    for bb in blackbin:
        hist[bb] = 0
    hist = hist/sum(hist.flatten())
    return hist.flatten()

def laphist(grayarr, bins=400):
    lap = lapacian(grayarr)
    return hist(lap, bins, rang=(-256*8, 256*8))[0]

@profile.timing
def onenorm_inverse(h1, h2):
    dist = sum( numpy.abs(h1 - h2) ) / 2 # assume h1, h2 are L1-normalized
    return 1 - dist

def hist_intersection(h1, h2):
    sim = sum( numpy.c_[h1, h2].min(1) )
    return sim

def plothists(h1, h2, h3, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(edges[:-1], h1, edges[1]-edges[0], facecolor='red', alpha=0.35)
    ax.bar(edges[:-1], h2, edges[1]-edges[0], facecolor='green', alpha=0.35)
    ax.bar(edges[:-1], h3, edges[1]-edges[0], facecolor='blue', alpha=0.35)
    plt.show()


from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def scatterplot(xs, ys, imgs):
    n_imgs = len(imgs)
    fig = plt.figure(figsize=(8,5),dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(xs.min()-0.1*xs.max(), 1.1*xs.max())
    ax.set_ylim(ys.min()-0.1*ys.max(), 1.1*ys.max())
    for i in xrange(n_imgs):
        imagebox = OffsetImage(imgs[i], zoom=0.5)
        ab = AnnotationBbox(imagebox, (xs[i], ys[i]),
                        xybox=(0,0),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0)
        ax.add_artist(ab)

    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.draw()
    axbar = fig.add_axes([0.18, 0.05, 0.68, 0.05])
    cb1 = mpl.colorbar.ColorbarBase(axbar, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    fig.savefig('scatterplot.png',dpi=200)


def loadImage(filename):
    return Image.open(filename)


def main():
    #img1 = loadImage('i01.jpg')
    img1 = loadImage('../../Images/i06.jpg')
    
    arr = PIL2array(img1)
    hi, edges = laphist(arr)
    plt.imshow(lapacian(rgbarr2gray(arr)))

    
#img2 = array2PIL(arr, img.size)
    #img2.save('test.jpg')

if __name__ == '__main__':
    main()

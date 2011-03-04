"""
    An interface (high-level) to Python Image Library.

    @date: March 1 , 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
"""

import pdb
import matplotlib.pyplot as plt

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

def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[0], img.size[1], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

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
        #ax.bar(edges[:-1], hi, edges[1]-edges[0], facecolor='green', alpha=0.75)
        #h2, edges2, patches = ax.hist(grayimg.flatten(), 
        #                bins, rang, normed=True, facecolor='green', alpha=0.75)
        #assert all(edges2 == edges)
        #plt.show()
    
    return hi, edges

def concathist(rgbarr):
    r, g, b = split3array(rgbarr)
    hr, edges = hist(r)
    hg = hist(g)[0]
    hb = hist(b)[0]
    #plothists(hr, hg, hb, edges)
    return hr, hg, hb

import profile

@profile.timing
def rgbhist(rgbarr, rbins=4, gbins=8, bbins=4, blackbin=[(0,0,0)]):
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

def loadImage(filename):
    return Image.open(filename)


def main():
    img1 = loadImage('i01.jpg')
    img2 = loadImage('../../Images/i06.jpg')
    
    arr = PIL2array(img1)
    rgbhist(arr)
    return
    hi1 = concathist(arr)
    arr = PIL2array(img2)
    hi2 = concathist(arr)
    print hist_intersection(hi1, hi2)
#img2 = array2PIL(arr, img.size)
    #img2.save('test.jpg')

if __name__ == '__main__':
    main()

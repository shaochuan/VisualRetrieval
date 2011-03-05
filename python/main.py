import os
import sys
import glob
import numpy
import libpil
import pdb

DEFAULT_IMAGE_ROOT_FOLDER = r'../../Images'

def get_image_folder():
    if '-i' in sys.argv:
        idx = sys.argv.index('-i')
        return sys.argv[idx+1]
    return DEFAULT_IMAGE_ROOT_FOLDER

import profile

@profile.timing
def colorkernel(imgs, 
        metric=libpil.hist_intersection,
        cachedfn='color_kernel.pkl'):
    if os.path.exists(cachedfn):
        return numpy.load(cachedfn)
    n_imgs = len(imgs)

    # calculation of kernel
    kernel = numpy.zeros((n_imgs, n_imgs))
    for i in xrange(n_imgs):
        arr1 = libpil.PIL2array(imgs[i])
        for j in xrange(i):
            arr2 = libpil.PIL2array(imgs[j])
            kernel[i,j] = metric(libpil.rgbhist(arr1),
                                 libpil.rgbhist(arr2))
    kernel = kernel + numpy.transpose(kernel)
    print kernel
    return kernel

@profile.timing
def texturekernel(imgs, 
        metric=libpil.onenorm_inverse,
        cachedfn='texture_kernel.pkl'):
    if os.path.exists(cachedfn):
        return numpy.load(cachedfn)
    n_imgs = len(imgs)
    # calculation of kernel
    kernel = numpy.zeros((n_imgs, n_imgs))
    for i in xrange(n_imgs):
        arr1 = libpil.PIL2array(imgs[i])
        gray1 = libpil.rgbarr2gray(arr1)
        for j in xrange(i):
            arr2 = libpil.PIL2array(imgs[j])
            gray2 = libpil.rgbarr2gray(arr2)
            kernel[i,j] = metric(libpil.laphist(gray1),
                                 libpil.laphist(gray2))
    kernel = kernel + numpy.transpose(kernel)
    print kernel
    return kernel


def output_compare_imgs(imgs, kernel, outfilename='out.png'):
    n_imgs = len(imgs)
    # most similar indices
    most_similar = kernel.argmax(1)
    p,q = (kernel.argmax() % n_imgs, kernel.argmax() / n_imgs)
    for i in xrange(0, len(imgs)):
        kernel[i,i] = 1
    # least similar indices
    least_similar = kernel.argmin(1)
    s,t = (kernel.argmin() % n_imgs, kernel.argmin() / n_imgs)
    for i in xrange(0, len(imgs)):
        kernel[i,i] = 0
    
    catimgs = []
    for i, ML in enumerate(zip(most_similar, least_similar)):
        M,L = ML
        imgo = imgs[i]
        imgm = imgs[M]
        imgl = imgs[L]
        catimgs.append( libpil.concat_imgs(imgo, imgm, imgl, ori='h') )
        #catimg.save("%d.png" % (i,))
    canvas = libpil.concat_imgs(*catimgs)
    canvas.save(outfilename)

    mostalike_imgpair = libpil.concat_imgs(imgs[p], imgs[q], ori='h')
    leastalike_imgpair = libpil.concat_imgs(imgs[s], imgs[t], ori='h')
    mostalike_imgpair.save(os.path.splitext(outfilename)[0] + '_mostalike.png')
    leastalike_imgpair.save(os.path.splitext(outfilename)[0] + '_leastalike.png')

def pdist(kernel):
    import generator
    m,n = kernel.shape
    assert m==n
    dst = numpy.zeros((m*(m-1)/2,1))
    k = 0
    for i, j in generator.combinations(xrange(m),2):
        dst[k] = 1-kernel[i,j]
        k += 1
    return dst.flatten()

def output_dendrogram(imgs, kernel, method='complete', dend_fn='_dendrogram.png'):
    dst = pdist(kernel)
    links = linkage(dst, method=method)
    tmp_dend_fn = method + '_' + dend_fn
    axis = dendrogram(links, 
               orientation='left',
               figsize=(7,12),
               outfilename=tmp_dend_fn)[1]
    figimg = libpil.loadImage(tmp_dend_fn)
    labels = [label._text for label in axis.get_yticklabels()]
    labels = map(int, labels)
    labels.reverse()
    for i, ind in enumerate(labels):
        imgs[ind].thumbnail((30,30))
        offset = i*(imgs[ind].size[1]+4) + 120
        figimg.paste(imgs[ind], (52,offset))
    figimg.save('fig_'+tmp_dend_fn)


from hcluster import linkage, dendrogram
def main():
    image_folder = get_image_folder()
    imgfiles = glob.glob(os.path.join(image_folder, "*.jpg"))
    imgs = map(libpil.loadImage, imgfiles)
    if 'color' in sys.argv:
        color_kernel = colorkernel(imgs, metric=libpil.onenorm_inverse)
        output_compare_imgs(imgs, color_kernel, outfilename='color.png')
        color_kernel.dump('color_kernel.pkl')
    if 'texture' in sys.argv:
        texture_kernel = texturekernel(imgs)
        output_compare_imgs(imgs, texture_kernel, outfilename='texture.png')
        texture_kernel.dump('texture_kernel.pkl')
    

    if 'color' in sys.argv and 'texture' in sys.argv:
        # 1-norm kernel normalization
        color_kernel /= max(color_kernel.sum(1))
        texture_kernel /= max(texture_kernel.sum(1))
        for s in xrange(0, 11, 2):
            r = s * 0.1
            kernel = texture_kernel * r + (1-r) * color_kernel
            kernel /= kernel.max()
            output_compare_imgs(imgs,
                    kernel, outfilename='r%.1f.png' % (r,))
            for method in ('single', 'complete'):
                output_dendrogram(imgs, 
                                  kernel, 
                                  method=method,
                                  dend_fn='r%.1f_dendrogram.png' % (r,))

    if 'scatter' in sys.argv:
        n_imgs = len(imgs)
        dominant_colors = numpy.zeros((n_imgs,))
        for i in xrange(n_imgs):
            hue = libpil.rgbarr2hsvarr( libpil.PIL2array(imgs[i]) )[0]
            histhue,edges = libpil.hist(hue, bins=40, rang=(0,1))
            bincenters = 0.5*(edges[1:]+edges[:-1])
            print 'dominant hue:', bincenters[ histhue.argmax() ]
            dominant_colors[i] = bincenters[ histhue.argmax() ]
        total_textures = numpy.zeros((n_imgs,))
        for i in xrange(n_imgs):
            lapimg = libpil.lapacian(
                     libpil.rgbarr2gray(
                     libpil.PIL2array(
                         imgs[i] )))
            lapimg = lapimg.flatten()
            meadtext = numpy.median(numpy.abs(lapimg))
            print 'meandian texture:', meadtext
            total_textures[i] = meadtext
        
        libpil.scatterplot(dominant_colors, total_textures, imgs)


if __name__ == '__main__':
    main()

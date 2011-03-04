import os
import sys
import glob
import numpy
import libpil
import pdb

DEFAULT_IMAGE_ROOT_FOLDER = r'../../Images'

def get_image_folder():
    if len(sys.argv) > 2:
        idx = sys.argv.index('-i')
        return sys.argv[idx+1]
    return DEFAULT_IMAGE_ROOT_FOLDER

import profile

@profile.timing
def calkernel(imgs, metric=libpil.hist_intersection):
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

def output_compare_imgs(imgs, kernel, outfilename='out.png'):
    # most similar indices
    most_similar = kernel.argmax(1)
    for i in xrange(0, len(imgs)):
        kernel[i,i] = 1
    # least similar indices
    least_similar = kernel.argmin(1)
    
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


def main():
    image_folder = get_image_folder()
    imgfiles = glob.glob(os.path.join(image_folder, "*.jpg"))
    imgs = map(libpil.loadImage, imgfiles)
    kernel = calkernel(imgs)
    output_compare_imgs(imgs, kernel, outfilename='hist_insection.png')
    kernel = calkernel(imgs, metric=libpil.onenorm_inverse)
    output_compare_imgs(imgs, kernel, outfilename='onenorm.png')


if __name__ == '__main__':
    main()

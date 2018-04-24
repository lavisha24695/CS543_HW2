import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from scipy import ndimage
import scipy.misc
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
levels = 13
#use levels 13
s_initial = 2
s = s_initial
threshold = 0.02
def show_all_circles(iage, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(iage, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
	    circ = Circle((x, y), r, color=color, fill=False, linewidth = 2)
	    #print(x,y,r)
	    ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()

def myfun(a):
	maxv = max(a)
	sum1 = 0 
	if maxv == a[len(a)/2]:
		return maxv
		#x1 =  [ i for i in a if i==maxv]
		#if len(x1)==1:
		#	return a[fil1*fil1/2]
	return 0

image = Image.open('assignment2_images/butterfly.jpg')
w = image.size[0]
h = image.size[1]

#image.show()
im = image.convert('L')
#im.show()
#image.getpixel((1,1))
#im.size
ar = np.array(im).astype(float)
ar1 = ar/255
scale_space = np.empty((levels, h,w))
final = np.empty((levels,h,w))
temp = ar1
cx = []
cy = []
rad = []

radius = s
for i in range(levels):
	ar2 = s*s*scipy.ndimage.filters.gaussian_laplace(temp, sigma = s)
	s = s*1.25
	ar4 = np.square(ar2)
	scale_space[i] = ar4
	#plt.imshow(ar4, cmap = "gray")
	#plt.show()
'''
for i in range(levels):
	ar2 = scipy.ndimage.filters.gaussian_laplace(temp, sigma = s)
	ar3 = np.abs(ar2)
	ar4 = np.square(ar3)
	scale_space[i] = scipy.misc.imresize(ar4, (h,w)).astype(float)/255
	temp = scipy.misc.imresize(temp,(temp.shape[0]/2,temp.shape[1]/2))
	temp = temp.astype(float)
	temp = temp/255


s = s_initial
for i in range(levels):
	siz = np.ceil(s* (1.25**i)*2*1.414)
	#siz = fil1*(i+1) + (i+1+1)%2
	if siz%2==0:
		siz = siz+1
	siz = siz.astype(int)
	print(siz)
	siz=3
	#scale_space[i] = scipy.ndimage.filters.generic_filter(scale_space[i], myfun, size=siz )
	#np.clip(final[i],0,max(final[i])*0.75)
	#plt.imshow(final[i], cmap = "gray")
	#plt.show()
'''
scale_space = scipy.ndimage.filters.generic_filter(scale_space, myfun, size=7 )
final = scale_space
maxv = np.zeros((levels,1))
for i in range(levels):
	print(np.max(final[i]))
	maxv[i] = np.max(final[i])*threshold
s = s_initial
for j in range(h):
	for k in range(w):
		for i in range(levels):
		#i = np.argmax(final[:,j,k])
			if final[i,j,k]>maxv[i]:
				cx.append(k)
				cy.append(j)
				rad.append(s*(1.25**i)*1.414)

show_all_circles(ar1, np.asarray(cx),np.asarray(cy),np.asarray(rad))
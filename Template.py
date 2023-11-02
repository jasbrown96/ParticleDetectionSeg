from sklearn.metrics import pairwise_distances

import time

from skimage.color import rgb2gray
from skimage import data, color
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter

from scipy import ndimage
from scipy.signal import find_peaks

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Snake import snake_contour

im_dir = '/content/drive/MyDrive/Washers/Experiments/Tifs for Jason - uploaded 1017/Multi-class images (for later)/230313 Tm month block 0.1/Tm_bf002_35.1_BF.tif'
im = cv2.imread(im_dir)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#plt.imshow(im)
img_ref = im[285:605,380:700]
#plt.imshow(img_ref)

def get_ref(img_ref, n_angles=10, plot=False):

  s = np.linspace(0, 2*np.pi, 400)
  r = 150 + 200*np.sin(s)
  c = 150 + 200*np.cos(s)
  init = np.array([r, c]).T

  snake = active_contour(gaussian(img_ref, 3, preserve_range=False),
                        init, alpha=0.025, beta=10, gamma=0.001) #alpha=0.015

  if plot:
    plt.imshow(img_ref)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img_ref, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.scatter(snake[:, 1], snake[:, 0], c=np.arange(len(snake)), lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img_ref.shape[1], img_ref.shape[0], 0])

    plt.scatter(np.mean(snake,axis=0)[1], np.mean(snake,axis=0)[0])
    plt.show()

  a = np.zeros_like(img_ref)
  for x,y in snake:
    a[int(x),int(y)]=1

  b = []
  snake_list=[]
  for i in range(n_angles):
    b.append(ndimage.rotate(a, 90//n_angles*i, reshape=True))
  for i in range(n_angles):
    snake_list.append(rotate_snake(snake,angle=90//n_angles*i))
  for i, init in enumerate(snake_list):
    a = np.mean(init,axis=0)
    init = init* 0.75
    init += a - np.mean(init,axis=0)
    snake_list[i] = init
  return b, snake_list

def rotate_snake(snake, angle):#angle degrees
  #convert to radians.
  angle = angle*np.pi/180
  center = np.mean(snake,axis=0)
  rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle), np.cos(angle)]])
  temp = (rot@(snake-center).T).T
  return (temp+center).astype(int)

def create_dir(results_dir):
  if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

def gen_hough(im, temp):
  pad=800
  c=np.zeros((im.shape[0]+pad,im.shape[1]+pad,len(temp)))
  for (i, j), value in np.ndenumerate(im):
    if value:
      i+=pad//2
      j+=pad//2
      for k in range(len(temp)):
        height, length = temp[k].shape[0], temp[k].shape[1]
        temp1 = c[i-height//2:i+height//2, j-length//2:j+length//2,k]
        temp2 = temp[k][height//2-height//2:height//2+height//2, length//2-length//2: length//2+length//2]
        c[i-height//2:i+height//2, j-length//2:j+length//2,k] += temp[k][height//2-height//2:height//2+height//2, length//2-length//2: length//2+length//2]
  return c[pad//2:-pad//2, pad//2:-pad//2]

def non_max_suppression(accumulator, thresh, distance):
  #peaks can be fed in as
  #Returns mask for peaks
  if len(accumulator.shape)==3:
    x,y = np.nonzero(np.amax(accumulator,axis=2)>thresh)
    z = np.argmax(accumulator,axis=2)
    z = z[x,y] #x,y track the coordinate for the maximal value and z tracks the orientation/which angle of the reference is the best
  else:
    x,y = np.nonzero(accumulator>thresh)
    z = None
  if x.size == 0:    #I.e. x is empty
    return None, None
  feat = np.vstack((x,y)).T
  dist = pairwise_distances(feat)
  temp = np.amax(accumulator,axis=2)
  mask = np.array([True for i in x])
  for i in range(len(x)):
    for j in range(i+1,len(x)):
      if (dist[i,j]<distance) and (mask[i] == True) and (mask[j] == True):
        val1, val2 = temp[x[i],y[i]], temp[x[j],y[j]]
        if  val1 >= val2:
          mask[j] = False
        elif val1 < val2:
          mask[i] = False
  return x[mask],y[mask],z[mask]

def circle_filter(pts1, pts2, dist):
  x1,y1,z=pts1
  x2,y2=pts2
  mask = [True for i in x1]

  for i in range(len(x1)):
    for j in range(len(x2)):
      if (x1[i] - x2[j])**2 + (y1[i]-y2[j])**2 <dist**2:
        mask[i] = False
        break
  return x1[mask], y1[mask], z[mask]

ref, snake_list = get_ref(img_ref)

def detect_squares(image_dir, ref=ref, threshold=70):

  im = cv2.imread(image_dir)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  #img_blur = cv2.GaussianBlur(im, (5,5), 0)
  #im = cv2.Canny(image=img_blur, threshold1=100, threshold2=150)
  im = canny(im, sigma=0, low_threshold=100, high_threshold=150)

  #Find squares accumulator
  accumulator_square = gen_hough(im, ref)

  #Find circle accumulator
  #hough_radii = np.arange(110, 130, 5)

  x,y,z = non_max_suppression(accumulator_square, thresh=threshold, distance=150)
  if x is None:
    return None, None#, accumulator_square

  return x,y,z#, accumulator_square

def detect_circles(image_dir, threshold=0.3):
  im = cv2.imread(image_dir)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  #img_blur = cv2.GaussianBlur(im, (5,5), 0)
  #im = cv2.Canny(image=img_blur, threshold1=100, threshold2=150)

  im = canny(im, sigma=0, low_threshold=100, high_threshold=150)

  #Find circle accumulator
  hough_radii = np.arange(110, 130, 5)
  hough_res = hough_circle(im, hough_radii)

  # Select the most prominent 3 circles
  accumulator_circle, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=150, min_ydistance=150, threshold=threshold)

  return cy,cx, radii


def detect_squares_nocirc(image_dir, ref=ref, thresh_square=70, thresh_circle=0.3):
  x,y,_ = detect_circles(image_dir, threshold=thresh_circle) # Note, x = cy and y = cx for the detection because of the way images are read in.X here represents dist from top and y is dist from left

  x1,y1,z = detect_squares(image_dir, ref=ref, threshold=thresh_square)
  if x1 is None:
    return [], []

  x2,y2,z = circle_filter([x1,y1,z], [x,y], dist=150)
  return x2,y2,z


def disp_particles(im_dir, x, y):
  img = cv2.imread(im_dir)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  img_blur = cv2.GaussianBlur(img, (5,5), 0)
  im = cv2.Canny(image=img_blur, threshold1=100, threshold2=150)
  plt.imshow(im)
  plt.scatter(y,x, c='r')
  plt.show()

def snake_crop(im_dir, x, y, z, delta = -0.003, disp=True):
  # delta negative means a clockwise contour will move out and a ccw curve will move in
  # delta positive and vice versa
  im = cv2.imread(im_dir)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  crop_list = []
  for i in np.arange(len(x)):
    #print(max(0,x[i]-175), min(im.shape[0], x[i]+175), max(0,y[i]-175), max(im.shape[1],y[i]+175) )
    t,b,l,r = max(0,x[i]-175), min(im.shape[0], x[i]+175), max(0,y[i]-175), min(im.shape[1],y[i]+175)
    img = im[t:b,l:r]
    #s = np.linspace(0, 2*np.pi, 400)
    #r = 150 + 200*np.sin(s)
    #c = 150 + 200*np.cos(s)
    #init = np.array([r, c]).T
    init = snake_list[z[i]]
    mean = np.mean(init,axis=0, dtype=int)
    init += [175,175] - mean #[175,175] is center of cropped region and also corresponds to the ping

    snake = snake_contour(img,init,alpha=0.03, beta=10, gamma=0.001, delta=delta)
    
    crop_list.append((img,snake, init))
    if disp:
      fig, ax = plt.subplots(figsize=(7, 7))
      ax.imshow(img, cmap=plt.cm.gray)
      ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
      ax.scatter(snake[:, 1], snake[:, 0], c=np.arange(len(snake)), lw=3)
      ax.set_xticks([]), ax.set_yticks([])
      ax.axis([0, img.shape[1], img.shape[0], 0])

      #plt.scatter(np.mean(snake,axis=0)[1], np.mean(snake,axis=0)[0])
      plt.scatter(mean[0],mean[1],c='b')
      plt.show()
      dist = dist_center(snake)
      plt.plot(dist)
      peaks = find_peaks(dist, prominence = 3)
      [plt.axvline(p, c='C3', linewidth=0.3) for p in peaks[0]]
      plt.show()
  return crop_list

def circle_crop(im_dir,cx,cy,radii, disp=True):
  image = cv2.imread(im_dir)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = color.gray2rgb(image)
  crop_list = []
  for center_y,center_x,radius in zip(cy,cx,radii):
    circy,circx = circle_perimeter(center_x,center_y,radius,shape=image.shape)
    image[circy,circx] = (255,20,20)

    t,b,l,r = max(0,center_x-radius), min(im.shape[0], center_x+radius), max(0,center_y-radius), min(im.shape[1],center_y+radius)
    crop_list.append((image[t:b,l:r],[circx,circy]))

    if disp:
      fig, ax = plt.subplots(figsize=(7, 7))
      ax.imshow(image[t:b,l:r])
      plt.show()

  return crop_list


def dist_center(snake):
  center = np.mean(snake,axis=0)
  print(center)
  dist = snake-center
  print(dist[:5])
  dist = np.sqrt(np.sum(dist**2,axis=1))
  print(dist.shape)
  return dist

def save_detect(im_dir, save_dir, x_square, y_square, z_square, x_circle, y_circle, radii, segment=True, edge_crop=True):
  im = cv2.imread(im_dir)
  plt.figure(figsize=(10,10))
  
  if edge_crop:
    height,width,_ = im.shape
    edge_dist = 140
    edge_mask = np.logical_or.reduce((y_circle<edge_dist, y_circle>height-edge_dist, x_circle<edge_dist, x_circle>width-edge_dist))
    x_circle = x_circle[~edge_mask]
    y_circle = y_circle[~edge_mask]

    edge_mask = np.logical_or.reduce((y_square<edge_dist, y_square>height-edge_dist, x_square<edge_dist, x_square>width-edge_dist))
    x_square = x_square[~edge_mask]
    y_square = y_square[~edge_mask]

  if segment:
     
     for center_y,center_x,radius in zip(y_circle,x_circle,radii):
        circy,circx = circle_perimeter(center_x,center_y,radius,shape=im.shape)
        plt.scatter(circx,circy, c='r')
    
     crop_list = snake_crop(im_dir,x_square,y_square, z_square, disp=False)
     for i, (crop, snake, init) in enumerate(crop_list):
        height,width = crop.shape
        plt.plot(init[:, 1]+y_square[i]-width//2, init[:, 0]+x_square[i]-height//2, 'r', alpha=0.75)
        plt.plot(snake[:, 1]+y_square[i]-width//2, snake[:, 0]+x_square[i]-height//2, 'b', alpha=0.75)
        plt.scatter(np.mean(snake,axis=0)[1]+y_square[i]-width//2, np.mean(snake,axis=0)[0]+x_square[i]-height//2, c='g')
 
  plt.imshow(im)
  plt.scatter(y_circle,x_circle, c='r')

  plt.scatter(y_square,x_square, c='b')

  plt.savefig(save_dir)
  plt.show()


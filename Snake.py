# https://www.mathworks.com/matlabcentral/fileexchange/28149-snake-active-contour
# https://scikit-image.org/docs/stable/auto_examples/edges/plot_active_contours.html

from skimage.filters import gaussian
import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.filters import sobel
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type

def GetContourNormals2D(xt,yt, a=4):
    # Extract x and y coordinates
    #xt = P[:, 0]
    #yt = P[:, 1]

    # Derivatives of contour
    n = len(xt)
    f = (np.arange(n) + a) % n
    b = (np.arange(n) - a) % n

    dx = xt[f] - xt[b]
    dy = yt[f] - yt[b]

    # Normals of contour points
    l = np.sqrt(dx**2 + dy**2)
    nx = -dy / l
    ny = dx / l

    # Create the N matrix with contour normals
    #N = np.column_stack((nx, ny))

    return nx,ny


def snake_contour(image, snake, alpha=0.01, beta=0.1, w_line=0, w_edge=1,  gamma=0.01,
                   max_px_move=1.0,max_num_iter=2500, convergence=0.1, delta = 0.01):


  image = gaussian(image, 3, preserve_range=False)

  img = img_as_float(image)
  float_dtype = _supported_float_type(image.dtype)
  img = img.astype(float_dtype, copy=False)

  convergence_order = 10
  RGB = img.ndim == 3

  #Assume Grayscale
  if w_edge != 0:
      if RGB:
          edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                  sobel(img[:, :, 2])]
      else:
          edge = [sobel(img)]
  else:
      edge = [0]

  if RGB:
      img = w_line*np.sum(img, axis=2) \
          + w_edge*sum(edge)
  else:
      img = w_line*img + w_edge*edge[0]

  # Interpolate for smoothness:
  intp = RectBivariateSpline(np.arange(img.shape[1]),
                              np.arange(img.shape[0]),
                              img.T, kx=2, ky=2, s=0)

  snake_xy = snake[:, ::-1]
  x = snake_xy[:, 0].astype(float_dtype)
  y = snake_xy[:, 1].astype(float_dtype)
  n = len(x)
  xsave = np.empty((convergence_order, n), dtype=float_dtype)
  ysave = np.empty((convergence_order, n), dtype=float_dtype)

  # Build snake shape matrix for Euler equation in double precision
  eye_n = np.eye(n, dtype=float)
  a = (np.roll(eye_n, -1, axis=0)
        + np.roll(eye_n, -1, axis=1)
        - 2 * eye_n)  # second order derivative, central difference
  b = (np.roll(eye_n, -2, axis=0)
        + np.roll(eye_n, -2, axis=1)
        - 4 * np.roll(eye_n, -1, axis=0)
        - 4 * np.roll(eye_n, -1, axis=1)
        + 6 * eye_n)  # fourth order derivative, central difference
  A = -alpha * a + beta * b
  # Only one inversion is needed for implicit spline energy minimization:
  inv = np.linalg.inv(A + gamma * eye_n)
  # can use float_dtype once we have computed the inverse in double precision
  inv = inv.astype(float_dtype, copy=False)

  for i in range(max_num_iter):
      # RectBivariateSpline always returns float64, so call astype here
      fx = intp(x, y, dx=1, grid=False).astype(float_dtype, copy=False)
      fy = intp(x, y, dy=1, grid=False).astype(float_dtype, copy=False)

      nx,ny = GetContourNormals2D(x,y)

      xn = inv @ (gamma*x + fx + delta*nx)
      yn = inv @ (gamma*y + fy + delta*ny)

      # Movements are capped to max_px_move per iteration:
      dx = max_px_move * np.tanh(xn - x)
      dy = max_px_move * np.tanh(yn - y)

      x += dx
      y += dy

      # Convergence criteria needs to compare to a number of previous
      # configurations since oscillations can occur.
      j = i % (convergence_order + 1)
      if j < convergence_order:
          xsave[j, :] = x
          ysave[j, :] = y
      else:
          dist = np.min(np.max(np.abs(xsave - x[None, :])
                                + np.abs(ysave - y[None, :]), 1))
          if dist < convergence:
              break
  return np.stack([y, x], axis=1)


# import cv2 as cv
# import matplotlib.pyplot as plt
# im_dir = '/content/drive/MyDrive/Washers/Experiments/Tifs for Jason - uploaded 1017/Multi-class images (for later)/230313 Tm month block 0.1/Tm_bf002_35.1_BF.tif'
# im = cv.imread(im_dir)
# im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# image = im[285:605,380:700]

# x = 100*np.cos(np.linspace(0,2*np.pi,400))+150
# y = 100*np.sin(np.linspace(0,2*np.pi,400))+150
# P = np.vstack([x[:-1],y[:-1]]).T

# P2 = snake(image,P,alpha=0.03, beta=10, gamma=0.001, delta=0.0025)

# plt.imshow(image)
# plt.plot(P2[:,1],P2[:,0], c='r')
# plt.show()
# Particle Detection and Segmentation
The Libraries Needed are OpenCV, Numpy, Scipy, Skimage

# Script to run demo

from Template import detect_squares_nocirc, detect_circles, save_detect, create_dir

im_dir = './##Desired BF image##'

square_thresh = 60
circle_thresh = 0.25
save_dir = './save'
create_dir(save_dir)

save_file = save_dir+'/demo.png'
x_square, y_square, z_square = detect_squares_nocirc(im_dir, thresh_square=square_thresh, thresh_circle=circle_thresh)
x_circle, y_circle, radii = detect_circles(im_dir, threshold=circle_thresh)
save_detect(im_dir, save_file, x_square, y_square, z_square, x_circle, y_circle, radii)




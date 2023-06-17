import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


class ImageProcessingPipeline:
    def __init__(self):
        self.images = []


    # equalization

    def equalize_histogram(self): # equalization
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.equalizeHist(self.images[i])
        
    def equalize_clahe(self): # contrast limited adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for i in np.arange(len(self.images)):
            self.images[i] = clahe.apply(self.images[i])


    # filters
    
    def filter_median(self, kernel_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.medianBlur(self.images[i], kernel_size)

    def filter_gauss(self, kernel_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.GaussianBlur(self.images[i], (kernel_size, kernel_size), 0)

    def filter_mean(self, kernel_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.blur(self.images[i], (kernel_size, kernel_size), 0)

    
    # binary conversion

    def conversion_grayscale(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)

    def conversion_binary_threshold(self, thresh):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], thresh, 255, cv2.THRESH_BINARY)

    def conversion_binary_otsu(self):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def conversion_binary_adaptive(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def conversion_binary_triangle(self):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    def conversion_binary_yen(self):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def conversion_binary_mean_adaptive(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def conversion_binary_gaussian_adaptive(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # color

    def conversion_bgr2hsv(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2HSV)

    def conversion_bgr2hls(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2HLS)
    
    def conversion_bgr2rgb(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB)

    def conversion_hsv2bgr(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_HSV2BGR)



    # segmentation using color masks

    def color_mask_hsv(self, lower, upper):
        # conver images to hsv
        self.conversion_bgr2hsv()
        # create mask
        for i in np.arange(len(self.images)):
            masked_image = cv2.bitwise_and(self.images[i], self.images[i], mask=cv2.inRange(self.images[i], lower, upper))
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
            self.images[i] = masked_image
        

    # edge detection / enhancement

    def edge_sobel(self, ksize=5):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.Sobel(self.images[i], cv2.CV_64F, 1, 1, ksize=5)


    def paint_8_connected(self, image, x, y):
        width, height = image.shape
        if (x - 1 > 0):
            image[x - 1][y] = 255
        if (x + 1 < width):
            image[x + 1][y] = 255
        if (y - 1 > 0):
            image[x - 1][y - 1] = 255
        if (y + 1 < height):
            image[x - 1][y + 1] = 255
        if (x - 1 > 0 and y - 1 > 0):
            image[x - 1][y - 1] = 255
        if (x + 1 < width and y - 1 > 0):
            image[x + 1][y - 1] = 255
        if (x - 1 > 0 and y + 1 < height):
            image[x - 1][y + 1] = 255
        if (x + 1 < width and y + 1 < height):
            image[x + 1][y + 1] = 255
        
        return image
    

    def paint_n_connected(self, image, x, y, radius):
        width, height = image.shape
        for i in np.arange(x - radius, x + radius):
            for j in np.arange(y - radius, y + radius):
                if (i > 0 and i < width and j > 0 and j < height):
                    image[i][j] = 255
        return image
    
        

    def edge_enhancement_raul(self):
        for i in np.arange(len(self.images)):
            original_image = self.images[i]
            height, width = original_image.shape
            for j in np.arange(width):
                for k in np.arange(height):
                    if (original_image[j][k] != 0):
                        if(j + 1 < width and original_image[j + 1][k] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (j - 1 > 0 and original_image[j - 1][k] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (k + 1 < height and original_image[j][k + 1] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (k - 1 > 0 and original_image[j][k - 1] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (j + 1 < width and k + 1 < height and original_image[j + 1][k + 1] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (j - 1 > 0 and k - 1 > 0 and original_image[j - 1][k - 1] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (j + 1 < width and k - 1 > 0 and original_image[j + 1][k - 1] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break
                        elif (j - 1 > 0 and k + 1 < height and original_image[j - 1][k + 1] == 0):
                            self.images[i] = self.paint_8_connected(original_image, j, k)
                            break

    def edge_detection_canny(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.Canny(self.images[i], 100, 200)

    

    

    # morphology

    def morph_dilation(self, niter=1, ksize=5):
        kernel = np.ones((ksize,ksize),np.uint8)
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.dilate(self.images[i], kernel, iterations=niter)

    def morph_erosion(self, niter=1, ksize=5):
        kernel = np.ones((ksize,ksize),np.uint8)
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.erode(self.images[i], kernel, iterations=niter)

    def morph_opening(self, niter=1, ksize=5):
        kernel = np.ones((ksize,ksize),np.uint8)
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.morphologyEx(self.images[i], cv2.MORPH_OPEN, kernel, iterations=niter)
    
    def morph_closing(self, niter=1, ksize=5):
        kernel = np.ones((ksize,ksize),np.uint8)
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.morphologyEx(self.images[i], cv2.MORPH_CLOSE, kernel, iterations=niter)

        


    
    # loading and resizing images

    def load_and_resize_images(self, directory, target_size):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                    self.images.append(resized_image)

    def load_and_resize_by_scale_images(self, directory, scale):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    self.images.append(resized_image)


    def load_images(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    self.images.append(image)


    def set_images(self, images):
        self.images = images

    def resize_by_scale(self, scale):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def resize_by_size(self, target_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], target_size, interpolation=cv2.INTER_CUBIC)


    # plotting images

    def plot_images(self):
        # Plotting the images
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))

        for i, ax in enumerate(axes.flat):
            if i < len(self.images):
                ax.imshow(cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB))
                ax.axis("off")

        plt.tight_layout()
        plt.show()
        

def test():
    pipe = ImageProcessingPipeline()
    # loading test image
    pipe.load_and_resize_images('./images', (640, 640))
    pipe.plot_images()
    pipe.filter_mean(5)
    pipe.conversion_grayscale()
    pipe.conversion_binary_threshold(127)
    pipe.plot_images()


def main():
    test()

if __name__ == '__main__':
    main()


# Image Processing Pipeline


This code allows you to create a flexible image processing pipeline, enabling the application of various transformations across multiple images simultaneously. The pipeline supports operations such as equalization, binarization, color conversion, morphology, edge detection, convolution with different filters, masking, resizing, and plotting. With a wide range of techniques available, it provides a comprehensive toolkit for effective image transformations.

The code's modularity lets you easily integrate your own image transformation functions into the pipeline class. This flexible architecture allows you to expand the pipeline's capabilities with your unique algorithms, creating a dynamic and adaptable toolkit for image processing.


## Code example

```
def test():
    pipe = ImageProcessingPipeline()
    # loading test images
    pipe.load_and_resize_images('./images', (640, 640))
    # plotting images
    pipe.plot_images()
    # applying transformations
    pipe.filter_mean(5)
    pipe.conversion_grayscale()
    pipe.conversion_binary_threshold(127)
    # plotting once again
    pipe.plot_images()

```

![image](https://github.com/Raaulsthub/image_processing_pipeline/assets/85199336/e21d2e89-50cd-4100-8156-90c3b237a0c8)

![image](https://github.com/Raaulsthub/image_processing_pipeline/assets/85199336/e6a2510f-ed18-4aee-917e-ef418ab3de4d)

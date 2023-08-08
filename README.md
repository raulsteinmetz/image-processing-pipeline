# Image Processing Pipeline


This code empowers you to construct a versatile image processing pipeline, enabling seamless application of transformations across a multitude of images simultaneously. Within this pipeline, a diverse array of operations awaits, including equalization, binarization, color conversion, morphology, edge detection, convolution utilizing an array of filters, precise masking, resizing, and sophisticated plotting. These capabilities are further enriched by a multitude of technique options, providing you with a comprehensive toolkit for crafting impeccable visual transformations.

The code boasts an exceptional level of modularity, granting you the freedom to seamlessly integrate your own image transformation functions into the meticulously designed pipeline class. This versatile architecture empowers you to expand the pipeline's capabilities with your unique transformation algorithms, contributing to a dynamic and ever-evolving toolkit for crafting captivating image modifications.


## Code example

```
def test():
    pipe = ImageProcessingPipeline()
    # loading test image
    pipe.load_and_resize_images('./images', (640, 640))
    pipe.plot_images()
    pipe.filter_mean(5)
    pipe.conversion_grayscale()
    pipe.conversion_binary_threshold(127)
    pipe.plot_images()

```

![image](https://github.com/Raaulsthub/image_processing_pipeline/assets/85199336/e21d2e89-50cd-4100-8156-90c3b237a0c8)

![image](https://github.com/Raaulsthub/image_processing_pipeline/assets/85199336/e6a2510f-ed18-4aee-917e-ef418ab3de4d)

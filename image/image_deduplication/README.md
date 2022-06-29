# Image Deduplication

Image deduplication is the process of finding exact or near-exact duplicates within a collection of images. Sometimes, some images are not exactly the same as other images, this is where the difficulty here lies - matching pure duplicates is a simple process, but matching images which are similar in the presence of changes in zoom, lighting, and noise is a much more challenging problem.

This [notebook](image_deduplication.ipynb) shows you how to use Towhee's dc API to compare duplicates or near-exact duplicates within a few lines of code.
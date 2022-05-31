# Reverse Image Search

**Reverse image search** helps you search for similar or related images given an input image. Reverse image search is a [content-based image retrieval](https://en.wikipedia.org/wiki/Content-based_image_retrieval) (CBIR) query technique that involves providing the CBIR system with a query image that it will then base its search upon. 



This reverse image search example mainly consists of three notebooks, and I think everyone can learn the basic operations of Towhee and Milvus through the [**getting started notebook**](./1_build_image_search_engine.ipynb). And the [**deep dive notebook**](./2_deep_dive_image_search.ipynb) will show you how to choose the model and how to deploy the service. In addition, [**visual ANNS**](./3_visual_anns_image_search.ipynb) is to visualize the process of searching for images with [Feder](https://github.com/zilliztech/feder).

## Learn from Notebook

- [Getting started](1_build_image_search_engine.ipynb)

In this notebook you will get the prerequisites, how to complete a simple image system search and visualize results, and how to report accuracy and performance metrics.

- [Deep Dive](./2_deep_dive_image_search.ipynb)

In this notebook you will learn how to use various models and evaluate recall metrics, also providing object detection method. There is also about how to improve system performance and stability, and finally show you how to start the FastAPI service.

- [Visualize ANNS](./3_visual_anns_image_search.ipynb)

This notebook will visualize the IVF_FLAT and HNSW index when searching images, then compare whether to normalize the vector and whether to add object detection, and finally visualize the cross-model retrieval process, which we can use text to search for images. 

> More information about Feder you can learn from "[Visualize Your Approximate Nearest Neighbor Search with Feder](https://zilliz.com/blog/Visualize-Your-Approximate-Nearest-Neighbor-Search-with-Feder)" and "[Visualize Reverse Image Search with Feder](https://zilliz.com/blog/Visualize-Reverse-Image-Search-with-Feder)"

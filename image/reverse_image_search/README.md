# Reverse Image Search

**Reverse image search** helps you search for similar or related images given an input image. Reverse image search is a [content-based image retrieval](https://en.wikipedia.org/wiki/Content-based_image_retrieval) (CBIR) query technique that involves providing the CBIR system with a query image that it will then base its search upon. 



This reverse image search example mainly consists of two notebooks, and I think everyone can learn the basic operations of Towhee and Milvus through the [**getting started notebook**](./1_build_image_search_engine.ipynb). And the [**deep dive notebook**](./2_deep_dive_image_search.ipynb) will show you how to improve performance and deploy the service.

## Learn from Notebook

- [Getting started](1_build_image_search_engine.ipynb)

In this notebook you will get the prerequisites, how to complete a simple image system search and visualize results, and how to evaluate system performance with selected metric.

- [Deep Dive](./2_deep_dive_image_search.ipynb)

In this notebook you will learn how to normalize embeddings, apply object detection, and reduce embedding dimension.
There is also a section starting a simple online demo using Gradio.

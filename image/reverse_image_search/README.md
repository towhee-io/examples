# Reverse Image Search

This image search example mainly consists of three notebooks, and two python files about "how to load large images" and "start an online service".



I think everyone can learn the basic operations of Towhee and Milvus through the [**getting started notebook**](./build_image_search_engine.ipynb). And the [**deep dive notebook**](./deep_dive_image_search.ipynb) will show you how to choose the model and how to deploy the service. In addition, [visual ANNS](./visual_anns_image_search.ipynb) is to visualize the process of searching for pictures

[**load.py**](./load.py) is used to import your large-scale image data, and [**server.py**](./server.py) will start a FastAPI-based service.

## Learn from Notebook

- [Getting started](build_image_search_engine.ipynb)

In this notebook you will get the prerequisites, how to complete a simple image system search and visualize results, and how to report accuracy and performance metrics.

- [Deep Dive](./deep_dive_image_search.ipynb)

In this notebook you will learn how to use various models and evaluate recall metrics, also providing object detection method. There is also about how to improve system performance and stability, and finally show you how to start the FastAPI service.

- [Visualize ANNS](./visual_anns_image_search.ipynb)

This notebook will visualize the IVF_FLAT and HNSW index when searching images, then compare whether to normalize the vector and whether to add object detection, and finally visualize the cross-model retrieval process, which we can use text to search for images.

## Load Large-scale Image Data

I think you already know from previous notebooks that a very important step in reverse image search is loading the data. If you have large-scale data, you can try running the `set_parallel` and `exception_safe` methods in [load.py](./load.py), which make the import process faster and safer.

> You can load your own data in this script.

```bash
$ python load.py
Collection number: 1000
```

## Deploy with FastAPI

After the data is loaded, you can start the search service for reverse image search, and also support inserting data services.

```bash
$ python server.py
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Next you can test the service with the following command.

```bash
# upload an image and search
$ curl -X POST "http://0.0.0.0:8000/search"  --data-binary @extracted_test/n01443537/n01443537_3883.JPEG -H 'Content-Type: image/jpeg'

# upload an image and insert
$ curl -X POST "http://0.0.0.0:8000/insert"  --data-binary @extracted_test/n01443537/n01443537_3883.JPEG -H 'Content-Type: image/jpeg'

# count the collection
$ curl -X POST "http://0.0.0.0:8000/count"
1001
```

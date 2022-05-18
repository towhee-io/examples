# Molecular Search

This molecular search example mainly consists of two notebooks, and two python files about "how to load large molecular formula" and "start an online service".



I think everyone can learn the basic operations of Molecular Search System through the [**getting started notebook**](./build_molecular_search_engine.ipynb). And the [**deep dive notebook**](./deep_dive_molecular_search.ipynb) will show you how to deploy the service.  

[**load.py**](./load.py) is used to import your large-scale data, and [**server.py**](./server.py) will start a FastAPI-based service.

## Learn from Notebook

- [Getting started](build_molecular_search_engine.ipynb)

In this notebook you will get the prerequisites, how to complete a simple molecular search system and visualize the results.

- [Deep Dive](./deep_dive_molecular_search.ipynb)

In this notebook you will learn how to improve system performance and stability, and finally show you how to start the FastAPI service.

## Load Large-scale Data

I think you already know from previous notebooks that a very important step in molecular search is loading the data. If you have large-scale data, you can try running the `set_parallel` and `exception_safe` methods in [load.py](./load.py), which make the import process faster and safer.

> You can load your own data in this script.

```bash
$ python load.py
Collection number: 10000
```

## Deploy with FastAPI

After the data is loaded, you can start the search service for molecular search, and also support inserting data services.

```bash
$ python server.py
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Next you can test the service with the following command.

```bash
# search the similar molecular
$ curl -X POST "http://0.0.0.0:8000/similarity"  --data "Cn1ccc(=O)nc1"

# search the superstructure molecular
$ curl -X POST "http://0.0.0.0:8000/superstructure"  --data "Cn1ccc(=O)nc1"

# search the substructure molecular
$ curl -X POST "http://0.0.0.0:8000/substructure"  --data "Cn1ccc(=O)nc1"

# insert a molecular
$ curl -X POST "http://0.0.0.0:8000/insert"  --data "Cn1ccc(=O)nc1"

# count the collection
$ curl -X POST "http://0.0.0.0:8000/count"
```

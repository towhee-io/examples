# Reverse Image Search

- Run notebook with [reverse_image_search.ipynb](./reverse_image_search.ipynb).

- Start server with fastapi.

  ```bash
  $ python api.py
  INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```

  Test the server:

  > Before running it, make sure you have created the `collection_name` collection in **api.py**. And the following example already has 200 entities in Milvus' 'reverse_image_search' collection.

  ```bash
  $ curl -X POST "http://0.0.0.0:8000/insert"  -d data/1.jpg
  {"mr":"(insert count: 1, delete count: 0, upsert count: 0, timestamp: 432699123898515457)"}
  $ curl -X POST "http://0.0.0.0:5000/count"
  201
  $ curl -X POST "http://0.0.0.0:8000/search"  -d data/1.jpg
  {"path":"/Users/chenshiyu/workspace/data/pic/1.jpg","result":[83682058,9605572,67200803,69780733,45421918,99335156,2345006,5632544,5632544,28616422]}%
  
  
  ```

  

# Reverse Image Search

- Run notebook with [reverse_image_search.ipynb](./reverse_image_search.ipynb).

- Start server with fastapi.

  ```bash
  $ python api.py
  INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```

  Test the server:

  > Before running it, make sure you have created the `collection_name` collection in **api.py**. And the following example already has 1000 
  > entities in Milvus' 'reverse_image_search' collection.

  ```bash
  # upload an image and search
  $ curl -X POST "http://0.0.0.0:8000/search"  --data-binary @extracted_test/n01443537/n01443537_3883.JPEG -H 'Content-Type: image/jpeg'
  {"path":"/Users/chenshiyu/workspace/data/pic/1.jpg","result":[83682058,9605572,67200803,69780733,45421918,99335156,2345006,5632544,5632544,
  28616422]}%
  # upload an image and insert
  $ curl -X POST "http://0.0.0.0:8000/insert"  --data-binary @extracted_test/n01443537/n01443537_3883.JPEG -H 'Content-Type: image/jpeg'
  {"mr":"(insert count: 1, delete count: 0, upsert count: 0, timestamp: 432699123898515457)"}
  # count the collection
  $ curl -X POST "http://0.0.0.0:8000/count"
  1001
  ```

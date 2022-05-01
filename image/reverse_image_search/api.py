import towhee
import time
import uvicorn
from fastapi import FastAPI
from diskcache import Cache

app = FastAPI()
cache = Cache('./tmp')
milvus_collection = towhee.connectors.milvus(uri='tcp://127.0.0.1:19530/reverse_image_search')

@towhee.register(name='get_path_id')
def get_path_id(path):
    timestamp = int(time.time()*10000)
    cache[path] = timestamp
    cache[timestamp] = path
    return int(timestamp)


with towhee.api['file']() as api:
    app_insert = (
        api.image_load['file', 'img']()
        .save_image['img', 'path'](dir='tmp/images')
        .get_path_id['path', 'id']()
        .image_embedding.timm['img', 'vec'](model_name='resnet101')
        .ann_insert[('id', 'vec'), 'res'](ann_index=milvus_collection)
        .select['id', 'path']()
        .serve('/insert', app)
    )

with towhee.api['file']() as api:
    app_search = (
        api.image_load['file', 'img']()
        .image_embedding.timm['img', 'vec'](model_name='resnet101')
        .ann_search['vec', 'result'](ann_index=milvus_collection)
        .runas_op['result', 'res_file'](func=lambda res: [cache[x.id] for x in res])
        .select['res_file']()
        .serve('/search', app)
    )


with towhee.api() as api:
    app_count = (
        api.map(lambda _: milvus_collection.count())
        .serve('/count', app)
        )

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8000)

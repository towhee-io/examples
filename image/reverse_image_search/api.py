from fastapi import FastAPI
import uvicorn
import towhee
from towhee import register
from pymilvus import connections, Collection


connections.connect(host='localhost', port='19530')

app = FastAPI()
collection_name = 'reverse_image_search'
search_args = dict(
    param={"params": {"nprobe": 10}},
    output_fields=['path'],
    limit=10
)


@register(name='milvus-insert')
class MilvusInsert:
    def __init__(self, collection):
        self.collection = collection
        if isinstance(collection, str):
            self.collection = Collection(collection)

    def __call__(self, *args, **kwargs):
        data = []
        for iterable in args:
            data.append([iterable])
        mr = self.collection.insert(data)
        return str(mr)


with towhee.api['file']() as api:
    app_insert = (
        api.image_load['file', 'img']()
        .save_image['img', 'path'](dir='tmp/images')
        .image_embedding.timm['img', 'vec'](model_name='resnet101')
        .runas_op['path', 'path'](func=lambda path: abs(hash(path)) % (10 ** 8))
        .milvus_insert[('path', 'vec'), 'mr'](collection=collection_name)
        .select['mr']()
        .serve('/insert', app)
        )


with towhee.api['file']() as api:
    app_search = (
        api.image_load['file', 'img']()
        .image_embedding.timm['img', 'vec'](model_name='resnet101')
        .milvus_search['vec', 'result'](collection=collection_name, **search_args)
        .runas_op['result', 'result'](func=lambda res: [x.path for x in res])
        .select['result']()
        .serve('/search', app)
        )


@register(name='milvus-count')
class MilvusCount:
    def __init__(self, collection):
        self.collection = collection
        if isinstance(collection, str):
            self.collection = Collection(collection)

    def __call__(self, *args):
        return self.collection.num_entities


with towhee.api() as api:
    app_count = (
        api.milvus_count(collection=collection_name)
        .serve('/count', app)
        )


if __name__ == '__main__':
    uvicorn.run(app=app)

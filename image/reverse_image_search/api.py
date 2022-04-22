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


with towhee.api('/insert') as api:
    app_insert = (
        api.as_entity(schema=['path'])
        .image_decode['path', 'img']()
        .image_embedding.timm['img', 'vec'](model_name='resnet50')
        .runas_op['path', 'hash_path'](func=lambda path: abs(hash(path)) % (10 ** 8))  # delete when support String in Milvus2.1
        .milvus_insert[('hash_path', 'vec'), 'mr'](collection=collection_name)
        .select['mr']()
        .bind(app)
        )


with towhee.api('/search') as api:
    app_search = (
        api.as_entity(schema=['path'])
        .image_decode['path', 'img']()
        .image_embedding.timm['img', 'vec'](model_name='resnet50')
        .milvus_search['vec', 'result'](collection=collection_name, **search_args)
        .runas_op['result', 'result'](func=lambda res: [x.path for x in res])
        .select['path', 'result']()
        .bind(app)
        )


@register(name='milvus-count')
class MilvusCount:
    def __init__(self, collection):
        self.collection = collection
        if isinstance(collection, str):
            self.collection = Collection(collection)

    def __call__(self, *args):
        return self.collection.num_entities


with towhee.api('/count') as api:
    app_count = (
        api.milvus_count(collection=collection_name)
        .bind(app)
        )


if __name__ == '__main__':
    uvicorn.run(app=app)

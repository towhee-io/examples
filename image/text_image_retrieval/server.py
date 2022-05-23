import towhee
import pandas as pd
import time
import uvicorn
from fastapi import FastAPI
from pymilvus import connections, Collection

app = FastAPI()

collection_name = 'text_image_search'
csv_file = 'reverse_image_search.csv'
model_name = 'clip_vit_b32'

connections.connect(host='127.0.0.1', port='19530')
milvus_collection = Collection(collection_name)

df = pd.read_csv(csv_file)
id_img = df.set_index('id')['path'].to_dict()
label_ids = {}
for label in set(df['label']):
    label_ids[label] = list(df[df['label']==label].id)


@towhee.register(name='get_path_id')
def get_path_id(path):
    timestamp = int(time.time()*10000)
    id_img[timestamp] = path
    return timestamp


@towhee.register(name='milvus_insert')
class MilvusInsert:
    def __init__(self, collection):
        self.collection = collection

    def __call__(self, *args, **kwargs):
        data = []
        for iterable in args:
            data.append([iterable])
        mr = self.collection.insert(data)
        self.collection.load()
        return str(mr)


with towhee.api['file']() as api:
    app_insert = (
        api.image_load['file', 'img']()
        .save_image['img', 'path'](dir='tmp/images')
        .get_path_id['path', 'id']()
        .towhee.clip['img', 'vec'](model_name=model_name,modality='image')
        .milvus_insert[('id', 'vec'), 'res'](collection=milvus_collection)
        .select['id', 'path']()
        .serve('/insert', app)
    )


with towhee.api['text']() as api:
    app_search = (
        api.towhee.clip['text', 'vec'](model_name=model_name,modality='text')
        .milvus_search['vec', 'result'](collection=milvus_collection)
        .runas_op['result', 'res_file'](func=lambda res: str([id_img[x.id] for x in res]))
        .select['res_file']()
        .serve('/search', app)
    )


with towhee.api() as api:
    app_count = (
        api.map(lambda _: milvus_collection.num_entities)
        .serve('/count', app)
        )


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8000)

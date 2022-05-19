import towhee
import pandas as pd
import time
import uvicorn
from fastapi import FastAPI
from pymilvus import connections, Collection

app = FastAPI()

collection_name = 'molecular_search'
csv_file = 'pubchem_10000.smi'
algorithm = 'daylight'

connections.connect(host='127.0.0.1', port='19530')
milvus_collection = Collection(collection_name)

df = pd.read_csv(csv_file)
id_smiles = df.set_index('id')['smiles'].to_dict()


@towhee.register(name='get_smiles_id')
def get_smiles_id(smiles):
    timestamp = int(time.time()*10000)
    id_smiles[timestamp] = smiles
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


with towhee.api['smiles']() as api:
    app_insert = (
        api.get_smiles_id['smiles', 'id']()
        .molecular_fingerprinting['smiles', 'fp'](algorithm='daylight')
        .milvus_insert[('id', 'fp'), 'res'](collection=milvus_collection)
        .select['id', 'res']()
        .serve('/insert', app)
    )


with towhee.api['smiles']() as api:
    app_search = (
        api.molecular_fingerprinting['smiles', 'fp'](algorithm='daylight')
        .milvus_search['fp', 'result'](collection=milvus_collection, metric_type='JACCARD')
        .runas_op['result', 'similar_smile'](func=lambda res: [id_smiles[x.id] for x in res])
        .select['smiles', 'similar_smile']()
        .serve('/similarity', app)
    )


with towhee.api['smiles']() as api:
    app_search = (
        api.molecular_fingerprinting['smiles', 'fp'](algorithm='daylight')
        .milvus_search['fp', 'result'](collection=milvus_collection, metric_type='SUPERSTRUCTURE')
        .runas_op['result', 'superstructure'](func=lambda res: [id_smiles[x.id] for x in res])
        .select['smiles', 'superstructure']()
        .serve('/superstructure', app)
    )


with towhee.api['smiles']() as api:
    app_search = (
        api.molecular_fingerprinting['smiles', 'fp'](algorithm='daylight')
        .milvus_search['fp', 'result'](collection=milvus_collection, metric_type='SUBSTRUCTURE')
        .runas_op['result', 'substructure'](func=lambda res: [id_smiles[x.id] for x in res])
        .select['smiles', 'substructure']()
        .serve('/substructure', app)
    )


with towhee.api() as api:
    app_count = (
        api.map(lambda _: milvus_collection.num_entities)
        .serve('/count', app)
        )


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8000)

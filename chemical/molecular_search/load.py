import towhee
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


def create_milvus_collection(collection_name, dim):
    connections.connect(host='127.0.0.1', port='19530')

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.BINARY_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='molecular similarity search')
    collection = Collection(name=collection_name, schema=schema)

    return collection


def main():
    collection_name = 'molecular_search'
    csv_file = 'pubchem_10000.smi'
    algorithm = 'daylight'
    milvus_collection = create_milvus_collection(collection_name, 2048)
    connections.connect(host='127.0.0.1', port='19530')

    (towhee.read_csv(csv_file)
     .exception_safe()
     .runas_op['id', 'id'](func=lambda x: int(x))
     .molecular_fingerprinting['smiles', 'fp'](algorithm=algorithm)
     .drop_empty()
     .to_milvus['id', 'fp'](collection=milvus_collection, batch=100)
     )
    print('Collection number: ', milvus_collection.num_entities)


if __name__ == '__main__':
    main()

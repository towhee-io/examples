import towhee
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


def create_milvus_collection(collection_name, dim):
    connections.connect(host='127.0.0.1', port='19530')

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def main():
    collection_name = 'reverse_image_search'
    csv_file = 'reverse_image_search.csv'
    model_name = 'resnet50'
    parallel_num = 4
    milvus_collection = create_milvus_collection(collection_name, 2048)
    connections.connect(host='127.0.0.1', port='19530')

    (towhee.read_csv(csv_file)
     .exception_safe()
     .set_parallel(parallel_num)
     .runas_op['id', 'id'](func=lambda x: int(x))
     .image_decode['path', 'img']()
     .image_embedding.timm['img', 'vec'](model_name=model_name)
     .tensor_normalize['vec', 'vec']()
     .drop_empty()
     .to_milvus['id', 'vec'](collection=milvus_collection, batch=100)
     )
    print('Collection number: ', milvus_collection.num_entities)


if __name__ == '__main__':
    main()

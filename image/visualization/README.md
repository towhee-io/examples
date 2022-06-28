# Visualization with Towhee

There are some tutorials to visualization and learn around the principle in image search. For example, we can visualize the attention mechanism of embedding models with heatmaps, or use Feder to visualize the embedding approximate nearest neighbors search (ANNS) process.

## Learn from Notebook

- [Visualize Models](./under_the_hood_embedding_models.ipynb)

With some visualization tools in towhee, this tutorial show some examples for model interpretability. Towhee provides state-of-the-art interpretability and visualization algorithms, including attribution-based algorithms, embedding visualization algorithms, attention visualization algorithms, to provide researchers and developers with an easy way to understand features and which features are contributing to a model’s output.

- [Visualize ANNS](./under_the_hood_anns_index.ipynb)

This notebook will visualize the IVF_FLAT and HNSW index when searching images with [feder](https://github.com/zilliztech/feder), then compare whether to normalize the vector and whether to add object detection, and finally visualize the cross-model retrieval process, which we can use text to search for images. 

> More information about Feder you can learn from "[Visualize Your Approximate Nearest Neighbor Search with Feder](https://zilliz.com/blog/Visualize-Your-Approximate-Nearest-Neighbor-Search-with-Feder)" and "[Visualize Reverse Image Search with Feder](https://zilliz.com/blog/Visualize-Reverse-Image-Search-with-Feder)"

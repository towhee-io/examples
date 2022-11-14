# code search

## 运行流程

* 首先使用代码摘要任务做为编码器预训练任务

  * ```
    运行脚本见scripts/codeSearchNet-py/transformer.sh
    1、修改MODEL_DIR参数，可指定模型和log保存路径，比如现在默认是tmp_CSN_py_1
    2、MODEL_NAME参数可在命令行运行时自主指定，我指定的是train_CSN_py_1，所以训练完自动生成了tmp_CSN_py_1目录，其下会有log文件及保存的模型参数，我删了log等文件，保存了train_CSN_py_1.mdl，后续可直接使用。
    ```

* 再进行代码搜索模型的训练

  * ```
      运行脚本见scripts/codeSearchNet-py/code_search.sh
      1、RGPU参数按需修改
      2、MODEL_DIR参数可修改，此参数目的和上面transformer.sh中描述的一样, 现在默认设置的是tmp2_ex3。
      3、pretrained参数指定了使用的预训练参数，就是上面代码摘要得到的参数，我设置的是train_CSN_py_1，需要事先将tmp_CSN_py_1目录下的train_CSN_py_1.mdl复制到MODEL_DIR参数指定的目录下。因为我已经跑过了，所以tmp2_ex3目录已存在，且
    已复制train_CSN_py_1.mdl到此处，也产生了最新的代码搜索的模型参数，即code_search_clip.mdl。（code_search_clip.mdl.checkpoint也是保存的模型参数，只是和code_search_clip.mdl保存的内容有区别，具体可见model_clip.py第301行到335行）
    ```

## 文件解释

总体代码是在[NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum)基础修改的。

* c2nl目录
* data目录
  * 来自codeSearchNet-py数据集
  * 此处的dev/test/train目录由convert_dataset.py根据`../../../source_CSN_dataset`生成，具体见其代码。source_CSN_dataset就是codeSearchNet-py数据集的原始数据。
* main目录
* scripts目录
  * codeSearchNet-py子目录下是预训练和代码搜索训练的运行脚本（有个ipynb可忽略，这是之前做成的代码搜索演示，没有再次修改了）
* tmp_CSN_py_1目录
  * 代码摘要预训练保存的目录，删掉了多余的log等文件，只留下了后面代码搜索训练需要加载的预训练参数train_CSN_py_1.mdl。
* tmp2_ex3目录
  * 最新代码搜索训练保存的目录
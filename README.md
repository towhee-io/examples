# Towhee Examples
<p align="center">
  <a href="https://github.com/towhee-io/towhee">
    <img src="https://github.com/towhee-io/towhee/raw/main/towhee_logo.png#gh-light-mode-only" alt="Logo">
  </a>
  <p align="center" style="padding-left: 10px; padding-right: 10px">
      Towhee Examples are used to analyze the unstructured data with towhee, such as reverse image search, reverse video search, audio classification, question and answer systems, molecular search, etc.
    <br />
    <br />
    <a href="https://github.com/towhee-io/towhee/issues">Report Bug or Request Feature</a>
  </p>
</p>


## About Towhee Examples

x2vec, [Towhee](https://github.com/towhee-io/towhee) is all you need! Towhee can generate embedding vectors via a pipeline of ML models and other operations. It aims to make democratize `x2vec`, allowing everyone - from beginner developers to large organizations - to generate dense embeddings with just a few lines of code.



There are many interesting examples that use Towhee to process various unstructured data like images, audio, video, etc. You can easily run these examples on your machine.

## Funny Example List

<table>
    <tr>
        <td><b></b></td>
        <td width="60%"><b>Bootcamp</b></td>
        <td><b>Operators</b></td>
    </tr>
    <tr>
        <td rowspan="1">Getting Started</td>
        <td ><a href="getting_started">Getting Started with Towhee</a>
             <p>An introduction to DataCollection, which can help you better learn the data processing pipeline with Towhee.</p>
        </td>
        <td >
            <a></a>
        </td>
    </tr>
    <tr>
        <td rowspan="5">Image</td>
        <td ><a href="image/reverse_image_search">Reverse Image Search</a><br />
             <p>Search for images that are similar or related to the input image, it supports a lot of models such as ResNet, VGG, EfficientNet, ViT, etc.</p>
        </td>
        <td >
            <a href="https://towhee.io/operators?limit=30&page=1&filter=1%3Aimage-embedding">Image Embedding</a><br />
            <a href="https://towhee.io/image-embedding/timm">Timm</a><br />
        </td>
    </tr>
    <tr>
        <td >
            <a href="image/image_animation">Image Animation</a><br />
            <p>Convert an image into an animated image.</p>
        </td>
        <td >
            <a href="https://towhee.io/img2img-translation/animegan">Animegan</a><br />
            <a href="https://towhee.io/img2img-translation/cartoongan">Cartoongan</a><br />
        </td>
    </tr>
    <tr>
        <td >
            <a href="image/image_deduplication">Image Deduplication</a><br />
            <p>Find exact or near-exact duplicates within a collection of images.</p>
        </td>
        <td >
            <a href="https://towhee.io/towhee/image-decode">Image Decode</a><br />
            <a href="https://towhee.io/image-embedding/timm">Timm</a><br />
        </td>
    </tr>
    <tr>
        <td >
            <a href="image/text_image_search">Text Image Search</a><br />
            <p>Returns images related to the description of the input query text, which is cross-modal retrieval.</p>
        </td>
        <td >
            <a href="https://towhee.io/towhee/clip">CLIP</a><br />
        </td>
    <tr>
        <td >
            <a href="image/visualization">Visualization</a><br />
            <p>Under the hood: Embedding models and ANNS indexes in image search.</p>
        </td>
        <td >
            <a href="https://github.com/towhee-io/towhee/tree/main/towhee/models/visualization">embedding_visualization</a><br />
        </td>
    </tr>
    <tr>
        <td rowspan="2">NLP</td>
        <td ><a href="nlp/question_answering">Q&A System</a><br />
             <p>Process user questions and give answers through natural language technology.</p>
        </td>
        <td >
            <a href="https://towhee.io/operators?limit=30&page=1&filter=1%3Atext-embedding">Text Embedding</a><br />
            <a href="https://towhee.io/text-embedding/dpr">DPR</a><br />
        </td>
    </tr>
    <tr>
        <td >
            <p>Recommendation System(TBD)</p>
            <p>Predict the "rating" or "preference" a user would give to an item.</p>
        </td>
        <td >
            <a></a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">Video</td>
        <td ><a href="video/reverse_video_search">Reverse Video Search</a><br />
             <p>It takes a video as input to search for similar videos.</p>
        </td>
        <td >
            <a href="https://towhee.io/video-classification?index=1&size=30&type=2">Video Classification</a><br />
            <a href="https://towhee.io/video-classification/pytorchvideo">Pytorchvideo</a><br />
        </td>
    </tr>
    <tr>
        <td >
            <a href="video/video_tagging">Video Classification</a>
            <p>Video Classification is the task of producing a label that is relevant to the video given its frames.</p>
        </td>
        <td >
            <a href="https://towhee.io/video-classification?index=1&size=30&type=2">Video Classification</a><br />
        </td>
    </tr>
    <tr>
        <td >
            <a href="video/text_video_retrieval">Text Video Search</a><br />
            <p>Search for similar or related videos with the input text.</p>
        </td>
        <td >
            <a href="https://towhee.io/towhee/clip4clip">CLIP4Clip</a><br />
        </td>
    </tr>
    <tr>
        <td rowspan="1">Audio</td>
        <td ><a href="audio/audio_classification">Audio Classification</a></br>
             <p>Categorize certain sounds into certain categories, such as ambient sound classification and speech recognition.</p>
        </td>
        <td >
            <a href="https://towhee.io/operators?limit=30&page=1&filter=1%3Aaudio-classification">Audio Classification</a>
        </td>
    </tr>
    <tr>
        <td rowspan="1">Medical</td>
        <td ><a href="medical/molecular_search">Molecular Search</a>
             <p>Search for similar molecular formulas based on the Tanimoto metric, and also supports searching for substructures and superstructures.</p>
        </td>
        <td >
            <a href="https://towhee.io/molecular-fingerprinting/rdkit">RDKit</a>
        </td>
    </tr>
    <tr>
        <td rowspan="1">Data Science</td>
        <td ><a href="data_science/credit_card_approval_prediction">Credit Card Approval Prediction</a>
             <p>Predict whether the bank issues a credit card to the applicant, and the credit scores can objectively quantify the magnitude of risk.</p>
        </td>
        <td >
            <p>Logistic Regression</p>
          	<p>Decision Tree</p>
          	<p>SVM</p>
        </td>
    </tr>
        <tr>
        <td rowspan="1">Training</td>
        <td ><a href="fine_tune">Fine Tune</a></br>
             <p>Tutorial about how to fine tuen with towhee.</p>
        </td>
        <td >
            <a href="https://towhee.io/operators?limit=30&page=1&filter=1%3Aimage-embedding">Image Embedding</a>
        </td>
    </tr>
</table>




## Contributing

Contributions to Milvus Bootcamp are welcome from everyone. See [Guidelines for Contributing](https://github.com/towhee-io/towhee/blob/main/CONTRIBUTING.md) for details.

## Support

Join the Towhee community on [Slack](https://join.slack.com/t/towheeio/shared_invite/zt-19xhoo736-PhIYh~hwOBsDSy5ZvGWJxA) to give feedback, ask for advice, and direct questions to the engineering team. You can also submit [Issues](https://github.com/towhee-io/towhee/issues) or join [Discussions](https://github.com/towhee-io/towhee/discussions).

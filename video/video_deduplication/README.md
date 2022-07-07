# Video Deduplication

Video Deduplication, also known as Video Copy Detection or Video Identification by Fingerprinting, means that given a query video, you need to find or retrieval the videos with the same content with query video.

Due to the popularity of Internet-based video sharing services, the volume of video content on the Web has reached unprecedented scales. Besides copyright protection, a video copy detection system is important in applications like video classification, tracking, filtering and recommendation.  

The problem is particularly hard in the case of content-based video retrieval, where, given a query video, one needs to calculate its similarity with all videos in a database to retrieve and rank the videos based on relevance. However, using Milvus and Towhee can help you build a Video Deduplication System easily.

## Learn from Notebook

- [Getting started](./1_video_deduplication_engine.ipynb)

In this notebook you will get prerequisites, build and use a basic Video Deduplication system, visualize sample results, and measure the system with performance metrics.

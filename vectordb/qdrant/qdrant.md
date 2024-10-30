# Qdrant: Vector database
- Qdrant is a vector similarity search engine that provides a production-ready service with a convenient API to store, search, and manage points (i.e. vectors) with an additional payload.

## Vector Databases?
- Vector databases are a type of database designed to store and query high-dimensional vectors efficiently.
- Vector databases are optimized for storing and querying these high-dimensional vectors efficiently, and they often using specialized data structures and indexing techniques such as Hierarchical Navigable Small World (HNSW) and Product Quantization, among others.

## Benefits of using vector databases
1. Efficient storage and indexing of high-dimensional data.
2. Ability to handle large-scale datasets with billions of data points.
3. Support for real-time analytics and queries.
4. Ability to handle vectors derived from complex data types such as images, videos, and natural language text.
5. Improved performance and reduced latency in machine learning and AI applications.
6. Reduced development and deployment time and cost compared to building a custom solution.

## High-Level Overview of Qdrant’s Architecture
![image](https://github.com/user-attachments/assets/f15dff37-bef1-4d97-ab63-43ef28e8351b)

- Collections: A collection is a named set of points (vector with a payload) among which you can search.
- Distance Metrics: These are used to measure similarities among vectors and they must be selected at the same time you are creating a collection.
- Points: The points are the central entity that Qdrant operates with and they consist of a vector and an optional id and payload.
    - id: a unique identifier for your vectors.
    - Vector: a high-dimensional representation of data, for example, an image, a sound, a document, a video, etc.
    - Payload: A payload is a JSON object with additional data you can add to a vector.
- Storage:
    - In-memory Storage: Store all vectors in RAM.
    - Memmap Storage: Create a virtual address sapce associated with the file on disk.
- Clients: The programming languages you can use to connect to Qdrant.
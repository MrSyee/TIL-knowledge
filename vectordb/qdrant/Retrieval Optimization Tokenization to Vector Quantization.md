# Retrieval Optimization: Tokenization to Vector Quantization.
- Lecture: https://learn.deeplearning.ai/courses/retrieval-optimization-from-tokenization-to-vector-quantization/lesson/1/introduction

## Introduction

- Vector Databases use specialized data structures to approximate the search of nearest neighbors.
- HNSW
    - Hierarchical
    - Navigable
    - Small words
- Product quantization
- Scalar quantization
- Binary quantization

## Embedding models

## Role of the tokenizers

- Encoding
    - BPE - Byte-Pair Encoding
      <img width="981" alt="image" src="https://github.com/user-attachments/assets/8243c522-a5bf-4410-9628-3dac6290a92f">

    - WordPiece
      <img width="959" alt="image" src="https://github.com/user-attachments/assets/80eacc58-d54c-4236-b757-31cf85d2f59f">

    - Unigram
      <img width="984" alt="image" src="https://github.com/user-attachments/assets/9485a596-1d16-46e9-adbb-c9394892e3e2">

- SenetencePiece
    - Implementation of the same tokenization algorithm
    - BPE or Unigram


## Practical implications of the tokenization

### SentenceTransformer
- Unknown tokens
    - emoji -> regular letters
        - "I feel 😊" -> "I feel happy"
- Identifiers

```
sbert_tokenizer.encode("Broadcom BCM2712").tokens
>>>
['[CLS]', 'broad', '##com', 'bc', '##m', '##27', '##12', '[SEP]']
```

- Typos

```
sentences = [
    "Great accommodation",
    "Great acommodation",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)

sentences = [
    "Great accommodation",
    "Great acommodation",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)
sentences = [
    "Great accommodation",
    "Great acommodation",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)

>>>
['[CLS]', 'great', 'accommodation', '[SEP]']
['[CLS]', 'great', 'ac', '##om', '##mo', '##dation', '[SEP]']
```

- Numerical values and date/time

```
sentences = [
    "16th February 2024",
    "2024-02-16",
    "17th February 2024",
    "18th February 2024",
    "19th February 2024",
    "20th February 2024",
    "15th February 2024",
]
for sentence in sentences:
    print(sbert_tokenizer.encode(sentence).tokens)

>>>
['[CLS]', '16th', 'february', '202', '##4', '[SEP]']
['[CLS]', '202', '##4', '-', '02', '-', '16', '[SEP]']
['[CLS]', '17th', 'february', '202', '##4', '[SEP]']
['[CLS]', '18th', 'february', '202', '##4', '[SEP]']
['[CLS]', '19th', 'february', '202', '##4', '[SEP]']
['[CLS]', '20th', 'february', '202', '##4', '[SEP]']
['[CLS]', '15th', 'february', '202', '##4', '[SEP]']
```

### tiktoken
- OpenAI의 임베딩 라이브러리
- 위 오류 케이스에 대해서 대응 가능하다.

### Vector search in practice (w/ Qdrant)
- Qdrant로 벡터 검색 실습

## Measuring Search Relevance
- RAG application의 검색 퀄리티를 올리기 위한 강의여서 스킵함.

## Optimizing HNSW search
- HNSW는 vector DB의 기초가 되는 알고리즘이다.
- To approximate the nearest-neighbor search.

### HNSW structure
- Hierarchical Navigable Samll Worlds is the most commonly algorithms to approximate nearest neighbor search.

- `m` parameters
    - Define how many edges should each node have.
    - 값이 증가할수록
        - search precision 증가
        - latency에 영향을 준다. (성능, 속도 tradeoff)
- `ef` closest points
    - query로부터 가까운 포인트를 `ef`개 찾는다.
    - 다음 layer로 가서 동일한 작업을 반복

### 실습

- ANN search

```
client.search(
    "wands-products",
    query_vector=models.NamedVector(
        name="product_name",
        vector=model.encode(queries_df.loc[0, "query"])
    ),
    limit=3,
    with_vectors=False,
    with_payload=False,
)
```

- kNN search
```
client.search(
    "wands-products",
    query_vector=models.NamedVector(
        name="product_name",
        vector=model.encode(queries_df.loc[0, "query"])
    ),
    limit=3,
    with_vectors=False,
    with_payload=False,
    search_params=models.SearchParams(
        exact=True,  # Turns on the exact search mode
    ),
)
```

- 파라미터 m 과 ef를 높히면 정확도가 올라간다. 대신 속도는 느려짐.

## Vector quantization
- optimization
    - reducing memory

### Product Quantization(PQ)
- 고차원의 벡터를 서브벡터로 나눈다.
- indexing 시간은 늘어난다.

<img width="884" alt="image" src="https://github.com/user-attachments/assets/30d8fc75-4571-4158-b628-3f6a3cfbfe18">

- x16(4D clusters) 부터 search time이 줄어든다.


### Rescoring
- qunatized vertor들을 탐색할 때, 많은 documents 들이 같은 representation을 갖는다.
- VectorDB는 original vertors를 그대로 두고 그 벡터의 결과들을 rescore 한다.


### Scalar Quantization(SQ)
- 각 벡터 값의 타입을 변경한다. e.g.  float -> int
- indexing and search time을 줄인다. 대신 precision도 줄어든다. 그런데 precision이 아주 약간밖에 안 줄어든다.
<img width="978" alt="image" src="https://github.com/user-attachments/assets/9762a659-a11c-4e02-9145-61181f495062">

### Binary Qunatization(BQ)
- 메모리를 아끼기 위해 float 값을 boolean 값으로 변경한다. float -> single bit
- precision이 크게 줄어들지만, rescoring 이후에는 꽤 높은 precision을 보여준다.
<img width="1010" alt="image" src="https://github.com/user-attachments/assets/c80f612b-8a5b-44e1-ba54-71d2bee68c7c">

### Summary
- precision: SQ == SQ + rescoring > BQ + rescoring > PQ + rescoring
- speed: BQ > SQ > PQ


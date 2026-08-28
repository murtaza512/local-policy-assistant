import keyword
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("======= Start INGESTION ===========")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
  path=str(PROJECT_ROOT / "chroma_db")
)

collection = client.get_or_create_collection(
  name="company_policies",
  metadata={"hnsw:space": "cosine"}
)

def metadata_from_heading(heading):
  heading = heading.lower()

  metadata = {
    "section": heading
  }

  if "enterprise" in heading:
    metadata["plan"] = "enterprise"

  if "standard" in heading:
    metadata["plan"] = "standard"

  if "premium" in heading:
    metadata["plan"] = "premium"

  if "refund" in heading:
    metadata["policy_type"] = "refund"

  if "cancellation" in heading:
    metadata["policy_type"] = "cancellation"

  if "vacation" in heading:
    metadata["policy_type"] = "vacation"

  return metadata

def chunk_by_heading(text):
  sections = []
  current_lines = []
  current_heading = None

  for line in text.splitlines():
    line = line.strip()

    if not line:
      continue

    if line.startswith("#"):
      if current_heading is not None:
        sections.append({
          "id": f"Chunk {len(sections) + 1}",
          "text": " ".join(current_lines),
          "metadata": metadata_from_heading(current_heading)
        })

      current_heading = line.lstrip("#").strip()
      current_lines = []
    else:
      current_lines.append(line)

  if current_heading is not None:
    sections.append({
      "id": f"Chunk {len(sections) + 1}",
      "text": " ".join(current_lines),
      "metadata": metadata_from_heading(current_heading)
    })

  return sections

def ingestion_pipeline():
  #  check if we already ingested the data
  print("======= LOAD chunks with metadata and their embeddings if exists already ===========")

  if collection.count() != 0:
    print("Data already ingested")
    return

  print("======== READ From policies ====== 1")
  data = (PROJECT_ROOT / "data/policies.txt").read_text()

  print("======== Create chunks from Policies ====== 2")
  chunks = chunk_by_heading(data)
  print(f"Chunks {len(chunks)}")

  print("======== Create chunks_embeddings  ====== 3")

  chunks_text = [
    f"{chunk['metadata']['section']}: {chunk['text']}"
    for chunk in chunks
  ]

  embeddings = model.encode(chunks_text)

  print(embeddings.shape)

  print("======= SAVE chunks with metadata and their embeddings ===========")
  print("Ingesting to chromadb")
  collection.upsert(
    ids=[
      chunk["id"]
      for chunk in chunks
    ],
    documents=[
      chunk["text"]
      for chunk in chunks
    ],
    metadatas=[
      chunk["metadata"]
      for chunk in chunks
    ],
    embeddings=embeddings.tolist()
  )

ingestion_pipeline()

print("======= END INGESTION ===========")


print("======= RETREIVAL ===========")

def retrieve(question, k=2, where=None):
  question_embedding = model.encode(question)

  query_args = {
    "query_embeddings": [question_embedding.tolist()],
    "n_results": k
  }

  if where is not None:
    query_args["where"] = where


  results = collection.query(**query_args)

  ids = results['ids'][0]
  documents = results['documents'][0]
  metadatas = results['metadatas'][0]
  distances = results['distances'][0]


  retrieved = []
  for id, text, metadata, distance in zip(ids, documents, metadatas, distances):

    retrieved.append({
      "id": id,
      "text": text,
      "metadata": metadata,
      "distance": float(distance),
      "similarity": round((1 - float(distance)), 4)
    })

  return retrieved


print("=======rank BM25 =======")
data = (PROJECT_ROOT / "data/policies.txt").read_text()
chunks = chunk_by_heading(data)
chunk_ids = [chunk["id"] for chunk in chunks]
chunk_sections = [chunk["metadata"]["section"] for chunk in chunks]
print("Cjunks_IDS", chunk_ids, chunk_sections)

documents = [
  chunk["text"]
  for chunk in chunks
]

tokenized_documents = [
  doc.lower().split()
  for doc in documents
]

bm25 = BM25Okapi(tokenized_documents)

def keyword_retrieve(question, k=3):
  query_tokens = question.lower().split()

  scores = bm25.get_scores(query_tokens)

  ranked_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
  )

  results = []

  for index in ranked_indices[:k]:
    results.append({
      "id": chunks[index]["id"],
      "text": chunks[index]["text"],
      "metadata": chunks[index]["metadata"],
      "keyword_score": float(scores[index])
    })

  return results

def reciprocal_rank_fusion(
  vector_results,
  keyword_results,
  constant=60
):
  scores = {}
  items = {}

  for rank, result in enumerate(vector_results, start=1):
    chunk_id = result["id"]

    scores[chunk_id] = (
      scores.get(chunk_id, 0)
      + 1 / (constant + rank)
    )

    items[chunk_id] = result

  for rank, result in enumerate(keyword_results, start=1):
    chunk_id = result["id"]

    scores[chunk_id] = (
      scores.get(chunk_id, 0)
      + 1 / (constant + rank)
    )

    items[chunk_id] = result

  ranked_ids = sorted(
    scores,
    key=scores.get,
    reverse=True
  )

  results = []

  for chunk_id in ranked_ids:
    result = items[chunk_id].copy()
    result["rrf_score"] = scores[chunk_id]

    results.append(result)

  return results

def hybrid_retrieve(question, k=3, candidate_k=3):
  vector_result = retrieve(question, k=candidate_k)
  keyword_result = keyword_retrieve(question, k=candidate_k)

  fusion = reciprocal_rank_fusion(
    vector_result,
    keyword_result
  )

  return fusion[:k]

print("==========LLM PART=====================")

from transformers import pipeline

MIN_SCORE = 0.5

def filter_results(results, min_score=MIN_SCORE):
  return [
    result
    for result in results
    if result['similarity'] >= min_score
  ]

def build_context(results):
  parts = []

  for result in results:
    part = (
      f"SECTION: {result['metadata']['section']}\n"
      f"{result['text']}"
    )

    parts.append(part)

  return "\n\n".join(parts)

generator = pipeline(
  task="text-generation",
  model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
  dtype="auto",
  device_map="auto"
)

def generate_answer(question, retrieved_results):
  if not retrieved_results:
    return "I don't know based on the provided company policies."

  context = build_context(retrieved_results)

  messages = [
    {
      "role": "system",
      "content": (
        "You are a company policy assisstant. "
        "Answer using only the supplied context. "
        "If the context does not contain the answer, say: "
        "\"I don't know based on the provided company policies.\""
      )
    },
    {
      "role": "user",
      "content": f"""
You must Answer using only the supplied context.
Find the sentence in the context that directly answers the question.
if direct answer exists, return that answer,
if no direct answer, return that exaclty:
I don't know based on the provided company policies.

CONTEXT:
{context}

QUESTION:
{question}

important:
The answer may be written using slightly different wording than the question.

Return one short sentence only.
""".strip()
    }
  ]

  output =  generator(
    messages,
    max_new_tokens=80,
    do_sample=False # To get prictable answer and grounded output
  )

  return output[0]["generated_text"][-1]["content"]


def answer_question(question, k=2, min_score=0.5, where=None):
  results = retrieve(question, k=k, where=where)

  # print("Retrieved", results)
  filtered_results = filter_results(results, min_score=min_score)
  if not filtered_results:
    return "I don't know based on the provided company policies."

  answer = generate_answer(
    question,
    filtered_results
  )

  return {
    "answer": answer,
    "sources": filtered_results
  }

# question = "What is the Enterprise refund policy?"
# question_1 = "How can corporate customers get their money back?"
question_1 = "What does policy REF-ENT-004 say?"

# print("Enterprice: ", answer_question("What is the refund processing period for Enterprise customers in Germany?", min_score=0.3, where={"plan": {"$eq": "enterprise"}}))
# print("Enterprise 2: ", answer_question("How long does an Enterprise refund take in Germany?"))
# print("question_3: What is company mat", answer_question("What is company's maternity policy?"))

# question =  "How long does an Enterprise refund take in Germany?"
# vector_result = retrieve(question_1, k=3)
# keyword_result = keyword_retrieve(question_1, k=3)

# print("VECTOR results")
# for result in vector_result:
#   print(
#     result["similarity"],
#     result["metadata"]["section"]
#   )

# print("\n KEYWORD results")
# for result in keyword_result:
#   print(
#     round(result["keyword_score"], 4),
#     result["metadata"]["section"]
#   )

# print("\n RRF Reciprocal rank fusion")

# hybrid_results = reciprocal_rank_fusion(vector_result, keyword_result)

# for result in hybrid_results:
#   print(
#     round(result["rrf_score"], 5),
#     result["metadata"]["section"]
#   )


print("=========== Evaluate reteriever =========")

evaluation_queries = [
  {
    "question": "How long does an Enterprise refund take in Germany?",
    "relevant_chunks": {"Chunk 1"}
  },
  {
    "question": "What does policy REF-ENT-004 say?",
    "relevant_chunks": {"Chunk 1"}
  },
  {
    "question": "How many vacation days do employee in Germany get?",
    "relevant_chunks": {"Chunk 4"}
  },
  {
    "question": "What is the Premium cancellation peroid?",
    "relevant_chunks": {"Chunk 3"}
  }
]

def result_ids(results):
  return [result["id"] for result in results]

def precision_at_k(case, k):
  top_k = set(case['retrieved_chunks'][:k])
  relevant = case["relevant_chunks"]

  found = top_k.intersection(relevant)

  return len(found) / k

def recall_at_k(case, k):
  top_k = set(case['retrieved_chunks'][:k])
  relevant = case["relevant_chunks"]

  found = top_k.intersection(relevant)

  return len(found) / len(relevant)


def reciprocal_rank(case):
  relevant = case["relevant_chunks"]

  for rank, chunk in enumerate(
    case["retrieved_chunks"],
    start=1
  ):
    if chunk in relevant:
      return 1 / rank

  return 0

def evaluate_retriever(retriever, queries, k=3):
  test_cases = []

  for case in queries:
    results = retriever(case["question"], k=k)

    test_cases.append({
      "question": case["question"],
      "relevant_chunks": case["relevant_chunks"],
      "retrieved_chunks": result_ids(results)
    })

  precision_scores = [
    precision_at_k(case, k)
    for case in test_cases
  ]

  recall_scores = [
    recall_at_k(case, k=k)
    for case in test_cases
  ]

  rr_scores = [
    reciprocal_rank(case)
    for case in test_cases
  ]

  return {
    "precision_at_k": sum(precision_scores) / len(precision_scores),
    "recall_at_k": sum(recall_scores) / len(recall_scores),
    "mrr": sum(rr_scores) / len(rr_scores)
  }

vector_metrics = evaluate_retriever(
  retrieve,
  evaluation_queries,
  k=3
)

bm25_metrics = evaluate_retriever(
  keyword_retrieve,
  evaluation_queries,
  k=3
)

hybrid_metrics = evaluate_retriever(
  hybrid_retrieve,
  evaluation_queries,
  k=3
)

print("vector_metrics", vector_metrics)
print("bm25_metrics", bm25_metrics)
print("hybrid_metrics", hybrid_metrics)

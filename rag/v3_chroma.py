import json
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
  print("question_embedding",question_embedding.ndim)

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


print("==========LLM PART=====================")

import torch
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

question = (
  "How long does an Enterprise refund "
  "take in Germany?"
)

print("Enterprice: ", answer_question("What is the refund processing period for Enterprise customers in Germany?", min_score=0.3, where={"plan": {"$eq": "enterprise"}}))
print("Enterprise 2: ", answer_question("How long does an Enterprise refund take in Germany?"))
print("question_3: What is company mat", answer_question("What is company's maternity policy?"))

# question =  "How long does an Enterprise refund take in Germany?"
# print(retrieve(question))
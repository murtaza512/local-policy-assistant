import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

print("======= Start INGESTION ===========")
STORAGE_DIR = Path('storage')
STORAGE_DIR.mkdir(exist_ok=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

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

def load_chunks():
  chunks_path = STORAGE_DIR / "chunks.json"
  if not chunks_path.exists():
    return None
  with open(chunks_path) as file:
    return json.load(file)

def load_embeddings():
  embeddings_path = STORAGE_DIR / "embeddings.npy"
  if not embeddings_path.exists():
    return None
  return np.load(embeddings_path)

def save_chunks(chunks):
  with open(STORAGE_DIR / "chunks.json", "w") as file:
    json.dump(chunks, file, indent=2)

def save_embeddings(embeddings):
  np.save(
    STORAGE_DIR / "embeddings.npy",
    embeddings
  )

def ingestion_pipeline():
  #  check if we already ingested the data
  print("======= LOAD chunks with metadata and their embeddings if exists already ===========")

  chunks = load_chunks()
  embeddings = load_embeddings()

  if chunks is not None and embeddings is not None:
    print("Chunks alread exists, No need to generate and ingest again")
    return {
      "chunks": chunks,
      "embeddings": embeddings
    }

  print("======== READ From policies ====== 1")
  data = Path("data/policies.txt").read_text()

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
  save_chunks(chunks)
  save_embeddings(embeddings)

  return {
    "chunks": chunks,
    "embeddings": embeddings
  }

ingestion_pipeline()

print("======= END INGESTION ===========")


print("======= RETREIVAL ===========")


def cosine_similarity(a, b):
  return (np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b)
  ))

def retrieve(question, k=2):
  question_embedding = model.encode(question)

  chunks_embeddings = ingestion_pipeline()
  chunks = chunks_embeddings["chunks"]
  embeddings = chunks_embeddings["embeddings"]

  scores = []
  for index, embedding in enumerate(embeddings):
    score = cosine_similarity(question_embedding, embedding)
    scores.append((score, index))

  scores.sort(reverse=True)

  results = []
  for score, index in scores[:k]:
    results.append({
      "heading": chunks[index]["metadata"]["section"],
      "text": chunks[index]["text"],
      "score": float(score)
    })

  return results


print("==========LLM PART=====================")

import torch
from transformers import pipeline

MIN_SCORE = 0.5

def filter_results(results, min_score=MIN_SCORE):
  return [
    result
    for result in results
    if result['score'] >= min_score
  ]

def build_context(results):
  parts = []

  for result in results:
    part = (
      f"SECTION: {result['heading']}\n"
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


def answer_question(question, k=2, min_score=0.5):
  results = retrieve(question, k=k)
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


print("Enterprice: ", answer_question("What is the refund processing period for Enterprise customers in Germany?"))
print("Enterprise 2: ", answer_question("How long does an Enterprise refund take in Germany?"))
print("question_3: What is company mat", answer_question("What is company's maternity policy?"))

# question =  "How long does an Enterprise refund take in Germany?"
# print(retrieve(question))
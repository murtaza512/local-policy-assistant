from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model = SentenceTransformer("all-MiniLM-L6-v2")

test = (PROJECT_ROOT / "data/policies.txt").read_text()

# print(test)

# step 2: split the text into chunks

def chunk_text(text, chunk_size=30, overlap=5):
  words = text.split()

  chunks = []

  start = 0

  while start < len(words):
    end = start + chunk_size
    chunk = words[start:end]

    chunks.append(" ".join(chunk))

    start = end - overlap

  return chunks

# chunks = chunk_text(test)
# print(len(chunks))

def cosine_similarity(a, b):
  return (np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b)
  ))

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
          "heading": current_heading,
          "text": " ".join(current_lines)
        })

      current_heading = line.lstrip("#").strip()
      current_lines = []
    else:
      current_lines.append(line)

  if current_heading is not None:
    sections.append({
      "heading": current_heading,
      "text": " ".join(current_lines)
    })

  return sections

chunks = chunk_by_heading(test)

chunks_text = [
  f"{chunk['heading']}: {chunk['text']}"
  for chunk in chunks
]

chunks_embeddings = model.encode(chunks_text)

print(chunks_embeddings.shape)

question = "How long does an Enterprice refund take in Germany?"

question_embedding = model.encode(question)

print(question_embedding.shape)
print(question_embedding.ndim)

def retrieve(question, k=2):
  question_embedding = model.encode(question)

  scores = []
  for index, embedding in enumerate(chunks_embeddings):
    score = cosine_similarity(question_embedding, embedding)
    scores.append((score, index))

  scores.sort(reverse=True)

  results = []
  for score, index in scores[:k]:
    results.append({
      "heading": chunks[index]["heading"],
      "text": chunks[index]["text"],
      "score": float(score)
    })

  return results


question_1 =  "How long does an Enterprise refund take in Germany?"
question_2 = "What is Standard Refund Policy?"

results = retrieve(
  question_1,
  k=2
)

results_2 = retrieve(
  question_2,
  k=2
)


print("\n Results:")
for result in results:
  print(result['heading'])
  print(result['text'])
  print(result['score'])
  print()


def build_context(results):
  parts = []

  for result in results:
    part = (
      f"SECTION: {result['heading']}\n"
      f"{result['text']}"
    )

    parts.append(part)

  return "\n\n".join(parts)

context = build_context(results)

print(f"CONTEXT: {context}")

def build_prompt(question, context):
  return f"""
You are Company policy assistant.

Answer the user's question using only the supllied context.

Rules:
- Do not invent information.
- If the context does not contain enough information, say:
"I don't know based on the provided company policies."
- Prefer a concise direct answer.

CONTEXT:
{context}

QUESTION:
{question}


ANSWER:
""".strip()


prompt = build_prompt(question, context)

print(f"PROMPT:\n {prompt}")

print("---" * 100)
questions = [
  "How long does an Enterprice refund take in Germany?",
  "What is the company's maternity leave policy?"
]

for ques in questions:
  results = retrieve(ques, k=2)
  context = build_context(results)
  prompt = build_prompt(ques, context)

  print(f"QUESTION \n {ques} \n")

  print(f"RETRIEVED:")
  for result in results:
    print(
      f"{result['score']:.4f} | "
      f"{result['heading']}"
    )

  print(f"PROMPT: \n {prompt}")


MIN_SCORE = 0.5

def filter_results(results, min_score=MIN_SCORE):
  return [
    result
    for result in results
    if result['score'] >= min_score
  ]

print("======FILTERAITON========")

for ques in questions:
  results = retrieve(ques, k=2)
  filtered = filter_results(results)

  print(f"QUESTION \n {ques} \n")

  if not filtered:
    print("No sufficiently relevant evidence found.")
    continue

  for result in filtered:
    print(
      f"{result['score']:0.4f} | "
      f"{result['heading']}"
    )


print("==========LLM PART=====================")

import torch
from transformers import pipeline

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
print("question_2: ", question_2, answer_question(question_2))
print("question_3: What is company mat", answer_question("What is company's maternity policy?"))
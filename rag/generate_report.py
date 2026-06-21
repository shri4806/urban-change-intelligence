import json
import os
import google.generativeai as genai

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables

load_dotenv("rag/.env")

#api key
api_key = os.getenv("GEMINI_API_KEY")
print("API Key Loaded:", api_key is not None)
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Load change report

with open("change_report.json", "r") as file:
    report = json.load(file)

print("Change report loaded successfully!")

# Load embeddings model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index

vectorstore = FAISS.load_local(
    "rag/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Knowledge base loaded successfully!")

# Create query from analytics
query = f"""
{report['change_percentage']}% urban change detected.
{report['regions_detected']} regions detected.
Largest development hotspot identified.
"""

# Retrieve relevant knowledge

results = vectorstore.similarity_search(
    query,
    k=3
)

print("\nRetrieved Knowledge:\n")

for i, doc in enumerate(results):

    print(f"\n--- Result {i+1} ---\n")

    print(doc.page_content[:500])

context = "\n\n".join(
    [doc.page_content for doc in results]
)

print("\nContext created successfully!")

#prompt for gemini
prompt = f"""
You are an urban planning and remote sensing expert.

Urban Change Statistics:

Change Percentage: {report['change_percentage']}%
Regions Detected: {report['regions_detected']}
Largest Region: {report['largest_region']} pixels

Top Development Zones:
{report['top_regions']}

Relevant Urban Planning Knowledge:

{context}

Write a short professional urban change assessment.

Include:
1. Executive Summary
2. Key Findings
3. Conclusion
"""

print("\nPrompt created successfully!")

response = model.generate_content(prompt)
print("\nGemini response received successfully!")

print("\n")
print("=" * 70)
print("URBAN CHANGE INTELLIGENCE REPORT")
print("=" * 70)

print("\n")

print(response.text)

with open("urban_report.txt", "w", encoding="utf-8") as file:
    file.write(response.text)

print("\nReport saved successfully!")
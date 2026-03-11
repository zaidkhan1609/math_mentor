Evaluation Summary — Reliable Multimodal Math Mentor
Scope

This document summarizes how the Multimodal Math Mentor application satisfies the assignment requirements and how reviewers can evaluate the system.

The system implements a multimodal AI tutoring pipeline capable of solving JEE-style mathematical problems using:

RAG (Retrieval-Augmented Generation)

Multi-agent orchestration

Human-in-the-loop (HITL)

Memory and pattern reuse

1 Multimodal Input & Parsing
Criterion	Status	Notes
Image input (JPG/PNG)	✅	Users upload screenshots or photos; OCR extracts text
Extraction preview + edit	✅	"Confirm Question (Edit if needed)" allows correction
Audio input	✅	Audio uploaded and transcribed using Whisper-style ASR
Text input	✅	Direct typed questions supported
Low confidence → HITL	✅	Parser/verifier can flag ambiguous input requiring human verification
2 Parser Agent
Criterion	Status	Notes
Clean OCR/ASR output	✅	Parser normalizes text and extracts math expressions
Structured output	✅	JSON format including topic, variables, constraints
Detect ambiguity	✅	needs_clarification flag when problem incomplete
HITL trigger	✅	UI warns user when parser flags ambiguity

Example structured output:

{
  "problem_text": "...",
  "topic": "probability",
  "variables": ["x"],
  "constraints": [],
  "needs_clarification": false
}
3 RAG Pipeline
Criterion	Status	Notes
Knowledge base	✅	Curated documents in knowledge_base/
Chunk → embed → vector store	✅	Implemented using embeddings + ChromaDB
Top-k retrieval	⚠️	Works locally when ChromaDB is built
Show sources in UI	✅	Sidebar displays retrieved context
No hallucinated citations	✅	If retrieval fails, solver proceeds without fake citations
4 Multi-Agent System

The system uses a 5-agent architecture.

Agent	Role
Parser Agent	Converts raw input to structured problem
Intent Router Agent	Classifies topic and routes workflow
Solver Agent	Solves problem using symbolic math and RAG
Verifier Agent	Checks correctness and flags uncertainty
Explainer Agent	Generates step-by-step student explanation

Pipeline:

Input
 ↓
Parser Agent
 ↓
Intent Router
 ↓
RAG Retriever
 ↓
Solver Agent
 ↓
Verifier Agent
 ↓
Explainer Agent
5 Application UI

The Streamlit interface includes:

Feature	Status
Input selector (Text / Image / Audio)	✅
OCR / transcript preview	✅
Agent trace visualization	✅
Retrieved context panel	✅
Final answer and explanation	✅
Confidence indicator	✅
Feedback buttons	✅

Feedback options:

✅ Correct
❌ Incorrect + comment
6 Deployment
Criterion	Status
Deployed application	✅
Reviewer-accessible link	✅

Deployment:

HuggingFace Space

https://huggingface.co/spaces/zaidkhan/math_mentor

The application can also run locally using Streamlit.

7 Human-in-the-Loop (HITL)

HITL is triggered when:

Trigger	Implementation
Low OCR/ASR confidence	User edits extracted text
Parser ambiguity	needs_clarification flag
Verifier uncertainty	System asks for confirmation
User feedback	Stored in memory

Users can:

approve solution

edit problem

reject solution

8 Memory & Self-Learning

The system stores interaction history in memory.json.

Stored data includes:

question
parsed_problem
retrieved_context
final_answer
verifier_outcome
user_feedback

Capabilities:

Feature	Status
Store solved problems	✅
Retrieve similar problems	⚠️ basic similarity
Pattern reuse	⚠️ simple reuse
Model retraining	❌ not required

Memory currently uses simple matching and can be extended with embedding similarity.

9 Deliverables
Deliverable	Status
GitHub repository	✅
README with setup instructions	✅
Architecture diagram	✅
.env.example	✅
Deployed application	✅
Evaluation summary	✅
Demo video	To be recorded
How Reviewers Can Evaluate
1 Run Locally
git clone <repo>
pip install -r requirements.txt
streamlit run app.py
2 Test Multimodal Inputs

Type algebra or calculus problem

Upload screenshot

Upload audio question

3 Test HITL

Provide ambiguous question and confirm UI requests clarification.

4 Test RAG

Build ChromaDB locally and verify retrieved context appears.

5 Test Memory

Solve a problem and repeat similar question to observe reuse behavior.

Limitations
Limitation	Description
ChromaDB on Spaces	Vector database not pre-built
Memory similarity	Currently simple text similarity
Verifier confidence	Based on LLM reasoning rather than numerical confidence
Overall Outcome

The system successfully demonstrates:

multimodal input handling

RAG-based reasoning

multi-agent orchestration

human-in-the-loop validation

memory-based learning

This satisfies the key objectives of the Reliable Multimodal Math Mentor assignment.

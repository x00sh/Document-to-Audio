import os
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Document Parsing — Docling converts PDF/DOCX to structured Markdown,
# preserving headings, tables, and lists so the LLM gets rich input.
from docling.document_converter import DocumentConverter

# TTS + audio output
from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
import soundfile as sf
import numpy as np

# Load GOOGLE_API_KEY from .env into the environment.
# langchain-google-genai picks it up automatically; we never hardcode the key.
load_dotenv()

_ts = lambda: datetime.now().strftime("%H:%M:%S")

# Single shared LLM for both the scriptwriter and the fact-checker.
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

# --- Chunking configuration --------------------------------------------------
CHUNK_THRESHOLD = 16000
CHUNK_SIZE = 9000
CHUNK_OVERLAP = 0

# Lazy singletons — expensive to construct, so build once and reuse across calls.
_tts_pipeline = None
_doc_converter = None

def get_tts_pipeline():
    global _tts_pipeline
    if _tts_pipeline is None:
        config = PipelineConfig(
            voice="af_nicole:0.5,af_bella:0.5", # use either "af_nicole" or "af_bella"
            provider="cuda",
            model_quality="fp32",
            generation=GenerationConfig(lang="en-us"),
        )
        _tts_pipeline = KokoroPipeline(config)
    return _tts_pipeline

def get_doc_converter():
    global _doc_converter
    if _doc_converter is None:
        _doc_converter = DocumentConverter()
    return _doc_converter


# --- Document parsing helper -------------------------------------------------
def parse_document(path: str) -> str:
    """Convert a .pdf or .docx to Markdown, preserving headings, tables, and lists."""
    result = get_doc_converter().convert(path)
    return result.document.export_to_markdown()


# 1. Define the State
class PodcastState(TypedDict):
    document_path: str        # input: path to the source .pdf/.docx
    document_name: str        # input: base name used for the output audio file
    document_text: str        # Markdown text, populated by parse_document_node
    chunks: list[str]         # the document split into processable pieces
    current_chunk_index: int  # outer-loop pointer into `chunks`
    script: str               # current chunk's draft script
    feedback: str             # fact-checker critique for the current chunk
    is_factual: bool          # did the current chunk's script pass the check
    iteration_count: int      # per-chunk rewrite count (reset between chunks)
    audio_segments: list      # accumulated per-chunk audio arrays
    audio_path: str           # path to the final combined audio file
    script_segments: list[str]  # accumulated per-chunk finalized scripts


# Structured output schema for the fact-checker, so is_factual is reliable.
class FactCheck(BaseModel):
    is_factual: bool = Field(description="True if the script is faithful to the source document.")
    feedback: str = Field(description="Specific, actionable critique of any inaccuracies; brief confirmation if accurate.")


# 2. Define Node Functions
def parse_document_node(state: PodcastState):
    # Convert the document to Markdown via Docling and store in graph state.
    text = parse_document(state["document_path"])
    print(f"[{_ts()}] Parsed {len(text)} characters from {state['document_path']}")
    return {"document_text": text}


def chunk_document(state: PodcastState):
    # Short documents are a single chunk; long ones are split on natural
    # boundaries. This also initializes all outer-loop state.
    text = state["document_text"]
    if len(text) <= CHUNK_THRESHOLD:
        chunks = [text]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_text(text)

    print(f"[{_ts()}] Split into {len(chunks)} chunk(s)")
    return {
        "chunks": chunks,
        "current_chunk_index": 0,
        "iteration_count": 0,
        "feedback": "",
        "audio_segments": [],
        "script_segments": [],
    }


def generate_script(state: PodcastState):
    # Gemini writes/rewrites the script for the current chunk, using any prior feedback.
    idx = state["current_chunk_index"]
    chunk = state["chunks"][idx]
    print(
        f"[{_ts()}] Generating script for chunk {idx + 1}/{len(state['chunks'])} "
        f"(Iteration {state['iteration_count'] + 1})"
    )

    prompt = (
    f"""ROLE: You are an expert audiobook narrator and podcast scriptwriter.

    CONTEXT: You are processing a single segment of a larger document. Your output will be concatenated directly with other generated audio segments to form a continuous, seamless listening experience.

    TASK: Translate the provided text chunk into a natural, engaging spoken-word script that remains strictly faithful to the source facts.

    CONSTRAINTS:
    - CONTINUITY: Do NOT add introductions, greetings, sign-offs, or meta-commentary (e.g., "Moving on to the next section"). Start immediately with the content.
    - TTS OPTIMIZATION: Write ONLY plain spoken-word prose. Absolutely NO markdown formatting, asterisks, hash symbols, bullet points, or speaker labels.
    - PRONUNCIATION: Spell out symbols and acronyms naturally as they would be spoken (e.g., write "five dollars" instead of "$5", and "L L M" instead of "LLM" or "large language model").
    - COMPLEX DATA: Do not read raw table rows or columns verbatim. Instead, summarize table data naturally as spoken comparisons or descriptions. 
    - CODE/EQUATIONS: If the text contains programming code or math equations, break down what the code or equation does step-by-step in plain prose.

    OUTPUT FORMAT: Return ONLY the raw string of plain text ready to be read by a text-to-speech engine.

    SOURCE SECTION:{chunk}

    CRITICAL REMINDER: Ensure your final output contains absolutely zero markdown and consists entirely of continuous spoken-word prose."""
    )

    feedback = state.get("feedback")
    if feedback:
        prompt += (
            "\nA fact-checker reviewed your previous draft and found issues. "
            "Revise the script to address this critique while keeping it faithful "
            f"to the source:\n{feedback}\n"
        )

    raw = llm.invoke(prompt).content
    # Gemini 3+ returns content as a list of typed blocks; extract text blocks.
    if isinstance(raw, list):
        script = " ".join(
            b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        script = raw
    return {"script": script}


def fact_check_script(state: PodcastState):
    # Gemini compares the script against the current chunk and returns structured output.
    idx = state["current_chunk_index"]
    print(f"[{_ts()}] Checking for hallucinations (chunk {idx + 1}/{len(state['chunks'])})...")

    chunk = state["chunks"][idx]
    checker = llm.with_structured_output(FactCheck)
    prompt = (
        "You are a strict fact-checker. Compare the PODCAST SCRIPT against the "
        "SOURCE SECTION.\n"
        "- If the script adds claims absent from the source, or contradicts it, set "
        "is_factual=False and give specific, actionable feedback naming what to fix.\n"
        "- If the script is faithful to the source, set is_factual=True with a brief "
        "confirmation.\n\n"
        f"SOURCE SECTION:\n{chunk}\n\n"
        f"PODCAST SCRIPT:\n{state['script']}\n"
    )

    result = checker.invoke(prompt)
    return {
        "feedback": result.feedback,
        "is_factual": result.is_factual,
        "iteration_count": state["iteration_count"] + 1,
    }


def generate_audio(state: PodcastState):
    # Synthesize audio for the current chunk's finalized script, append it to the
    # running output, and advance the outer loop to the next chunk.
    idx = state["current_chunk_index"]
    print(f"[{_ts()}] Generating audio for chunk {idx + 1}/{len(state['chunks'])}...")

    pipeline = get_tts_pipeline()
    res = pipeline.run(state["script"])
    segment = res.audio
    sample_rate = res.sample_rate

    audio_segments = state["audio_segments"] + [segment]

    # Write the running concatenation each pass so the file is complete whenever
    # the outer loop ends (no separate finalize node, matching the flow diagram).
    audio_path = f"{state['document_name']}_podcast.mp3"
    sf.write(audio_path, np.concatenate(audio_segments), sample_rate)

    # Mirror the audio pattern: accumulate finalized scripts and write the full
    # text each pass so the file is complete whenever the outer loop ends.
    script_segments = state["script_segments"] + [state["script"]]
    script_path = f"{state['document_name']}_podcast_script.txt"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(script_segments))
    print(f"[{_ts()}] Saved script → {script_path}")

    return {
        "audio_segments": audio_segments,
        "audio_path": audio_path,
        "script_segments": script_segments,
        "current_chunk_index": idx + 1,
        "iteration_count": 0,  # reset the rewrite loop for the next chunk
        "feedback": "",        # don't leak this chunk's critique into the next
    }


# 3. Conditional Router Functions
def should_rewrite(state: PodcastState):
    # Inner loop: stop after a pass OR after 3 rewrites (no infinite loops).
    if state["is_factual"] or state["iteration_count"] >= 3:
        return "generate_audio"
    return "generate_script"


def has_remaining_chunks(state: PodcastState):
    # Outer loop: process the next chunk, or finish when all are done.
    if state["current_chunk_index"] < len(state["chunks"]):
        return "generate_script"
    return END


# 4. Build the Graph
workflow = StateGraph(PodcastState)

workflow.add_node("parse_document", parse_document_node)
workflow.add_node("chunk_document", chunk_document)
workflow.add_node("generate_script", generate_script)
workflow.add_node("fact_check_script", fact_check_script)
workflow.add_node("generate_audio", generate_audio)

workflow.add_edge(START, "parse_document")
workflow.add_edge("parse_document", "chunk_document")

workflow.add_conditional_edges(
    "chunk_document",
    has_remaining_chunks,
    {"generate_script": "generate_script", END: END},
)

workflow.add_edge("generate_script", "fact_check_script")

workflow.add_conditional_edges(
    "fact_check_script",
    should_rewrite,
    {"generate_script": "generate_script", "generate_audio": "generate_audio"},
)

workflow.add_conditional_edges(
    "generate_audio",
    has_remaining_chunks,
    {"generate_script": "generate_script", END: END},
)

app = workflow.compile()


# --- Run the pipeline --------------------------------------------------------
# Point this at your source document (.pdf or .docx). Parsing and chunking now
# happen inside the graph, so we only pass the path and output name.
DOCUMENT_NAME = "dissertation"
DOCUMENT_PATH = r"C:\Users\zeesh\Documents\dissertation.docx"
# Each chunk takes several super-steps (generate -> fact-check -> [rewrites] ->
# audio), so the default recursion_limit of 25 can be hit on long documents.
# Give the graph generous headroom for many chunks.
result = app.invoke(
    {"document_path": DOCUMENT_PATH, "document_name": DOCUMENT_NAME},
    config={"recursion_limit": 200},
)

print("\nDone.")
print("Chunks processed:", len(result["chunks"]))
print("Final chunk passed fact-check:", result["is_factual"], "| rewrites on last chunk:", result["iteration_count"])
print("Audio written to:", result["audio_path"])
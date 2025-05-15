import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Load BioBERT QA model
@st.cache_resource
def load_model():
    model_name = "yikuan8/Clinical-Longformer"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

# Questions to extract from clinical trial protocols
questions = [
    "What is the objective of the study?",
    "What is the study indication?",
    "What is the investigational drug or procedure?",
    "How many participants are in the study?",
    "What is the duration of the study?",
    "What are the primary end points in the study?",
    "What are the secondary end points in the study?",
    "What are the exploratory end points in the study?",
    "What are the treatment arms or groups in the study?",
    "What are the different visits in the study?",
    "What are the different forms in the study?"
]

# PDF text extractor
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Chunking text to fit within model input
def chunk_text(text, max_words=300):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Run QA per chunk and return best answer
def answer_question_from_chunks(question, chunks):
    best_answer = {"score": 0, "answer": "Not found in the document."}
    for chunk in chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            if result["score"] > best_answer["score"] and result["answer"].strip():
                best_answer = result
        except:
            continue
    return best_answer["answer"]

# Streamlit App
st.title("🔬 Clinical Trial PDF Analyzer (Free & Medical-Aware)")

uploaded_file = st.file_uploader("📤 Upload Clinical Trial PDF", type=["pdf"])

if uploaded_file:
    st.info("📄 Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(pdf_text)

    st.info("🤖 Analyzing document with BioBERT...")
    progress_bar = st.progress(0)
    answers = {}

    total_questions = len(questions)
    for idx, q in enumerate(questions):
        ans = answer_question_from_chunks(q, chunks)
        answers[q] = ans
        progress_bar.progress((idx + 1) / total_questions)

    progress_bar.empty()
    st.success("✅ Analysis complete. See results below 👇")


    for question, answer in answers.items():
        st.markdown(f"**{question}**")
        st.markdown(f"> {answer}")

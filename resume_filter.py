import os
import heapq
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import streamlit as st

# -----------------------------
# NLTK setup (safe for Render)
# -----------------------------
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

# Download NLTK data locally if missing
try:
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
    nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
except:
    st.warning("⚠️ Could not download NLTK data — ensure nltk_data folder is included.")

# -----------------------------
# Folder paths and configs
# -----------------------------
RESUME_FOLDER = "resumes"
TOP_K = 2          # Top N resumes to select
THRESHOLD = 0.10   # Minimum similarity threshold

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Resume Filter App")
st.write("Upload resumes in the `resumes/` folder and provide a job description in `job_description.txt`.")

# -----------------------------
# Load resumes safely
# -----------------------------
def load_resumes():
    resumes = []
    file_names = []

    if not os.path.exists(RESUME_FOLDER):
        st.error(f"Folder '{RESUME_FOLDER}' not found. Please create it and upload PDF resumes.")
        return resumes, file_names

    for file in os.listdir(RESUME_FOLDER):
        if file.lower().endswith(".pdf"):
            try:
                with open(os.path.join(RESUME_FOLDER, file), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    resumes.append(text.strip().lower())
                    file_names.append(file)
            except Exception as e:
                st.warning(f"⚠️ Could not read {file}: {e}")
    return resumes, file_names

# -----------------------------
# Load job description safely
# -----------------------------
def load_job_description():
    if not os.path.exists("job_description.txt"):
        st.error("Missing 'job_description.txt' file in project root.")
        return ""
    with open("job_description.txt", "r", encoding="utf-8") as f:
        return f.read().lower()

# -----------------------------
# Extract candidate name
# -----------------------------
def extract_name(text, filename):
    if not text.strip():
        return filename.replace(".pdf", "")
    first_line = text.strip().split("\n")[0]
    return first_line if len(first_line) < 50 else filename.replace(".pdf", "")

# -----------------------------
# Main filtering function
# -----------------------------
def filter_resumes():
    st.info("⏳ Processing resumes...")

    resumes, file_names = load_resumes()
    if not resumes:
        return

    job_desc = load_job_description()
    if not job_desc.strip():
        return

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Get top K matches
    top_indices = heapq.nlargest(TOP_K, range(len(similarity_scores)), key=similarity_scores.__getitem__)

    selected, rejected = [], []

    for i, score in enumerate(similarity_scores):
        candidate_name = extract_name(resumes[i], file_names[i])
        if i in top_indices and score >= THRESHOLD:
            selected.append(f"✅ {candidate_name} — Selected (Similarity: {score:.4f})")
        else:
            rejected.append(f"❌ {candidate_name} — Rejected (Similarity: {score:.4f})")

    # Display results
    st.subheader("✅ Top Matching Resumes:")
    if selected:
        for s in selected:
            st.success(s)
    else:
        st.info("No resumes met the similarity threshold.")

    st.subheader("❌ Rejected Resumes:")
    for r in rejected:
        st.error(r)

# -----------------------------
# Run button
# -----------------------------
if st.button("Run Resume Filter"):
    filter_resumes()


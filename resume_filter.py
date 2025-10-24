import os
import heapq
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# -----------------------------
# Download necessary NLTK data
# -----------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# -----------------------------
# Folder paths and configs
# -----------------------------
RESUME_FOLDER = "resumes"
TOP_K = 2  # Top N resumes to select
THRESHOLD = 0.10  # minimum similarity to be considered selected


# -----------------------------
# Load resumes from folder
# -----------------------------
def load_resumes():
    resumes = []
    file_names = []
    for file in os.listdir(RESUME_FOLDER):
        if file.endswith(".pdf"):
            try:
                with open(os.path.join(RESUME_FOLDER, file), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    resumes.append(text.strip())
                    file_names.append(file)
            except:
                print(f"⚠️ Could not read {file}")
    return resumes, file_names


# -----------------------------
# Load Job Description
# -----------------------------
def load_job_description():
    with open("job_description.txt", "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Extract candidate name (first line of resume text)
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
    print("⏳ Processing resumes...\n")

    resumes, file_names = load_resumes()
    if not resumes:
        print("⚠️ No resumes found in folder.")
        return

    job_desc = load_job_description()

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Get top K resumes
    top_indices = heapq.nlargest(TOP_K, range(len(similarity_scores)), key=similarity_scores.__getitem__)

    print("✅ Top Matching Resumes:\n")
    selected = []
    rejected = []

    for i, score in enumerate(similarity_scores):
        candidate_name = extract_name(resumes[i], file_names[i])
        if i in top_indices and score >= THRESHOLD:
            selected.append(f"✅ {candidate_name} - Selected (Similarity: {score:.4f})")
        else:
            rejected.append(f"❌ {candidate_name} - Rejected (Similarity: {score:.4f})")

    # Print final result
    for s in selected:
        print(s)
    print("\n❌ Rejected Resumes:\n")
    for r in rejected:
        print(r)


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    filter_resumes()

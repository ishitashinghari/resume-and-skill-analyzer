import os
import re
import json
import tempfile
import logging

from dotenv import load_dotenv
from flask import Flask, render_template, request
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY (and anything else) from a .env file in the project root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

generic_words = {
    "skill", "experience", "knowledge", "ability", "team", "work",
    "responsible", "responsibility", "requirement", "qualification",
    "documentation", "position", "role", "job", "student", "graduate"
}

# Known compounds that the camelCase-repair regex would otherwise split
# into junk (e.g. "GitHub" -> "Git Hub" -> lemmatized noise like "hub").
PROTECTED_COMPOUNDS = {
    "GitHub": "GITHUBTOK",
    "LinkedIn": "LINKEDINTOK",
    "YouTube": "YOUTUBETOK",
    "PowerPoint": "POWERPOINTTOK",
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload cap

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT('all-MiniLM-L6-v2')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def fix_pdf_spacing(text):
    """
    PyPDF2 often drops spaces at column/icon/bullet boundaries, producing
    joins like 'linkedinlinkedin' or 'relevance.*May2026'. Insert spaces
    at the likely seams before any further processing.

    Known compounds (GitHub, LinkedIn, etc.) are protected from the
    camelCase-repair regex so they don't get split into junk tokens.
    """
    for original, token in PROTECTED_COMPOUNDS.items():
        text = text.replace(original, token)

    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)   # camelCase-style joins
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)    # "2026Misinformation" -> "2026 Misinformation"
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)    # "May2026" -> "May 2026"
    text = re.sub(r'[•●▪]', ' ', text)                  # bullet glyphs

    for original, token in PROTECTED_COMPOUNDS.items():
        text = text.replace(token, original)

    return text


def light_clean(text):
    """
    Whitespace-only cleanup. Preserves case and punctuation so KeyBERT
    sees natural sentence structure when extracting keyphrases.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_and_strip(text):
    """
    Full normalization for embeddings: lemmatize, lowercase, and strip
    stopwords/punctuation/short tokens. Run on already-lightly-cleaned text.

    Short tokens (len <= 2) are dropped UNLESS the original surface form
    was all-caps (tok.text.isupper()) — this preserves technical acronyms
    like "AI", "ML", "UI", "DB" that would otherwise be discarded purely
    for being short, even though they're meaningful keywords.
    """
    doc = nlp(text)
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower()
        if len(lemma) > 2 or tok.text.isupper():
            tokens.append(lemma)
    return ' '.join(tokens)


def lemmatize_phrase(phrase):
    """
    Normalize a single keyword/keyphrase to its lemma form, used only
    for set-matching (not for display).
    """
    doc = nlp(phrase)
    lemmas = [
        tok.lemma_.lower()
        for tok in doc
        if not tok.is_stop and not tok.is_punct and not tok.is_space
    ]
    joined = ' '.join(lemmas).strip()
    return joined if joined else phrase.lower().strip()


def keyword_present_in_text(keyword_lemma, full_text_tokens):
    """
    True if every word of the (lemmatized) keyword phrase appears
    somewhere in the full lemmatized resume text — not just in the
    resume's own top-N extracted keyword list.

    Falls back to prefix matching (in both directions) when there's no
    exact token match. This catches cases the lemmatizer itself misses,
    e.g. "preprocess" (job) vs "preprocessing" (resume, tagged as a noun
    so spaCy doesn't reduce it to the verb lemma), or "git" (job) vs
    "github" (resume, protected as a single compound token so it never
    lemmatizes down to "git"). A minimum length of 3 keeps this from
    firing on short/unrelated tokens.
    """
    words = [w for w in keyword_lemma.split() if w not in generic_words]
    if not words:
        return False

    for w in words:
        if w in full_text_tokens:
            continue
        if len(w) >= 3 and any(
            len(t) >= 3 and (t.startswith(w) or w.startswith(t))
            for t in full_text_tokens
        ):
            continue
        return False
    return True


def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        text = ""
    return text


def detect_resume_sections(text):
    text_lower = text.lower()
    return {
        "education": bool(re.search(r'\b(education|academic background|coursework)\b', text_lower)),
        "projects": bool(re.search(r'\b(projects?|portfolio)\b', text_lower)),
    }


def get_candidate_phrases(text, max_len=3):
    """
    Generate clean candidate phrases via spaCy noun chunks instead of
    letting KeyBERT slide an arbitrary n-gram window over raw text.
    This is what stops fragments like "ai driven user" or "tasks build ai"
    from ever becoming candidates in the first place.
    """
    doc = nlp(text)
    candidates = set()

    for chunk in doc.noun_chunks:
        phrase = " ".join(
            tok.text for tok in chunk
            if not tok.is_stop and not tok.is_punct
        ).strip().lower()
        if phrase and 1 <= len(phrase.split()) <= max_len and len(phrase) > 2:
            candidates.add(phrase)

    # Keep standalone technical tokens (acronyms, proper nouns) that
    # noun-chunking sometimes drops, e.g. "RAG", "KNN", "CLIP", "Flask"
    for tok in doc:
        if (tok.is_alpha and not tok.is_stop
                and (tok.is_upper or tok.pos_ == "PROPN")
                and len(tok.text) > 1):
            candidates.add(tok.text.lower())

    return list(candidates)


def extract_keywords(raw_text, light_cleaned_text, top_n=15):
    """
    raw_text: spacing-repaired but otherwise unprocessed text, used to
              generate noun-chunk candidates.
    light_cleaned_text: whitespace-normalized text KeyBERT scores against.
    """
    if not light_cleaned_text.strip():
        return []

    candidates = get_candidate_phrases(raw_text)
    if not candidates:
        return []

    keywords = kw_model.extract_keywords(
        light_cleaned_text,
        candidates=candidates,
        top_n=top_n,
        use_mmr=True,
        diversity=0.6
    )
    return [kw[0] for kw in keywords]


def get_ai_feedback(resume_text, job_desc_text, common_keywords, missing_keywords):
    """
    Ask Groq for qualitative feedback on the resume relative to the job
    description: what's strong, what's weak, and concrete suggestions.
    Returns a dict with keys: strengths, weaknesses, suggestions (lists
    of short strings). Returns None if no API key is configured or the
    call fails, so the rest of the app keeps working without it.
    """
    if not groq_client:
        return None

    system_prompt = (
        "You are an experienced technical recruiter and resume reviewer. "
        "Given a candidate's resume text and a job description, evaluate "
        "how well the resume positions the candidate for that specific "
        "role. Be specific and concrete — reference actual content from "
        "the resume, not generic advice. Respond with ONLY a JSON object "
        "(no markdown, no code fences, no preamble) with exactly these "
        "keys: \"strengths\" (a list of 2-4 short strings), \"weaknesses\" "
        "(a list of 2-4 short strings), \"suggestions\" (a list of 2-4 "
        "short, actionable strings). Each string should be one concise "
        "sentence."
    )

    user_prompt = (
        f"JOB DESCRIPTION:\n{job_desc_text[:4000]}\n\n"
        f"RESUME:\n{resume_text[:4000]}\n\n"
        f"Keywords already matched: {', '.join(common_keywords) or 'none'}\n"
        f"Keywords missing: {', '.join(missing_keywords) or 'none'}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_completion_tokens=700,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
        data = json.loads(raw)

        return {
            "strengths": data.get("strengths", [])[:4],
            "weaknesses": data.get("weaknesses", [])[:4],
            "suggestions": data.get("suggestions", [])[:4],
        }
    except Exception as e:
        logger.warning(f"Groq feedback failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        file = request.files.get('resume')
        job_desc = request.form.get('jobdesc', '')

        # --- validation ---
        if not file or file.filename == '':
            return render_template("index.html", error="Please choose a resume PDF.")

        if not file.filename.lower().endswith('.pdf'):
            return render_template("index.html", error="Only PDF files are supported.")

        if not job_desc.strip():
            return render_template("index.html", error="Please paste a job description.")

        # --- save to a temp file, always cleaned up afterwards ---
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)

        try:
            file.save(tmp_path)

            raw_resume_text = extract_text_from_pdf(tmp_path)
            raw_resume_text = fix_pdf_spacing(raw_resume_text)

            if not raw_resume_text.strip():
                return render_template(
                    "index.html",
                    error="Couldn't extract text from that PDF. It may be scanned/image-based."
                )

            job_desc_fixed = fix_pdf_spacing(job_desc)

            # Light cleaning preserves phrasing for KeyBERT
            resume_clean = light_clean(raw_resume_text)
            job_clean = light_clean(job_desc_fixed)

            # Full lemmatized/stopword-stripped text for semantic embeddings
            resume_text = lemmatize_and_strip(resume_clean)
            job_text = lemmatize_and_strip(job_clean)

            sections = detect_resume_sections(raw_resume_text)

            # --- semantic similarity ---
            embeddings = embedding_model.encode(
                [resume_text, job_text],
                convert_to_tensor=True
            )
            raw_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
            semantic_score = max(0.0, min(50.0, raw_sim * 50))

            # --- keyword extraction (candidate-based, not raw n-gram sliding) ---
            resume_keywords = extract_keywords(raw_resume_text, resume_clean)
            job_keywords = extract_keywords(job_desc_fixed, job_clean)

            # Full lemmatized resume text, tokenized, used as ground truth for
            # "is this skill actually present" — independent of top-N cutoffs
            resume_lemma_tokens = set(resume_text.split())

            common_keywords = []
            missing_keywords = []
            seen_lemmas = set()

            for job_kw in job_keywords:
                lemma = lemmatize_phrase(job_kw)
                if lemma in seen_lemmas:
                    continue
                seen_lemmas.add(lemma)

                if lemma in generic_words:
                    continue

                if keyword_present_in_text(lemma, resume_lemma_tokens):
                    common_keywords.append(job_kw)
                else:
                    missing_keywords.append(job_kw)

            common_keywords = sorted(common_keywords)
            missing_keywords = sorted(missing_keywords)

            keyword_score = (len(common_keywords) / len(job_keywords)) * 30 if job_keywords else 0
            keyword_score = max(0.0, min(30.0, keyword_score))

            education_bonus = 20 if sections["education"] else 0

            final_score = round(semantic_score + keyword_score + education_bonus, 2)

            # --- AI qualitative feedback (optional, needs GROQ_API_KEY) ---
            ai_feedback = get_ai_feedback(
                resume_clean, job_clean, common_keywords, missing_keywords
            )

            raw_score_components = {
                "semantic": round(semantic_score, 2),
                "keywords": round(keyword_score, 2),
                "education_bonus": education_bonus
            }

            return render_template(
                "result.html",
                score=final_score,
                raw_score_components=raw_score_components,
                common_keywords=common_keywords,
                missing_keywords=missing_keywords,
                resume_keywords=resume_keywords,
                job_keywords=job_keywords,
                has_education=sections["education"],
                has_projects=sections["projects"],
                ai_feedback=ai_feedback
            )

        finally:
            # Never keep uploaded resumes on disk longer than the request
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
import os
import math
import re
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox

FOLDER_NAME = "collection"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with"
}

# =========================================================
# TF-IDF Engine (Backend)
# =========================================================
# 1. Read all .txt files from the collection folder
documents = []
file_names = []

if not os.path.isdir(FOLDER_NAME):
    raise FileNotFoundError(
        f"Missing '{FOLDER_NAME}' folder. Please create it and add .txt files."
    )

for filename in os.listdir(FOLDER_NAME):
    if filename.endswith(".txt"):
        file_path = os.path.join(FOLDER_NAME, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())
            file_names.append(filename)

# 2. Preprocess text: remove punctuation, lowercase, stop-words, and stem
def basic_stemmer(word):
    if len(word) <= 3:
        return word

    def strip_double_consonant(stem):
        if len(stem) >= 2:
            last = stem[-1]
            if last == stem[-2] and last not in "aeiou":
                return stem[:-1]
        return stem

    if word.endswith("ing") and len(word) > 5:
        return strip_double_consonant(word[:-3])
    if word.endswith("ed") and len(word) > 4:
        return strip_double_consonant(word[:-2])
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]

    return word

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS]
    return [basic_stemmer(w) for w in filtered]

cleaned_docs = [clean_text(doc) for doc in documents]

# 3. TF-IDF computation
def compute_tf(doc):
    tf = {}
    for word in doc:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        count = tf[word]
        tf[word] = 1 + math.log10(count) if count > 0 else 0
    return tf

def compute_idf(docs):
    idf = {}
    total_docs = len(docs)
    all_words = set([word for doc in docs for word in doc])
    for word in all_words:
        containing_docs = sum([1 for doc in docs if word in doc])
        if total_docs > 0 and containing_docs > 0:
            idf[word] = math.log10(total_docs / float(containing_docs))
        else:
            idf[word] = 0
    return idf

# Compute IDF once at startup
idf_scores = compute_idf(cleaned_docs)

# 4. Main search function
def search_engine(query):
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return []
    scores = []
    for i, doc in enumerate(cleaned_docs):
        doc_tf = compute_tf(doc)
        doc_score = 0
        for word in cleaned_query:
            if word in doc_tf:
                tf_idf = doc_tf[word] * idf_scores.get(word, 0)
                doc_score += tf_idf
        if doc_score > 0:
            scores.append((doc_score, file_names[i], documents[i]))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores

def make_snippet(text, query_terms, max_len=160):
    if not text:
        return ""

    lowered = text.lower()
    first_hit = None
    for term in query_terms:
        idx = lowered.find(term)
        if idx != -1 and (first_hit is None or idx < first_hit):
            first_hit = idx

    if first_hit is None:
        snippet = text[:max_len]
        return snippet.strip() + ("..." if len(text) > max_len else "")

    start = max(first_hit - 40, 0)
    end = min(start + max_len, len(text))
    snippet = text[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"

# =========================================================
# GUI (Google Style)
# =========================================================
# Main window
root = tk.Tk()
root.title("My Python Search Engine - [Google Style]")
root.geometry("600x550")
root.configure(bg='#202124')

# Called when the user clicks Search
def highlight_terms(text_widget, terms, start_index, end_index):
    for term in terms:
        if not term:
            continue
        search_start = start_index
        while True:
            match_index = text_widget.search(
                term,
                search_start,
                stopindex=end_index,
                nocase=True
            )
            if not match_index:
                break
            match_end = f"{match_index}+{len(term)}c"
            text_widget.tag_add('highlight', match_index, match_end)
            search_start = match_end

def on_search_click():
    query = search_entry.get()

    # Clear previous results
    results_display.config(state=tk.NORMAL)
    results_display.delete('1.0', tk.END)

    if not query.strip():
        messagebox.showwarning("Warning", "Please enter a search query!")
        results_display.config(state=tk.DISABLED)
        return

    start_time = time.perf_counter()
    results = search_engine(query)
    elapsed = time.perf_counter() - start_time

    # Render results
    if results:
        header = f"Found {len(results)} relevant document(s) in {elapsed:.4f} seconds\n"
        header += "="*30 + "\n\n"
        results_display.insert(tk.END, header)
        query_terms = clean_text(query)
        original_terms = re.sub(r'[^\w\s]', '', query.lower()).split()
        for rank, (score, fname, doc) in enumerate(results, start=1):
            snippet = make_snippet(doc, query_terms)
            results_display.insert(tk.END, f"{rank}. [Score: {score:.4f}]\n", 'score')
            results_display.insert(tk.END, f"File: {fname}\n", 'fname')
            snippet_start = results_display.index(tk.INSERT)
            results_display.insert(tk.END, f"Snippet: \"{snippet}\"\n\n", 'content')
            snippet_end = results_display.index(tk.INSERT)
            highlight_terms(results_display, original_terms, snippet_start, snippet_end)
    else:
        results_display.insert(tk.END, "No results found. Try different keywords.")

    # Make the ScrolledText read-only again
    results_display.config(state=tk.DISABLED)

# --- UI elements ---

# 1. Title label
logo_label = tk.Label(root, text="IR Search Engine", font=("Arial", 28, "bold"), bg='#202124', fg='#8ab4f8')
logo_label.pack(pady=(30, 20))

# 2. Container for the search bar
search_frame = tk.Frame(root, bg='#202124')
search_frame.pack(pady=10)

# 3. Search entry bar
search_entry = tk.Entry(
    search_frame,
    font=("Arial", 14),
    width=40,
    bd=1,
    relief="solid",
    bg='#303134',
    fg='#e8eaed',
    insertbackground='#e8eaed'
)
search_entry.pack(ipady=7)
search_entry.focus_set()

# 4. Search button
search_button = tk.Button(
    root,
    text="Google Search",
    font=("Arial", 11),
    bg='#3c4043',
    fg='#e8eaed',
    command=on_search_click,
    relief="groove",
    activebackground='#5f6368',
    activeforeground='#e8eaed'
)
search_button.pack(pady=20)

# 5. Results area (ScrolledText)
results_display = scrolledtext.ScrolledText(
    root,
    width=70,
    height=18,
    font=("Arial", 11),
    bd=0,
    bg='#202124',
    fg='#d4d4d4',
    insertbackground='#d4d4d4'
)
results_display.pack(padx=20, pady=10)
results_display.config(state=tk.DISABLED)

# Simple color styling for readability
results_display.tag_config('score', foreground='#fbbc05', font=("Arial", 11, "bold"))
results_display.tag_config('fname', foreground='#4fc1ff', font=("Arial", 11))
results_display.tag_config('content', foreground='#d4d4d4', font=("Arial", 11, "italic"))
results_display.tag_config('highlight', background='#fbbc05', foreground='#202124')

# Run the app
root.mainloop()


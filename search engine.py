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
# TF-IDF & Inverted Index Engine (Backend)
# =========================================================

documents = []
file_names = []

if not os.path.isdir(FOLDER_NAME):
    raise FileNotFoundError(
        f"Missing '{FOLDER_NAME}' folder. Please create it and add .txt files."
    )

for filename in os.listdir(FOLDER_NAME): #to make this path collection/book.txt
    if filename.endswith(".txt"):
        file_path = os.path.join(FOLDER_NAME, filename)
        with open(file_path, "r", encoding="utf-8") as file: # with -> close file after usage , "r" -> read mode , encoding for arabic words
            documents.append(file.read())
            file_names.append(filename)


def basic_stemmer(word):
    if len(word) <= 3:
        return word

    def strip_double_consonant(stem):
        if len(stem) >= 2:
            last = stem[-1] #take last character
            if last == stem[-2] and last not in "aeiou":# if aeiou dont remove ex agreeing -> agree
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
    text = re.sub(r'[^\w\s]', '', text.lower()) # remove anything like ! ? - . except letters, and covert to lowercase
    words = text.split() #splite to words - tokenization
    filtered = [w for w in words if w not in STOP_WORDS]
    return [basic_stemmer(w) for w in filtered]


cleaned_docs = [clean_text(doc) for doc in documents] # pass every doc at documents[] to clean_text(doc)

#cleaned_docs = [ as example
#     ["cat", "run"],
#     ["cat", "play"]
# ]

def compute_tf(doc): #(علاقة الكلمة بالملف).
    tf = {} # dictionary => (key: word , value: tf)
    for word in doc:
        tf[word] = tf.get(word, 0) + 1 #if this word not fount at dictionary , make count = 0 then add 1
    for word in tf:
        count = tf[word]
        tf[word] = 1 + math.log10(count) if count > 0 else 0 #calc 1+log(count of word)
    return tf


def compute_idf(docs):# بيقيس الكلمة نادرة ولا منتشرة في كل الملفات (علاقة الكلمة بالمشروع كله).
    idf = {}
    total_docs = len(docs)
    all_words = set([word for doc in docs for word in doc]) #have unique words in files -> set , enter docs to words
    for word in all_words:
        containing_docs = sum([1 for doc in docs if word in doc]) # to each word if word in docs write 1 then sum ones
        if total_docs > 0 and containing_docs > 0:
            idf[word] = math.log10(total_docs / float(containing_docs))
        else:
            idf[word] = 0
    return idf


# Compute IDF once at startup
idf_scores = compute_idf(cleaned_docs) #dictionary

# ---------------------------------------------------------
#Building the Inverted Index & Document Magnitudes
# ---------------------------------------------------------

#{ word: { doc_index: tfidf_value } }
inverted_index = {}
doc_magnitudes = [0.0] * len(documents) #list that filled by zeros علي قد عدد الملفات اللي عندنا

for doc_idx, doc in enumerate(cleaned_docs):
    tf = compute_tf(doc) #calc term frequency for each doc in cleaned docs - tf is dictionary => (key: word , value: tf calculated)
    sum_squares = 0.0

    for word, tf_val in tf.items():
        tfidf_val = tf_val * idf_scores.get(word, 0)#default zero if not found
        if tfidf_val > 0:

            if word not in inverted_index:
                inverted_index[word] = {}

            inverted_index[word][doc_idx] = tfidf_val #index construction
            sum_squares += tfidf_val ** 2 #length normalization -1

    doc_magnitudes[doc_idx] = math.sqrt(sum_squares) #length normalization -2

#inverted_index = {
#     "cat":  { 0: 0.45, 1: 0.21 },
#     "run":  { 0: 1.15 },
#     "play": { 1: 0.85 }
# }

# ---------------------------------------------------------
# Main search function using Inverted Index & Cosine Similarity
# ---------------------------------------------------------
def search_engine(query):
    cleaned_query = clean_text(query) #list of words
    if not cleaned_query:
        return []

    # 1. Compute Query Vector and Magnitude (||Q||)
    query_tf = compute_tf(cleaned_query)
    query_vector = {}
    query_sum_squares = 0.0
    for word, tf_val in query_tf.items():
        tfidf_val = tf_val * idf_scores.get(word, 0)
        query_vector[word] = tfidf_val
        query_sum_squares += tfidf_val ** 2

    query_magnitude = math.sqrt(query_sum_squares)
    if query_magnitude == 0:
        return []

    # 2. Accumulate Dot Products ONLY for documents that contain query terms
    # هنا تظهر قوة الـ Inverted Index، لا نمر على كل الملفات بل نأخذ المستهدفة فقط
    dot_products = {}  # { doc_index: accumulated_dot_product }

    for word, q_tfidf in query_vector.items():
        if word in inverted_index:
            # نمر فقط على الملفات التي تحتوي على هذه الكلمة
            for doc_idx, d_tfidf in inverted_index[word].items():
                dot_products[doc_idx] = dot_products.get(doc_idx, 0.0) + (q_tfidf * d_tfidf) #if search more than 1 word

    # 3. Calculate Cosine Similarity for the candidate documents
    scores = []
    for doc_idx, dot_product in dot_products.items():
        magnitude_product = query_magnitude * doc_magnitudes[doc_idx]

        if magnitude_product > 0:
            cosine_score = dot_product / magnitude_product
        else:
            cosine_score = 0.0

        if cosine_score > 0:
            scores.append((cosine_score, file_names[doc_idx], documents[doc_idx]))# pass cosine score - original file name and its content

    # Sort by highest cosine similarity score
    scores.sort(reverse=True, key=lambda x: x[0]) #sort according score
    return scores


# =========================================================
# GUI
# =========================================================
root = tk.Tk()
root.title("Search Engine")
root.geometry("600x550")
root.configure(bg='#1B1F1B')


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


def on_search_click():
    query = search_entry.get()

    results_display.config(state=tk.NORMAL)
    results_display.delete('1.0', tk.END)

    if not query.strip():
        messagebox.showwarning("Warning", "Please enter a search query!")
        results_display.config(state=tk.DISABLED)
        return

    start_time = time.perf_counter()
    results = search_engine(query)
    elapsed = time.perf_counter() - start_time

    if results:
        header = f"Found {len(results)} relevant document(s) in {elapsed:.6f} seconds\n"
        header += "=" * 30 + "\n\n"
        results_display.insert(tk.END, header)
        query_terms = clean_text(query)
        original_terms = re.sub(r'[^\w\s]', '', query.lower()).split()
        for rank, (score, fname, doc) in enumerate(results, start=1):
            snippet = make_snippet(doc, query_terms)
            results_display.insert(tk.END, f"{rank}. [Cosine Score: {score:.4f}]\n", 'score')
            results_display.insert(tk.END, f"File: {fname}\n", 'fname')
            snippet_start = results_display.index(tk.INSERT)
            results_display.insert(tk.END, f"Snippet: \"{snippet}\"\n\n", 'content')
            snippet_end = results_display.index(tk.INSERT)
            highlight_terms(results_display, original_terms, snippet_start, snippet_end)
    else:
        results_display.insert(tk.END, "No results found. Try different keywords.")

    results_display.config(state=tk.DISABLED)


# --- UI elements ---
logo_label = tk.Label(root, text="Search Engine", font=("Arial", 28, "bold"), bg='#202124', fg='#8ab4f8')
logo_label.pack(pady=(30, 20))

search_frame = tk.Frame(root, bg='#202124')
search_frame.pack(pady=10)

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
search_entry.bind('<Return>', lambda event: on_search_click())

search_button = tk.Button(
    root,
    text="Search",
    font=("Arial", 11),
    bg='#3c4043',
    fg='#e8eaed',
    command=on_search_click,
    relief="groove",
    activebackground='#5f6368',
    activeforeground='#e8eaed'
)
search_button.pack(pady=20)

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

results_display.tag_config('score', foreground='#fbbc05', font=("Arial", 11, "bold"))
results_display.tag_config('fname', foreground='#4fc1ff', font=("Arial", 11))
results_display.tag_config('content', foreground='#d4d4d4', font=("Arial", 11, "italic"))
results_display.tag_config('highlight', background='#fbbc05', foreground='#202124')

root.mainloop()
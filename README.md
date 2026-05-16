# TF-IDF Python Search Engine 🔍

## Project Overview
This project is a custom-built, lightweight Information Retrieval (IR) system from scratch. It implements a search engine using the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to rank a collection of text documents based on user queries. It features a modern Dark Mode Graphical User Interface (GUI) built with Tkinter.

## Core Features ✨
1. **Custom TF-IDF Algorithm:** Built entirely from scratch without external ML libraries. Implements Log Normalization for Term Frequency to prevent bias toward longer documents.
2. **Text Preprocessing Engine:**
   - **Punctuation & Case Folding:** Cleans text by converting to lowercase and removing special characters.
   - **Stop-words Removal:** Custom filter to exclude non-informative words (e.g., 'the', 'is', 'in').
   - **Advanced Custom Stemming:** Strips common suffixes ('ing', 'ed', 's') and correctly handles double consonant endings (e.g., converting 'programming' directly to 'program').
3. **Google-Style GUI:** A user-friendly dark-themed interface that highlights search terms within the document snippets and displays exact query execution time.

## Mathematical Model 🧮
The engine uses the following models for ranking:
* **Log-Normalized TF:** $$TF = 1 + \log_{10}(\text{count})$$ (if count > 0, else 0)
* **IDF:** $$IDF = \log_{10} \left( \frac{N}{df} \right)$$
* **Score:** $$TF\text{-}IDF = TF \times IDF$$

## How to Run 🚀
1. Ensure Python 3.x is installed.
2. Place your `.txt` files inside a folder named `collection` in the same directory as the script.
3. Run the script: `python main.py`
4. Enter your query in the search bar and hit "Google Search".
import re
import os
import sys
import pickle
import shutil
import sqlite3
import datetime
import numpy as np
import networkx as nx
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.converter import PDFPageAggregator
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

os.chdir(os.path.dirname(sys.argv[0]))
if not os.path.exists("C:/nltk_data"):
    # Use shutil.copytree() to copy the entire folder and its contents
    shutil.copytree("./nltk_data", "C:/nltk_data")


def is_valid_date(date_str):
    date_formats = [
        '%Y-%m-%d',
        '%b %d, %Y',
        '%b %d, %Y %I:%M:%S %p',
        '%b %d, %Y %I:%M:%S %p %Z',
        '%b %d, %Y %I:%M:%S %p %Z',
        '%d %b %Y',
        '%d %b %Y %I:%M:%S %p',
        '%d %b %Y %I:%M:%S %p %Z',
        '%d %b %Y %I:%M:%S %p %Z',
    ]

    for date_format in date_formats:
        try:
            # Attempt to parse the date string with the current format
            datetime.datetime.strptime(date_str, date_format)
            return True
        except ValueError:
            continue

    # If none of the formats match, return False
    return False


def extract_text_with_coordinates(pdf_path):
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    extracted_text = []

    with open(pdf_path, 'rb') as pdf_file:
        for idx, page in enumerate(PDFPage.get_pages(pdf_file)):
            interpreter.process_page(page)
            layout = device.get_result()

            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox):
                    text = lt_obj.get_text()
                    x0, y0, x1, y1 = lt_obj.bbox
                    extracted_text.append((idx, text, (x0, y0, x1, y1)))

    return extracted_text


def split_fn(sentence, delimiters):
    # Create a regex pattern that matches any of the delimiters
    regex_pattern = "|".join(map(re.escape, delimiters))

    # Split the sentence into words and keep the delimiters as separate elements
    tokens = re.split(f'({regex_pattern})', sentence)

    # Remove empty strings from the list
    tokens = [token for token in tokens if token]

    return tokens


def remove_escape_sequences(text):
    pattern = r"\\[a-z]"
    text = re.sub(pattern, " ", text)
    return text


def get_last_inserted_rowid():
    try:
        conn = sqlite3.connect(r"Document_finder_db2.db")
        c = conn.cursor()
        c.execute('''SELECT MAX(rowid) FROM document_info''')
        tup = c.fetchone()
        conn.close()
        return tup[0]
    except Exception:
        print('Cannot access the database right now')


def cleaning_for_summarization(text):
    pattern = r"((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)"
    text = re.sub(pattern, " ", text)

    sentences = sent_tokenize(text)
    #     for j in range(len(sentences)):
    #         sentences[j] = re.sub("[^a-zA-Z]"," ",sentences[j])

    clean_sentences = sentences

    for j in range(len(clean_sentences)):
        clean_sentences[j] = word_tokenize(clean_sentences[j])

    return clean_sentences


def get_summary(text, word_embeddings):
    tokenized_sent = cleaning_for_summarization(text)
    sentence_vectors = []
    for i in tokenized_sent:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i]) / (len(i) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    # similarity matrix
    sim_mat = np.zeros([len(tokenized_sent), len(tokenized_sent)])

    for i in range(len(tokenized_sent)):
        for j in range(len(tokenized_sent)):
            if i != j:
                sim_mat[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(tokenized_sent)), reverse=True)
    summarize_text = []
    if len(ranked_sentence) == 1:
        summarize_text.append(" ".join(ranked_sentence[0][1]))
    elif len(ranked_sentence) == 0:
        summarize_text = []
    else:
        for i in range(4):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

    if len(".. ".join(summarize_text)) > 1400:
        summary = summarize_text[0]
    else:
        summary = ".. ".join(summarize_text)

    return summary.replace(' , ', ', ').replace(' .', '.').replace(' ( ', '(').replace(' )', ')')


def get_data_point(tokens, data_point, result):
    token = data_point + "\n"
    if token in tokens:
        idx = tokens.index(token)
        if data_point == "Description":
            word_embeddings = pickle.load(open(r"word_embeddings.json", "rb"))
            summary = get_summary(tokens[idx + 1].replace("\n", " ").strip(), word_embeddings)
            result.append((data_point, summary))
        else:
            result.append((data_point, tokens[idx + 1].replace("\n", " ").strip()))
    return result


def extract_information(pdf_path):
    extracted_text = extract_text_with_coordinates(pdf_path)
    extracted_text.sort(key=lambda item: item[0] * 100000 + (10000 - item[2][3]))

    txt = ""

    prev_page_idx = -1
    title = ""
    for page_idx, text, coordinates in extracted_text:
        if page_idx == 1 and title == "":
            title = text.replace('\n', '').strip()
        # remove date
        if prev_page_idx != page_idx or is_valid_date(text.strip()) or text.strip().isdigit():
            prev_page_idx = page_idx
            continue
        # print(f"Text: {text.strip()}")
        txt += text.strip()
        txt += '\n'
        # print(f"Coordinates: x0={coordinates[0]}, y0={coordinates[1]}, x1={coordinates[2]}, y1={coordinates[3]}\n")
        prev_page_idx = page_idx

    delimiters = [
        "Opportunity Details\n",
        "Notice ID\n",
        "Notice Status\n",
        "Related Notice\n",
        "Department/Ind. Agency\n",
        "Active/Inactive\n",
        "Sub-Tier\n",
        "Office\n"
        "Looking for contract opportunity help?\n"
        "General Information\n",
        "Contract Opportunity Type\n",
        "Date Offers Due\n",
        "Inactive Date\n",
        "Allow Vendors to Add/remove from Interested Vendors \nList\n",
        "Allow Vendors to Add/remove from Interested Vendors List\n",
        "Updated Published Date\n",
        "Inactive Policy\n",
        "Initiative\n",
        "Allow Vendors to View Interested Vendors List\n",
        "Classification\n",
        "Original Set Aside\n",
        "Place of Performance\n",
        "Product Service Code\n",
        "NAICS Code(s)\n",
        "NAICS Code\n",
        "NAICS Definition\n",
        "Description\n",
        "Attachment/Links\n",
        "Attachments\n",
        "Links\n",
        "Contact Information\n",
        "Primary Point of Contact\n",
        "Secondary Point of Contact\n",
        "Secondary Point of Contact\n",
        "Third Point of Contact\n",
        "Fourth Point of Contact\n",
        "History\n"
    ]

    data_points = [
        "Notice ID",
        "Department/Ind. Agency",
        "Contract Opportunity Type",
        "Date Offers Due",
        "Updated Published Date",
        "Original Set Aside",
        "Place of Performance",
        "NAICS Code",
        "NAICS Definition",
        "Description",
        "Primary Point of Contact",
        "Secondary Point of Contact",
    ]
    txt = remove_escape_sequences(txt)

    tokens = split_fn(txt, delimiters)
    result = [("Title", title)]
    for data_point in data_points:
        result = get_data_point(tokens, data_point, result)

    filename, _, _ = pdf_path.rpartition('.')
    with open(filename + ".txt", "w+", encoding="utf-8") as file:
        for data_point, content in result:
            file.write(data_point + " : " + content + "\n")
            print(data_point, " : ", content)


if __name__ == "__main__":
    # Replace with the path to your PDF file
    extract_information(sys.argv[1])

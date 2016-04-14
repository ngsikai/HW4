from __future__ import division
import sys
import getopt
import os
import string
import math
import pickle
import operator
import xml.etree.ElementTree as ET
from operator import itemgetter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords


bits_per_posting = 22
relevant_docs_count = 10
alpha = 1.0
beta = 1.0
gamma = -3.0
stemmer = PorterStemmer()
stopwords = map(lambda word: stemmer.stem(word).lower(), stopwords.words('english'))


'''
Retrieves Title and Desc of the query file and
puts it through preliminary scoring to select top relevant_docs_count documents.

Then, expand the initial query using the selected documents using the Rocchio Algorithm and
puts it through a final round of scoring.
'''
def search_index(dictionary_file, postings_file, queries_file, output_file):
    main_dict = pickle.load(open(dictionary_file, 'rb'))
    postings_lists = open(postings_file, 'r')
    search_results = open(output_file, 'w')

    # Title and Desc dict contains term -> tf mapping
    query_title_dict = {}
    query_desc_dict = {}
    query_dict = {"TITLE": query_title_dict, "DESCRIPTION": query_desc_dict}

    # Takes in query file, parses it with xml etree library and gets its root
    tree = ET.parse(queries_file)
    root = tree.getroot()

    for child in root:
        name = child.tag.upper()
        if name == "TITLE" or name == "DESCRIPTION":
            query_dict[name] = get_query_dict(child.text)

    # Score with initial query
    scores_dict = get_scores_dict(main_dict, postings_lists, query_dict)
    initial_scores = sort_scores(scores_dict)
    relevant_docs = map(lambda x: x[0], initial_scores[:relevant_docs_count])

    # Query Expansion with Rocchio Algorithm
    main_doc_dict = main_dict["MAIN_DOC_DICT"]
    query_dict = expand_query(query_dict, main_doc_dict, relevant_docs, postings_lists)

    # Score with expanded query
    expanded_scores_dict = get_scores_dict(main_dict, postings_lists, query_dict)
    final_scores = sort_scores(expanded_scores_dict)

    # Write to search_results
    search_results.write(stringify(final_scores))

    postings_lists.close()
    search_results.close()


'''
Takes in query and outputs dictionary with term -> tf mapping
'''
def get_query_dict(text):
    if type(text) is unicode:
        text = text.encode('ascii', errors='ignore')
    query_dict = {}
    for word in filter(lambda word: is_valid_token(word), word_tokenize(text)):
        token = stemmer.stem(word).lower()
        if token in query_dict:
            query_dict[token] += 1
        else:
            query_dict[token] = 1
    return query_dict


'''
Returns dictionary of doc_id -> score mapping.
Documents are scored accordingly if their title or abstract contains any of the query terms.

There are four mappings possible:
term in both query title and document title => 2 x 2 (highest weightage)
term in query title and document abstract => 2 x 1 
term in query description and document title => 1 x 2
term in both query description and document abstract => 1 x 1 (lowest weightage)
'''
def get_scores_dict(main_dict, postings_lists, query_dict):
    scores_dict = {}
    query_norm_factor = 0
    wt = 0

    for dict_name, dictionary in query_dict.iteritems():
        if dict_name == "TITLE":
            wt = 2
        elif dict_name == "DESCRIPTION":
            wt = 1

        for term, tf in dictionary.iteritems():
            query_norm_factor += update_scores_dict(postings_lists, scores_dict, term, tf, main_dict["TITLE"], wt*2)
            query_norm_factor += update_scores_dict(postings_lists, scores_dict, term, tf, main_dict["ABSTRACT"], wt)
    
    query_norm_factor = math.pow(query_norm_factor, 0.5)

    doc_norm_factors = main_dict["DOCUMENT_NORM_FACTORS"]
    return normalise(scores_dict, query_norm_factor, doc_norm_factors)


'''
Computes document scores for a given query_term using lnc.ltc and updates it in the given scores_dict
Returns weight^2 of the given query_term for normalization
'''
def update_scores_dict(postings_lists, scores_dict, query_term, query_tf, dictionary, wt):
    if query_term in dictionary and query_tf >= 0:
        # Calculates query_wt
        query_tf_wt = 1 + math.log10(query_tf)
        idf = dictionary[query_term][1]
        query_wt = query_tf_wt * idf

        # Retrieves postings list of query_term
        df = dictionary[query_term][0]
        ptr = dictionary[query_term][2]
        postings_list = get_postings_list(df, ptr, postings_lists)

        for posting in postings_list:
            doc_name = posting[0]
            doc_tf = posting[1]
            doc_wt = 1 + math.log10(doc_tf)
            score = query_wt * doc_wt
            if doc_name in scores_dict:
                scores_dict[doc_name] += score * wt
            else:
                scores_dict[doc_name] = score * wt

        return query_wt ** 2
    else:
        return 0


'''
Expands query using Rocchio Algorithm with the given set of relevant documents
'''
def expand_query(query_dict, main_doc_dict, relevant_docs, postings_lists):
    all_docs = main_doc_dict["TITLE"].keys()
    irrelevant_docs = get_irrelevant_docs(all_docs, relevant_docs)
    relevant_vector = get_vector(relevant_docs, main_doc_dict, postings_lists)
    irrelevant_vector = get_vector(irrelevant_docs, main_doc_dict, postings_lists)

    query_dict = multiply_vector_dict(query_dict, alpha)
    relevant_vector = multiply_vector_dict(relevant_vector, beta)
    irrelevant_vector = multiply_vector_dict(irrelevant_vector, gamma)

    return add_three_vectors(query_dict, relevant_vector, irrelevant_vector)


def multiply_vector_dict(vector_dict, multiple):
    for dictionary in vector_dict.itervalues():
        for term, tf in dictionary.iteritems():
            dictionary[term] = tf * multiple
    return vector_dict


'''
Given a set of docs, returns a dictionary of term -> tf mapping for the entire set of docs
This dictionary is a vector representation of the set of docs
'''
def get_vector(docs, main_doc_dict, postings_lists):
    title_vector_dict = {}
    abstract_vector_dict = {}
    vector_dict = {"TITLE": title_vector_dict, "ABSTRACT": abstract_vector_dict}
    
    for doc in docs:
        for dict_name, dictionary in main_doc_dict.iteritems():
            if doc in dictionary:
                ptr = dictionary[doc][0]
                offset = dictionary[doc][1]
                doc_dict = eval(get_doc_dict(ptr, offset, postings_lists))
                vector_dict[dict_name] = add_vector_dict(vector_dict[dict_name], doc_dict)

    N = len(docs)
    return multiply_vector_dict(vector_dict, 1/N)


def add_three_vectors(query_dict, relevant_vector, irrelevant_vector):
    query_dict["TITLE"] = add_vector_dict(query_dict["TITLE"], relevant_vector["TITLE"])
    query_dict["TITLE"] = add_vector_dict(query_dict["TITLE"], irrelevant_vector["TITLE"])

    query_dict["DESCRIPTION"] = add_vector_dict(query_dict["DESCRIPTION"], relevant_vector["ABSTRACT"])
    query_dict["DESCRIPTION"] = add_vector_dict(query_dict["DESCRIPTION"], irrelevant_vector["ABSTRACT"])
    return query_dict


def add_vector_dict(vector_dict, doc_dict):
    for term, tf in doc_dict.iteritems():
        if term in vector_dict:
            vector_dict[term] = vector_dict[term] + tf
        else:
            vector_dict[term] = tf
    return vector_dict


def get_irrelevant_docs(all_docs, relevant_docs):
    return list(set(all_docs) - set(relevant_docs))


def sort_scores(scores_dict):
    tuples_list = scores_dict.items()
    tuples_list.sort(key=operator.itemgetter(0))
    tuples_list.sort(key=operator.itemgetter(1), reverse=True)
    return tuples_list


def normalise(scores_dict, query_norm_factor, doc_norm_factors):
    for doc_id, score in scores_dict.iteritems():
        norm_factor = doc_norm_factors[doc_id] * query_norm_factor
        scores_dict[doc_id] = score / norm_factor
    return scores_dict


def get_postings_list(df, term_pointer, postings_lists):
    postings_lists.seek(term_pointer)
    results_list = postings_lists.read(df * bits_per_posting).strip().split()
    tuples_list = []
    for i in range(0, len(results_list), 2):
        doc_name = results_list[i].strip()
        tf = int(results_list[i + 1].strip())
        tuples_list.append((doc_name, tf))
    return tuples_list


def get_doc_dict(ptr, offset, postings_lists):
    postings_lists.seek(ptr)
    return postings_lists.read(offset)


def is_valid_token(word):
    if word in stopwords:
        return False
    elif word in string.digits:
        return False
    elif word in string.punctuation:
        return False
    else:
        for char in list(word):
            if char in string.digits:
                return False
            elif char in string.punctuation:
                return False
    return True


def stringify(list):
    ans = ""
    for element in list:
        ans += str(element[0]) + " "
    return ans.strip()


def usage():
    print "usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"

input_file_d = input_file_p = input_file_q = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-d':
        input_file_d = a
    elif o == '-p':
        input_file_p = a
    elif o == '-q':
        input_file_q = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_d is None or input_file_p is None or input_file_q is None or output_file is None:
    usage()
    sys.exit(2)


search_index(input_file_d, input_file_p, input_file_q, output_file)

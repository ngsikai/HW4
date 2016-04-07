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


bits_per_posting = 22
stemmer = PorterStemmer()


def search_index(dictionary_file, postings_file, queries_file, output_file):
    dictionary = pickle.load(open(dictionary_file, 'rb'))
    postings_lists = open(postings_file, 'r')
    search_results = open(output_file, 'w')

    query_title_dict = {}
    query_desc_dict = {}
    query_dict = {"TITLE": query_title_dict, "DESCRIPTION": query_desc_dict}

    tree = ET.parse(queries_file)
    root = tree.getroot()
    for child in root:
        name = child.tag.upper()
        if name == "TITLE" or name == "DESCRIPTION":
            query_dict[name] = get_query_dict(child.text)

    scores_dict = get_scores_dict(dictionary, postings_lists, query_dict)
    final_scores = sort_scores(scores_dict)
    search_results.write(stringify(final_scores))

    postings_lists.close()
    search_results.close()


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


def get_scores_dict(main_dict, postings_lists, query_dict):
    scores_dict = {}
    query_norm_factor = 0
    wt = 0

    for dict_name, dictionary in query_dict.iteritems():
        if dict_name == "TITLE":
            wt = 2
        elif dict_name == "DESCRIPTION":
            wt = 1

        for term, term_freq in dictionary.iteritems():
            query_norm_factor += update_scores_dict(postings_lists, scores_dict, term, term_freq, main_dict["TITLE"], wt*2)
            query_norm_factor += update_scores_dict(postings_lists, scores_dict, term, term_freq, main_dict["ABSTRACT"], wt)
    query_norm_factor = math.pow(query_norm_factor, 0.5)

    doc_norm_factors = main_dict["DOCUMENT_NORM_FACTORS"]
    return normalise(scores_dict, query_norm_factor, doc_norm_factors)


def update_scores_dict(postings_lists, scores_dict, query_term, query_tf, dictionary, wt):
    if query_term in dictionary:
        query_tf_wt = 1 + math.log10(query_tf)
        idf = dictionary[query_term][1]
        query_wt = query_tf_wt * idf

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

        return query_tf_wt ** 2
    else:
        return 0


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


def is_valid_token(word):
    if word in string.digits:
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

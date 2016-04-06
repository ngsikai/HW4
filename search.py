from __future__ import division
import sys
import getopt
import os
import math
import pickle
import operator
from operator import itemgetter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *

K = 10
bits_per_posting = 30


# For each line in query file, performs query and outputs result
def search_index(dictionary_file, postings_file, queries_file, output_file):
    dictionary = pickle.load(open(dictionary_file, 'rb'))
    postings_lists = open(postings_file, 'r')
    queries_list = open(queries_file, 'r').read().split("\n")
    search_results = open(output_file, 'w')

    for index, query in enumerate(queries_list):
        # in case blank line is caught as a query, write an empty line
        if query == "":
            search_results.write("\n")
        else:
            scores_dict = get_scores_dict(query, dictionary, postings_lists)
            top_results = get_top_results(scores_dict)
            search_results.write(stringify(top_results))
            if index != len(queries_list) - 1:
                search_results.write("\n")

    postings_lists.close()
    search_results.close()


def get_top_results(scores_dict):
    tuples_list = [(doc_id, score) for doc_id, score in scores_dict.items()]
    tuples_list.sort(key = operator.itemgetter(0))
    tuples_list.sort(key = operator.itemgetter(1), reverse = True)
    return tuples_list[:K]

def get_scores_dict(query_str, dictionary, postings_lists):
    doc_count = dictionary["DOCUMENT_COUNT"]
    doc_norm_factors = dictionary["DOCUMENT_NORM_FACTORS"]
    scores_dict = {}
    # query_norm_factor is used to compute final normalising values
    query_norm_factor = 0
    query_dict = process_query(query_str)
    for term in query_dict:
        if term in dictionary:
            # query computations
            df = dictionary[term][0]
            idf = math.log10(doc_count / df)
            query_tf = query_dict[term]
            query_tf_wt = 1 + math.log10(query_tf)
            query_wt = idf * query_tf_wt
            query_norm_factor += math.pow(query_wt, 2)
            # retrieving postings list for term
            term_pointer = dictionary[term][1]
            postings_list = get_postings_list(df, term_pointer, postings_lists)
            # document computations
            for posting in postings_list:
                doc_id = posting[0]
                doc_tf = posting[1]
                doc_wt = 1 + math.log10(doc_tf)
                score = query_wt * doc_wt
                if doc_id in scores_dict:
                    scores_dict[doc_id] += score
                else:
                    scores_dict[doc_id] = score
    query_norm_factor = math.pow(query_norm_factor, 0.5)
    return normalise(scores_dict, query_norm_factor, doc_norm_factors)


# Normalises scores dictionary
def normalise(scores_dict, query_norm_factor, doc_norm_factors):
    for doc_id, score in scores_dict.iteritems():
        norm_factor = doc_norm_factors[str(doc_id)] * query_norm_factor
        scores_dict[doc_id] = score / norm_factor
    return scores_dict


# Converts query string into dictionary form
def process_query(query_str):
    stemmer = PorterStemmer()
    query_list = query_str.split(" ")
    query_dict = {}
    for query in query_list:
        query = stemmer.stem(query).lower()
        if query in query_dict:
            query_dict[query] += 1
        else:
            query_dict[query] = 1
    return query_dict


# Seeks, loads and returns a postings list
def get_postings_list(df, term_pointer, postings_lists):
    postings_lists.seek(term_pointer)
    results_list = postings_lists.read(df * bits_per_posting - 1).strip().split()
    results_list = map((lambda x: int(x, 2)), results_list)
    tuples_list = []
    for i in range(0, len(results_list), 2):
        tuples_list.append((results_list[i], results_list[i + 1]))
    return tuples_list


# Converts list of tuples to string for writing to output file
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

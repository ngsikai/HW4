import sys
import getopt
import os
import string
import math
import pickle
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *


stemmer = PorterStemmer()


def write_dictionary(dictionary, dictionary_file):
    pickle.dump(dictionary, dictionary_file)


def write_postings(main_dict, postings_lists, postings_file):
    for dictionary in main_dict.itervalues(): # dictionary is either title_dict or abstract_dict
        for term, lst in dictionary.iteritems():
            file_pointer = postings_file.tell()
            df = lst[0]
            idf = lst[1]
            term_pointer = lst[2]
            postings_list = postings_lists[term_pointer]
            for posting in postings_list:
                postings_file.write(convert_doc_name(posting[0]) + " ")
                postings_file.write(convert_to_bits(posting[1]) + " ")
            dictionary[term] = [df, idf, file_pointer]
    return main_dict


def convert_doc_name(doc_name):
    return doc_name.ljust(15)


def convert_to_bits(num):
    return '{0:014b}'.format(int(num))


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


def build_index(input_doc_path, output_file_d, output_file_p):
    dictionary_file = open(output_file_d, 'wb')
    postings_file = open(output_file_p, 'w')

    title_dict = {}
    abstract_dict = {}
    main_dict = {"TITLE": title_dict, "ABSTRACT": abstract_dict}

    doc_norm_factors = {}

    postings_lists = {}

    doc_names = get_doc_names(input_doc_path)
    for doc_name in doc_names:
        path = input_doc_path + '/' + doc_name
        tree = ET.parse(path)
        root = tree.getroot()
        doc_norm_factor = 0
        for child in root:
            name = child.attrib['name'].upper()
            if name == "TITLE" or name == "ABSTRACT":
                text = child.text
                if type(child.text) is unicode:
                    text = child.text.encode('ascii', errors = 'ignore')

                doc_dict = get_doc_dict(text)
                for term, term_freq in doc_dict.iteritems():
                    doc_wt = 1 + math.log10(term_freq)
                    doc_norm_factor += math.pow(doc_wt, 2)
                    dictionary = main_dict[name] # dictionary is either title_dict or abstract_dict
                    if term not in dictionary:
                        term_pointer = len(postings_lists)
                        dictionary[term] = [1, 0, term_pointer]
                        postings_lists[term_pointer] = [(doc_name, term_freq)]
                    else:
                        df = dictionary[term][0]
                        term_pointer = dictionary[term][2]
                        dictionary[term] = [df + 1, 0, term_pointer]
                        postings_lists[term_pointer].append((doc_name, term_freq))

        doc_norm_factors[doc_name] = math.pow(doc_norm_factor, 0.5)

    main_dict = compute_idf(main_dict, len(doc_names))    
    main_dict = write_postings(main_dict, postings_lists, postings_file)
    main_dict["DOCUMENT_NORM_FACTORS"] = doc_norm_factors

    print doc_norm_factors

    write_dictionary(main_dict, dictionary_file)
    
    dictionary_file.close()
    postings_file.close()


def get_doc_names(path):
    return os.listdir(path)


def get_doc_dict(text):
    doc_dict = {}
    for word in filter(lambda x: is_valid_token(x), word_tokenize(text)):
        token = stemmer.stem(word).lower()
        if token in doc_dict:
            doc_dict[token] += 1
        else:
            doc_dict[token] = 1
    return doc_dict


def compute_idf(main_dict, N):
    N = float(N)
    for dictionary in main_dict.itervalues():
        for term, lst in dictionary.iteritems():
            df = lst[0]
            idf = math.log10(N / df)
            term_ptr = lst[2]
            dictionary[term] = [df, idf, term_ptr]
    return main_dict


def usage():
    print "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file"

input_doc_path = output_file_d = output_file_p = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        input_doc_path = a
    elif o == '-d':
        output_file_d = a
    elif o == '-p':
        output_file_p = a
    else:
        assert False, "unhandled option"
if input_doc_path is None or output_file_d is None or output_file_p is None:
    usage()
    sys.exit(2)


build_index(input_doc_path, output_file_d, output_file_p)

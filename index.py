import sys
import getopt
import os
import math
import string
import pickle
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *


# Takes in a path string and outputs a list of file names in the directory
def get_doc_names(path):
    doc_names = os.listdir(path)
    return doc_names


def write_dictionary(dictionary, dictionary_file):
    pickle.dump(dictionary, dictionary_file)


def write_postings(main_dict, postings_lists, postings_file):
    for dictionary in main_dict.itervalues():
        for term, values in dictionary.iteritems():
            file_pointer = postings_file.tell()
            df = values[0]
            idf = values[1]
            term_pointer = values[2]
            postings_list = postings_lists[term_pointer]
            for posting in postings_list:
                postings_file.write(convert_filename(posting[0]) + " ")
                postings_file.write(convert_to_bytes(posting[1]) + " ")
            dictionary[term] = (df, idf, file_pointer)
    return main_dict


def convert_filename(filename):
    return filename.ljust(15)


def convert_to_bytes(num):
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

    postings_lists = {}
    doc_norm_factors = {}

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
                    if term not in main_dict[name]:
                        term_pointer = len(postings_lists)
                        main_dict[name][term] = (1, 0, term_pointer)
                        postings_lists[term_pointer] = [(doc_name, term_freq)]
                    else:
                        df = main_dict[name][term][0]
                        term_pointer = main_dict[name][term][2]
                        main_dict[name][term] = (df + 1, 0, term_pointer)
                        postings_lists[term_pointer].append((doc_name, term_freq))

        doc_norm_factors[doc_name] = math.pow(doc_norm_factor, 0.5)
    main_dict = compute_idf(main_dict, len(doc_names))
    main_dict = write_postings(main_dict, postings_lists, postings_file)
    main_dict["DOCUMENT_NORM_FACTORS"] = doc_norm_factors
    write_dictionary(main_dict, dictionary_file)
    dictionary_file.close()
    postings_file.close()


def get_doc_dict(text):
    stemmer = PorterStemmer()

    doc_dict = {}
    for word in filter(lambda x: is_valid_token(x), word_tokenize(text)):
        token = stemmer.stem(word).lower()
        # updating of term frequency
        if token in doc_dict:
            doc_dict[token] += 1
        else:
            doc_dict[token] = 1
    return doc_dict


def compute_idf(main_dict, N):
    for dictionary in main_dict.itervalues():
        for term, value in dictionary.iteritems():
            df = value[0]
            idf = math.log10(N // df)
            term_ptr = value[2]
            dictionary[term] = (df, idf, term_ptr)
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

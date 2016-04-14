import sys
import getopt
import os
import string
import math
import pickle
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords


stemmer = PorterStemmer()
stopwords = map(lambda word: stemmer.stem(word).lower(), stopwords.words('english'))

'''
Indexes the Title and Abstract of each XML document and outputs a pickled dictionary and postings_file
'''
def build_index(input_doc_path, output_file_d, output_file_p):
    dictionary_file = open(output_file_d, 'w')
    postings_file = open(output_file_p, 'w')

    # Title and Abstract dict contains term -> [df, idf, pointer] mapping
    # The pointer points to the location of the term's postings list
    title_dict = {}
    abstract_dict = {}
    main_dict = {"TITLE": title_dict, "ABSTRACT": abstract_dict}
    
    # Title and Abstract doc dict contains doc_id -> [pointer, offset] mapping
    # The pointer points to the location of the document's doc_dict (i.e. term -> tf mapping)
    title_doc_dict = {}
    abstract_doc_dict = {}
    main_doc_dict = {"TITLE": title_doc_dict, "ABSTRACT": abstract_doc_dict}
    
    # Doc norm factors contains doc_id -> norm factor mapping
    doc_norm_factors = {}

    # Postings lists contains temp_pointer -> [(doc_id, tf), (doc_id, tf), ..]
    postings_lists = {}

    doc_names = get_doc_names(input_doc_path)
    for doc_name in doc_names:
        # Constructs document path, parses it with xml etree library and gets its root
        path = input_doc_path + '/' + doc_name
        tree = ET.parse(path)
        root = tree.getroot()

        # Removes .xml extension to shorten doc_id
        doc_name = remove_ext(doc_name)
        doc_norm_factor = 0

        for child in root:
            name = child.attrib['name'].upper()
            if name == "TITLE" or name == "ABSTRACT":
                doc_dict = get_doc_dict(child.text)
                main_doc_dict[name][doc_name] = doc_dict # main_doc_dict is either title_doc_dict or abstract_doc_dict
                
                for term, tf in doc_dict.iteritems():
                    doc_wt = 1 + math.log10(tf)
                    doc_norm_factor += math.pow(doc_wt, 2)

                    # Updating title_dict or abstract_dict
                    dictionary = main_dict[name] # dictionary is either title_dict or abstract_dict
                    if term not in dictionary:
                        temp_pointer = len(postings_lists)
                        dictionary[term] = [1, 0, temp_pointer]
                        postings_lists[temp_pointer] = [(doc_name, tf)]
                    else:
                        df = dictionary[term][0]
                        temp_pointer = dictionary[term][2]
                        dictionary[term] = [df + 1, 0, temp_pointer]
                        postings_lists[temp_pointer].append((doc_name, tf))
        doc_norm_factors[doc_name] = math.pow(doc_norm_factor, 0.5)

    # Update idf, change temp_pointer to real pointer, write postings and doc_dicts to postings_file
    main_dict = compute_idf(main_dict, len(doc_names))
    main_dict = write_postings(main_dict, postings_lists, postings_file)
    main_doc_dict = write_doc_dict(main_doc_dict, postings_lists, postings_file)

    # Include additional dictionaries to be pickled
    main_dict["DOCUMENT_NORM_FACTORS"] = doc_norm_factors
    main_dict["MAIN_DOC_DICT"] = main_doc_dict
    write_dictionary(main_dict, dictionary_file)

    dictionary_file.close()
    postings_file.close()


'''
Computes the idf value for each term using its corresponding df value
N represents the total number of documents indexed
'''
def compute_idf(main_dict, N):
    N = float(N)
    for dictionary in main_dict.itervalues(): # dictionary is either title_dict or abstract_dict
        for term, lst in dictionary.iteritems():
            df = lst[0]
            idf = math.log10(N / df)
            term_ptr = lst[2]
            dictionary[term] = [df, idf, term_ptr]
    return main_dict


def write_dictionary(dictionary, dictionary_file):
    pickle.dump(dictionary, dictionary_file)


'''
Updates pointer in main_dict to actual location in postings_file and writes the corresponding
postings_list to postings_file
'''
def write_postings(main_dict, postings_lists, postings_file):
    for dictionary in main_dict.itervalues(): # dictionary is either title_dict or abstract_dict
        for term, lst in dictionary.iteritems():
            # Replaces temp_pointer with the real pointer to its postings list in postings.txt
            file_pointer = postings_file.tell()
            df = lst[0]
            idf = lst[1]
            dictionary[term] = [df, idf, file_pointer]

            # Retrieves postings_list and writes it to postings.txt
            term_pointer = lst[2]
            postings_list = postings_lists[term_pointer]
            for posting in postings_list:
                doc_name = posting[0]
                tf = posting[1]
                postings_file.write(pad_doc_name(doc_name) + " ")
                postings_file.write(pad_tf(tf) + " ")
    return main_dict


'''
Updates pointer and offset in main_doc_dict to actual location in postings_file and 
writes the corresponding doc_dict to postings_file
NOTE: main_doc_dict is below all postings in postings.txt
'''
def write_doc_dict(main_doc_dict, postings_lists, postings_file):
    for dictionary in main_doc_dict.itervalues(): # dictionary is either title_doc_dict or abstract_doc_dict
        for doc_name, doc_dict in dictionary.iteritems():
            file_pointer = postings_file.tell()
            postings_file.write(str(doc_dict))
            offset = postings_file.tell() - file_pointer
            dictionary[doc_name] = [file_pointer, offset]
    return main_doc_dict


def remove_ext(doc_name):
    return doc_name[:len(doc_name) - 4]


def pad_doc_name(doc_name):
    return doc_name.ljust(15) # pad whitespaces until 15 characters


def pad_tf(num):
    return str(num).ljust(5) # pad whitespaces until 5 characters


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


def get_doc_names(path):
    return os.listdir(path)


'''
Takes in text and outputs dictionary with term -> tf mapping
'''
def get_doc_dict(text):
    if type(text) is unicode:
        text = text.encode('ascii', errors = 'ignore')
    doc_dict = {}
    for word in filter(lambda word: is_valid_token(word), word_tokenize(text)):
        token = stemmer.stem(word).lower()
        if token in doc_dict:
            doc_dict[token] += 1
        else:
            doc_dict[token] = 1
    return doc_dict


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

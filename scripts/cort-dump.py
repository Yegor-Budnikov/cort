#!/usr/bin/env python

import argparse
import codecs
import logging
import pickle
import sys
import os
import subprocess

import cort
from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference import experiments
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.coreference import cost_functions
from cort.util import import_helper


__author__ = 'yegor-budnikov'
"""
Dummy counter
Writes a line every 100 documents
"""

dummy_counter_for_train = 0


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')


def parse_args():
    parser = argparse.ArgumentParser(description='Dump coreference resolution '
                                                 'models.')
    parser.add_argument('-in',
                        required=True,
                        dest='input_filename',
                        help='The input file. Must follow the format of the '
                             'CoNLL shared tasks on coreference resolution '
                             '(see http://conll.cemantix.org/2012/data.html).)')
    parser.add_argument('-out',
                        dest='output_filename',
                        required=True,
                        help='The output file the dumped data will be saved '
                             'to.')
    parser.add_argument('-extractor',
                        dest='extractor',
                        required=True,
                        help='The function to extract instances.')
    parser.add_argument('-perceptron',
                        dest='perceptron',
                        # required=True,
                        help='The perceptron to use.')
    parser.add_argument('-clusterer',
                        dest='clusterer',
                        required=True,
                        help='The clusterer to use.')
    parser.add_argument('-n_iter',
                        dest='n_iter',
                        default=5,
                        help='Number of perceptron iterations. Defaults to 5.')
    parser.add_argument('-cost_scaling',
                        dest='cost_scaling',
                        default=1,
                        help='Scaling factor of the cost function. Defaults '
                             'to 1')
    parser.add_argument('-random_seed',
                        dest='seed',
                        default=23,
                        help='Random seed for training data shuffling. '
                             'Defaults to 23.')
    parser.add_argument('-features',
                        dest='features',
                        help='The file containing the list of features. If not'
                             'provided, defaults to a standard set of'
                             'features.')

    return parser.parse_args()

from cort.core.mentions import Mention
from cort.core.spans import Span

class SimpleMention:
    def __init__(self, document, span):
        """ Initialize a mention in a document.

        Args:
            document (CoNLLDocument): The document the mention belongs to.
            span (Span): The span of the mention in its document.

        """
        self.document = document
        self.span = span

    def __str__(self):
        return str(self.document) + ": [" + str(self.span.begin) + ", " + str(self.span.end) + "]"


    def __repr__(self):
        return str(self.document) + ": [" + str(self.span.begin) + ", " + str(self.span.end) + "]"


def simplify_mention(mention):
    try:
        return SimpleMention(mention.document.identifier, mention.span)
    except TypeError:
        return SimpleMention("Alas", Span(-1, -1))


def simplify_arc(arc):
    return (simplify_mention(arc[0]), simplify_mention(arc[1]))

def simplify_substructure(substructure):
    simplified = []
    for arc in substructure:
        simplified.append(simplify_arc(arc))
    return simplified

def simplify_all_substructures(substructures):
    simplified = []
    for substructure in substructures:
        simplified.append(simplify_substructure(substructure))
    return simplified

def simplify_arc_information(arc_information):
    simplified_nonnumeric_information = {}
    simplified_numeric_information = {}
    simplified_value_information = {}
    for arc in arc_information.keys():
        all_nonnumeric_feats, all_numeric_feats, numeric_vals = arc_information[arc][0]
        if all_nonnumeric_feats != []:
            simplified_nonnumeric_information[simplify_arc(arc)] = all_nonnumeric_feats
        if all_numeric_feats != []:
            simplified_numeric_information[simplify_arc(arc)] = all_numeric_feats
        if numeric_vals != []:
            simplified_value_information[simplify_arc(arc)] = numeric_vals
    return simplified_nonnumeric_information, simplified_numeric_information, simplified_value_information


args = parse_args()

if args.features:
    mention_features, pairwise_features = import_helper.get_features(
        args.features)
else:
    mention_features = [
        features.fine_type,
        features.gender,
        features.number,
        features.sem_class,

        features.compreno_sem_class,
        features.compreno_surf_slot,

        features.deprel,
        features.head_ner,
        features.length,
        features.head,
        features.first,
        features.last,
        features.preceding_token,
        features.next_token,
        features.governor,
        features.ancestry
    ]

    pairwise_features = [
        features.exact_match,
        features.head_match,

        features.compreno_sem_class_exact_match,
        features.compreno_surf_slot_exact_match,
        features.compreno_common_sem_anscestor,

        features.same_speaker,
        features.alias,
        features.sentence_distance,
        features.embedding,
        features.modifier,
        features.tokens_contained,
        features.head_contained,
        features.token_distance
    ]


# perceptron = import_helper.import_from_path(args.perceptron)(
#     cost_scaling=int(args.cost_scaling),
#     n_iter=int(args.n_iter),
#     seed=int(args.seed)
# )

# extractor = instance_extractors.InstanceExtractor(
#     import_helper.import_from_path(args.extractor),
#     mention_features,
#     pairwise_features,
#     cost_functions.null_cost,
#     perceptron.get_labels()
# )

logging.info("Reading in data.")
training_corpus = corpora.Corpus.from_file("training", codecs.open(args.input_filename, "r", "utf-8"))



logging.info("Extracting system mentions.")
dummy_counter_for_train = 0
for doc in training_corpus:
    if dummy_counter_for_train % 100 == 99:
        logging.info("We are extracting doc " + str(dummy_counter_for_train) + ": " + doc.identifier)
    dummy_counter_for_train += 1
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)


# logging.info("\tVerifying attributes.")
# for doc in training_corpus:
#     doc.antecedent_decisions = {}
#     print(doc, doc.antecedent_decisions)
#     for mention in doc.system_mentions:
#         if not "antecedent" in mention.attributes.keys():
#             mention.attributes["antecedent"] = None
#         if not "set_id" in mention.attributes.keys():
#             mention.attributes["set_id"] = None
#
# logging.info("\tExtracting instances and features.")
# substructures, arc_information = extractor.extract(training_corpus)
#
# simplified_substructures = simplify_all_substructures(substructures)
#
# a,b,c = simplify_arc_information(arc_information)
#
# for i in a:
#     print (i)
#
# logging.info("Done.")



logging.info("Writing model to file.")
pickle.dump(model, open(args.output_filename, "wb"), protocol=2)

logging.info("Done.")

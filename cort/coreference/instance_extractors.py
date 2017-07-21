""" Extract instances and features from a corpus. """

import array
import multiprocessing
import sys
import logging

import mmh3
import numpy

import pickle
from os import path
import os

__author__ = 'smartschat'

"""
Dummy counter
Writes a line every 100 documents
"""

percentage = {
    99:"1%",
    199:"2,5%",
    299:"4,3%",
    399: "4,5%",
    499: "4,5% again :)",
    599: "4,8%",
    699: "5,5%",
    799: "6,6%",
    899: "7,5%",
    999: "7,5% agian :)",
    1099: "8%",
    1199: "8,6%",
    1299: "9%",
    1399: "9,3% -- a big one -- next 18,6% in 4,5 hours",
    1499: "18,6% -- a big one -- next 29,1% in 5 hours and a quater",
    1599: "29,1%",
    1699: "35,1%",
    1799: "41%",
    1899: "49% -- a big one - next 78,7% in 15 hours",
    1999: "78,7%",
    2099: "79%",
    2199: "79,7%",
    2299: "81,4%",
    2399: "86,8%",
    2499: "89%",
    2599: "91%",
    2699: "95,5%",
    2799: "99,7%"
}


dummy_counter = 0

def pDump(data, dataname, mode):
    filename = '/home/redll/cort/my_test/pickle_files/' + dataname + '.pickle'
    os.makedirs(path.dirname(filename), exist_ok=True)

    f = open(filename, mode)
    pickle.dump(data, f)
    f.close()

def pLoad(dataname, mode):
    filename = '/home/redll/cort/my_test/pickle_files/' + dataname + '.pickle'
    with open(filename, mode) as f:
        data_new = pickle.load(f)
    return data_new


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')

# for python 2 multiprocessing
def unwrap_extract_doc(arg, **kwarg):
    return InstanceExtractor._extract_doc(*arg, **kwarg)


class InstanceExtractor:
    """ Extract instances and their corresponding features from a corpus.

    Attributes:
        extract_substructures (function: CoNLLDocument ->
             list(list((Mention,Mention)))): Function for extracting the search
             space for a coreference resolution approach. The ith list in the
             nested list contains the search space for the ith substructure.
             The search space is represented as a nested list of mention pairs,
             which are candidate arcs in the graph to predict.
        mention_features (list(function: Mention -> str)): A list of features
            for mentions.
        pairwise_features (list(function: (Mention, Mention) -> str)): A list
            of features for mention pairs.
        cost_function (function: (Mention, Mention) -> int): A function
            assigning costs to mention pairs.
        labels (list(str)): A list of arc labels emplyoed by the approach.
            Defaults to the list containing only "+".
        convert_to_string_function (function): The function used to convert
            feature values to (unicode) strings. For Python 2 it is
            ``unicode``, for Python 3 it is ``string``.
    """
    def __init__(self,
                 extract_substructures,
                 mention_features,
                 pairwise_features,
                 cost_function,
                 labels=("+",)):
        """ Initialize instance and feature extraction.

        Args:
            extract_substructures (function: CoNLLDocument ->
                list(list((Mention,Mention)))): Function for extracting the
                search space for a coreference resolution approach. The ith
                list in the nested list contains the search space for the ith
                substructure. The search space is represented as a nested list
                of mention pairs, which are candidate arcs in the graph to
                predict.
            mention_features (list(function: Mention -> str)): A list of
                features for mentions.
            pairwise_features (list(function: (Mention, Mention) -> str)): A
                list of features for mention pairs.
            cost_function (function: (Mention, Mention) -> int): A function
                assigning costs to mention pairs.
            labels (list(str)): A list of arc labels emplyoed by the
                approach.
        """
        self.extract_substructures = extract_substructures
        self.mention_features = mention_features
        self.pairwise_features = pairwise_features
        self.cost_function = cost_function
        self.labels = labels

        if sys.version_info[0] == 2:
            self.convert_to_string_function = unicode
        else:
            self.convert_to_string_function = str

    def extract(self, corpus):
        # logging.info("\tWe are in\n")
        """ Extract instances and features from a corpus.

        Args:
            corpus (Corpus): The corpus to extract instances and features from.

        Returns:
            A tuple which describes the extracted instances and their
            features. The individual components are:

            * substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            * arc_information (dict((Mention, Mention),
                                    ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
        """

        all_substructures = []
        arc_information = {}

        id_to_doc_mapping = {}

        debug_number_iterator = 0
        # logging.info("\tWe are starting\n")
        for doc in corpus:
            # if debug_number_iterator % 100 == 0: logging.info(debug_number_iterator)
            id_to_doc_mapping[doc.identifier] = doc
            debug_number_iterator += 1
        # logging.info("\tWe have finished\n")

        # pool = multiprocessing.Pool(maxtasksperchild=1)
        #
        # if sys.version_info[0] == 2:
        #     results = pool.map(unwrap_extract_doc,
        #                        zip([self] * len(corpus.documents),
        #                            corpus.documents))
        # else:
        #     results = pool.map(self._extract_doc, corpus.documents)
        #
        # pool.close()
        # pool.join()

        global dummy_counter
        dummy_counter = 0
        results = [self._extract_doc(doc) for doc in corpus.documents]
        # results = [doc.identifier for doc in corpus.documents]

        logging.info("\tWe have finished biiig cycle\n")

        num_labels = len(self.labels)

        dummy_counter = 0
        for result in results:
            (doc_identifier,
             anaphors,
             antecedents,
             nonnumeric_features,
             numeric_features,
             numeric_vals,
             costs,
             consistency,
             nonnumeric_feature_mapping,
             numeric_feature_mapping,
             substructures_mapping) = result
            # substructures_mapping) = pLoad(result, "rb")

            #print(result[0] + ": " + str(sys.getsizeof(result)))
            # print(sys.getsizeof(all_substructures)/1000000)
            # print(sys.getsizeof(arc_information)/1000000)
            doc = id_to_doc_mapping[doc_identifier]

            if dummy_counter % 100 == 99:
                logging.info("\tFor some reason we are mapping something: " + str(dummy_counter) + "\n")
            dummy_counter += 1

            for i in range(0, len(substructures_mapping) - 1):
                struct = []
                begin = substructures_mapping[i]
                end = substructures_mapping[i + 1]

                for pair_index in range(begin, end):
                    arc = (doc.system_mentions[anaphors[pair_index]],
                           doc.system_mentions[antecedents[pair_index]])

                    struct.append(arc)

                    # find position of arc's features in document array
                    nonnumeric_features_start = nonnumeric_feature_mapping[
                        pair_index]
                    nonnumeric_features_end = nonnumeric_feature_mapping[
                        pair_index + 1]

                    numeric_features_start = numeric_feature_mapping[pair_index]
                    numeric_features_end = numeric_feature_mapping[
                        pair_index + 1]

                    arc_information[arc] = \
                        ((nonnumeric_features[
                          nonnumeric_features_start:nonnumeric_features_end
                          ],
                          numeric_features[
                          numeric_features_start:numeric_features_end
                          ],
                          numeric_vals[
                          numeric_features_start:numeric_features_end
                          ]),
                         costs[
                         num_labels * pair_index:num_labels * pair_index
                         + num_labels],
                         consistency[pair_index])
                    # curr_arc_information = (arc,
                    #     ((nonnumeric_features[
                    #       nonnumeric_features_start:nonnumeric_features_end
                    #       ],
                    #       numeric_features[
                    #       numeric_features_start:numeric_features_end
                    #       ],
                    #       numeric_vals[
                    #       numeric_features_start:numeric_features_end
                    #       ]),
                    #      costs[
                    #      num_labels * pair_index:num_labels * pair_index
                    #                              + num_labels],
                    #      consistency[pair_index]))

                # pDump(struct, "all_substructures", "ab")
                # pDump(curr_arc_information, "arc_information", "ab")
                all_substructures.append(struct)
        #     (doc_identifier,
        #      anaphors,
        #      antecedents,
        #      nonnumeric_features,
        #      numeric_features,
        #      numeric_vals,
        #      costs,
        #      consistency,
        #      nonnumeric_feature_mapping,
        #      numeric_feature_mapping,
        #      substructures_mapping) = ([],[],[],[],[],[],[],[],[],[],[])
        # in python 2, array.array does not support the buffer interface

        dummy_counter = 0

        if sys.version_info[0] == 2:
            for arc in arc_information:
                feats, cost, cons = arc_information[arc]
                arc_information[arc] = (
                    (numpy.array(feats[0], dtype=numpy.uint32),
                     numpy.array(feats[1], dtype=numpy.uint32),
                     numpy.array(feats[2], dtype="float32")),
                    numpy.array(cost, dtype=float),
                    cons)

        return all_substructures, arc_information

    def _extract_doc(self, doc):
        global dummy_counter
        if dummy_counter % 50 == 49:
            logging.info("We are extracting doc " + str(dummy_counter) + ": " + doc.identifier + "\n")
            if dummy_counter in percentage.keys():
                logging.info("\tCurrent progress: " + str(percentage[dummy_counter]))
        dummy_counter += 1
        cache = {}
        substructures = self.extract_substructures(doc)

        mentions_to_ids = {}

        for i, mention in enumerate(doc.system_mentions):
            mentions_to_ids[mention] = i

        anaphors = array.array('H')
        antecedents = array.array('H')
        costs = array.array('H')
        consistency = array.array('B')
        nonnumeric_feature_mapping = array.array('I')
        numeric_feature_mapping = array.array('I')
        substructures_mapping = array.array('I')
        nonnumeric_features = array.array('I')
        numeric_features = array.array('I')
        numeric_vals = array.array("f")

        nonnumeric_feature_mapping.append(0)
        numeric_feature_mapping.append(0)
        substructures_mapping.append(0)

        for struct in substructures:
            # skip empty
            if not struct:
                continue

            for arc in struct:
                # ids for anaphor and antecedent
                anaphors.append(mentions_to_ids[arc[0]])
                antecedents.append(mentions_to_ids[arc[1]])

                # cost for each label
                for label in self.labels:
                    costs.append(self.cost_function(arc, label))

                # is decision to make them coreferent consistent with gold?
                consistency.append(arc[0].decision_is_consistent(arc[1]))

                # features for the arc: stored in array which applies to whole
                # document
                (arc_nonnumeric_features, arc_numeric_features,
                 arc_numeric_vals) = self._extract_features(arc, cache)

                nonnumeric_features.extend(arc_nonnumeric_features)
                numeric_features.extend(arc_numeric_features)
                numeric_vals.extend(arc_numeric_vals)

                # auxiliary arrays that store the position of features for arcs
                # in the document array
                nonnumeric_feature_mapping.append(
                    nonnumeric_feature_mapping[-1] + len(
                        arc_nonnumeric_features))
                numeric_feature_mapping.append(
                    numeric_feature_mapping[-1] + len(arc_numeric_features))

            # store position of substructures in document array
            substructures_mapping.append(substructures_mapping[-1] +
                                         len(struct))
        # logging.info("We have extracted doc\n")

        # cResult = (doc.identifier,
        #         anaphors,
        #         antecedents,
        #         nonnumeric_features,
        #         numeric_features,
        #         numeric_vals,
        #         costs,
        #         consistency,
        #         nonnumeric_feature_mapping,
        #         numeric_feature_mapping,
        #         substructures_mapping)
        # pDump(cResult, doc.identifier, "wb")

        # return doc.identifier

        return (doc.identifier,
                anaphors,
                antecedents,
                nonnumeric_features,
                numeric_features,
                numeric_vals,
                costs,
                consistency,
                nonnumeric_feature_mapping,
                numeric_feature_mapping,
                substructures_mapping)

    def _extract_features(self, arc, cache):
        anaphor, antecedent = arc
        inst_feats = []
        numeric_features = []

        numeric_types = {"float", "int"}

        if not antecedent.is_dummy():
            # mention features
            for mention in [anaphor, antecedent]:
                if mention not in cache:
                    # cache[mention] = [feature(mention) for feature
                    #                   in self.mention_features]
                    cache[mention] = []
                    for feature in self.mention_features:
                        if type(feature(mention)).__name__ == "tuple":
                            cache[mention].append(feature(mention))
                        if type(feature(mention)).__name__ == "list":
                            cache[mention].extend(feature(mention))

            ana_features = cache[anaphor]
            ante_features = cache[antecedent]

            # first: non-numeric features (categorial, boolean)
            inst_feats += ["ana_" + feat + "=" +
                           self.convert_to_string_function(val) for feat, val in
                           ana_features if type(val).__name__ not in
                           numeric_types]

            len_ana_features = len(inst_feats)

            inst_feats += ["ante_" + feat + "=" +
                           self.convert_to_string_function(val) for feat, val in
                           ante_features if type(val).__name__ not in
                           numeric_types]

            # concatenated features
            inst_feats += ["ana_" + ana_info[0] + "=" +
                           self.convert_to_string_function(ana_info[1]) +
                           "^ante_" + ante_info[0] + "=" +
                           self.convert_to_string_function(ante_info[1])
                           for ana_info, ante_info in
                           zip(ana_features, ante_features)]

            # pairwise features
            pairwise_features = [feature(anaphor, antecedent) for feature
                                 in self.pairwise_features]
            inst_feats += [feature + "=" +
                           self.convert_to_string_function(val) for feature, val
                           in pairwise_features
                           if val and type(val).__name__ not in numeric_types]

            # feature combinations
            fine_type_indices = {len_ana_features * i for i
                                 in [0, 1, 2]}

            inst_feats += [
                inst_feats[i] + "^" + word for i in fine_type_indices
                for j, word in enumerate(inst_feats)
                if j not in fine_type_indices
            ]

            # now numeric features
            ana_numeric = [("ana_" + feat, val) for feat, val
                           in ana_features
                           if type(val).__name__ in numeric_types]
            ante_numeric = [("ante_" + feat, val) for feat, val
                            in ante_features
                            if type(val).__name__ in numeric_types]
            pair_numeric = [(feat, val) for feat, val in pairwise_features
                            if type(val).__name__ in numeric_types]

            # feature combinations for numeric features
            for numeric_features in [ana_numeric, ante_numeric, pair_numeric]:
                numeric_features += [
                    (inst_feats[i] + "^" + numeric_features[j][0],
                     numeric_features[j][1]) for i in fine_type_indices
                    for j, numeric_feature in enumerate(numeric_features)
                ]

            numeric_features = ana_numeric + ante_numeric + pair_numeric

        # to hash
        all_nonnumeric_feats = array.array(
            'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word
                  in inst_feats])
        all_numeric_feats = array.array(
            'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word, _
                  in numeric_features])
        numeric_vals = array.array("f", [val for _, val in numeric_features])

        # features hashes
        # for word in inst_feats:
        #     self.feature_table.add((word, mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 ))
        # for word, _ in numeric_features:
        #     self.feature_table.add((word, mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 ))
        # for _, val in numeric_features:
        #     self.feature_table.add((val, val))

        # features counters
        # self.feature_counter = {}
        # for word in inst_feats:
        #     if word in self.feature_counter.keys():
        #         self.feature_counter[word] += 1
        #     else:
        #         self.feature_counter[word] = 1
        # for word, _ in numeric_features:
        #     if word in self.feature_counter.keys():
        #         self.feature_counter[word] += 1
        #     else:
        #         self.feature_counter[word] = 1
        # for _, val in numeric_features:
        #     if val in self.feature_counter.keys():
        #         self.feature_counter[val] += 1
        #     else:
        #         self.feature_counter[val] = 1

        return all_nonnumeric_feats, all_numeric_feats, numeric_vals

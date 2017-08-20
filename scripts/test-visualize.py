from __future__ import print_function

import subprocess
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')
__author__ = 'yegor.budnikov'


# systems = ["pair", "closest", "latent", "tree"]
# data_sets = ["dev", "test"]

systems = ["latent"]
# systems = ["pair", "closest", "latent", "tree"]
# systems = ["tree"]
# data_sets = ["dev-english", "test-english"]
data_sets = ["dev-english"]
# data_sets = ["train-english", "dev-english", "test-english"]

# "C:\\Users\\ebudnikov\\AppData\\Local\\Continuum\\Miniconda3_1\\python.exe",
# "-in", "/home/redll/cort/my_test/try.conll",

# for system in systems:
#     for data_set in data_sets:
#         print("Visualizing", system, "on", data_set)
#         # if data_set == "dev-english":
#         #     model = "E:\\buML\\cort\\data\\models\\gold-english-train\\model-" + system + "-train.obj"
#         # else:
#         #     model = "E:\\buML\\cort\\data\\models\\gold-english-train+dev\\model-" + system + "-train+dev.obj"
#
#         model = "E:\\buML\\cort\\data\\models\\gold-english-train\\model-" + system + "-train.obj"
#         print("model = ", model)
#         subprocess.call([
#             "C:\\Anaconda\\python.exe",
#             "E:\\buML\\cort\\src\\bin\\cort-visualize",
#             # "-corenlp", "smsht",
#             "-h"])
#             # "-in", "E:\\buML\\cort\\data\\sets\\" + data_set + ".auto",
#             # "-gold", "E:\\buML\\cort\\data\\sets\\" + data_set + ".gold",
#             # "-model", model,
#             # "-out", "model-" + system + "-output",
#             # "-ante", "output.antecedents",
#             # "-extractor", get_extractor(data_set, system),
#             # "-perceptron", get_perceptron(system),
#             # "-clusterer", get_clusterer(system)])

from cort.core import corpora
# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')


logging.info("\t0\tExtracting reference")

reference = corpora.Corpus.from_file("reference", open("E:\\buML\\cort\\data\\sets\\test-english.gold"))


logging.info("\t1")
'''
logging.info("\tExtracting model-pair-output")
pair = corpora.Corpus.from_file("pair", open("E:\\buML\\cort\\data\\results\\baseline\\model-pair-output"))

logging.info("\tExtracting model-closest-output")
closest = corpora.Corpus.from_file("closest", open("E:\\buML\\cort\\data\\results\\baseline\\model-closest-output"))
'''
logging.info("\tExtracting model-latent-output")
latent = corpora.Corpus.from_file("latent", open("E:\\buML\\cort\\data\\results\\report_results\\WIN_model-latent-output_jul17"))
latent_no_sem_class = corpora.Corpus.from_file("latent_no_sem_class", open("E:\\buML\\cort\\data\\results\\baseline\\model-latent-output"))

'''
logging.info("\tExtracting model-tree-output")
tree = corpora.Corpus.from_file("tree", open("E:\\buML\\cort\\data\\results\\baseline\\model-tree-output"))

'''
logging.info("\t2")

# optional -- not needed when you only want to compute recall errors
# pair.read_antecedents(open('E:\\buML\\cort\\data\\results\\baseline\\model-pair-output.antecedents'))
# closest.read_antecedents(open('E:\\buML\\cort\\data\\results\\baseline\\model-closest-output.antecedents'))
latent.read_antecedents(open('E:\\buML\\cort\\data\\results\\report_results\\WIN_model-latent-output_jul17.antecedents'))
latent_no_sem_class.read_antecedents(open('E:\\buML\\cort\\data\\results\\baseline\\model-latent-output.antecedents'))
# tree.read_antecedents(open('E:\\buML\\cort\\data\\results\\baseline\\model-tree-output.antecedents'))

from cort.analysis import error_extractors
from cort.analysis import spanning_tree_algorithms


logging.info("\t3")
extractor = error_extractors.ErrorExtractor(
    reference,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)


logging.info("\t1\textractor.add_system")

#extractor.add_system(pair)
#extractor.add_system(closest)
extractor.add_system(latent)
extractor.add_system(latent_no_sem_class)
#extractor.add_system(tree)


logging.info("\t2\textractor.get_errors")

errors = extractor.get_errors()

# print(errors)
logging.info("\t3")

# errors_by_type = errors.categorize(
#     lambda error: error[0].attributes['semantic_class']
# )

all_gold = set()
for doc in reference:
    for mention in doc.annotated_mentions:
        all_gold.add(mention)

all_extracted = set()
for doc in latent:
    for mention in doc.annotated_mentions:
        all_extracted.add(mention)


def is_anaphor_gold(mention):
    if mention in all_gold:
        return "is_gold"
    else:
        return "is_not_gold"

def is_anaphor_extracted(mention1, mention2):
    if not mention1 in all_gold and not mention2 in all_gold and mention1 in all_extracted and mention2 in all_extracted:
        return "so_spurius"
    if mention1 in all_gold and mention2 in all_gold and not mention1 in all_extracted and not mention2 in all_extracted:
        return "so_missing"
    if mention1 in all_gold and mention1 in all_extracted and mention2 in all_gold and mention2 in all_extracted:
        return "is_extracted_and_gold"
    elif mention1 in all_gold and not mention2 in all_extracted:
        return "ante_is_missing_in_extracted"
    elif not mention1 in all_gold and mention2 in all_extracted:
        return "mention_is_spurius_in_extracted"
    elif mention2 in all_gold and not mention1 in all_extracted:
        return "mention_is_missing_in_extracted"
    elif not mention2 in all_gold and mention1 in all_extracted:
        return "ante_is_spurius_in_extracted"
    else:
        return "wat"

errors_by_type = errors.categorize(lambda err: err[0].attributes['type'][0])
errors_by_gram = errors.categorize(lambda err: err[0].attributes['grammatical_function'][0])

errors_by_type1 = errors.filter(lambda error: error[0].attributes['type'] == "NAM" and
                                error[1].attributes['type'] == "NOM")


logging.info("\t3\terrors_by_type")

# print(errors_by_type)

# precision_errs = errors_by_type["latent"]["precision_errors"]["all"]
# recall_errs = errors_by_type["latent"]["recall_errors"]["all"]
#
# precision_errs_pair = errors_by_type["pair"]["precision_errors"]["all"]
#recall_errs_pair = errors_by_type["pair"]["recall_errors"]["all"]

# precision_errs_closest = errors_by_type["closest"]["precision_errors"]["all"]
#recall_errs_closest = errors_by_type["closest"]["recall_errors"]["all"]

# precision_errs_latent = errors_by_type["latent"]["precision_errors"]["all"]
recall_errs_latent = errors_by_gram["latent"]["recall_errors"]["all"]
recall_errs_latent_no_sem_class = errors_by_gram["latent_no_sem_class"]["recall_errors"]["all"]

# precision_errs_tree = errors_by_type["tree"]["precision_errors"]["all"]
#recall_errs_tree = errors_by_type["tree"]["recall_errors"]["all"]

# dictt = dict()
# # print(precision_errs.keys())
# for key in precision_errs.keys():
#     dictt[key] = 0
# for key in recall_errs.keys():
#     dictt[key] = 0
#
# for key in precision_errs.keys():
#     dictt[key] += len(precision_errs[key])
# for key in recall_errs.keys():
#     dictt[key] += len(recall_errs[key])

# ante_is_spurius_in_extracted = precision_errs['ante_is_spurius_in_extracted']
# is_extracted_and_gold = precision_errs['is_extracted_and_gold']
# mention_is_spurius_in_extracted = precision_errs['mention_is_spurius_in_extracted']
# so_spurius = precision_errs['so_spurius']
#
#
# ante_is_missing_in_extracted = recall_errs['ante_is_missing_in_extracted']
# rec_is_extracted_and_gold = recall_errs['is_extracted_and_gold']
# mention_is_missing_in_extracted = recall_errs['mention_is_missing_in_extracted']
# so_missing = recall_errs['so_missing']

# for key in dictt.keys():
#     print(key)
#     print(dictt[key])
#
# logging.info("\tprecision_errs")
# print(len(precision_errs))
#
# logging.info("\trecall_errs")
# print(len(recall_errs))

# print('is_extracted_and_gold')
# print(len(is_extracted_and_gold) + len(rec_is_extracted_and_gold))
#
# print('ante_is_spurius_in_extracted')
# print(len(ante_is_spurius_in_extracted))
#
# print('ante_is_missing_in_extracted')
# print(len(recall_errs))
#
# print('mention_is_spurius_in_extracted')
# print(len(mention_is_spurius_in_extracted))
#
# print('mention_is_missing_in_extracted')
# print(len(mention_is_missing_in_extracted))
#
# print('so_spurius')
# print(len(so_spurius))
#
# print('so_missing')
# print(len(so_missing))




print('')
logging.info("\t5\tvisualize")


#errors_by_type.visualize("pair")
#errors_by_type.visualize("closest")
errors_by_type.visualize("latent")
errors_by_type.visualize("latent_no_sem_class")


#errors_by_type.visualize("tree")


logging.info("\t6\tplot")

from cort.analysis import plotting

# plotting.plot(
#     [("pair", [(cat, len(errs)) for cat, errs in recall_errs_pair.items()]),
#      ("latent", [(cat, len(errs)) for cat, errs in recall_errs_latent.items()])],
#     "Recall Errors",
#     "Type of anaphor",
#     "Number of Errors")
plotting.plot(
    [("sem_class_latent", [(cat, len(errs)) for cat, errs in recall_errs_latent.items()]),
     ("no_sem_class_latent", [(cat, len(errs)) for cat, errs in recall_errs_latent_no_sem_class.items()])],
    "Recall Errors",
    "Type of anaphor",
    "Number of Errors",
    filename="E:\\buML\\cort\\data\\results\\report_results\\WIN_model-latent-output_jul17.png")
print("!")
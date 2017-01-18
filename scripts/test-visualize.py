from __future__ import print_function

import subprocess


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

print(0)

reference = corpora.Corpus.from_file("reference", open("E:\\buML\\cort\\data\\sets\\test-english.gold"))

print(1)

latent = corpora.Corpus.from_file("latent", open("model-latent-output"))
# tree = corpora.Corpus.from_file("tree", open("tree-output.conll"))

print(2)

# optional -- not needed when you only want to compute recall errors
latent.read_antecedents(open('model-latent-output.antecedents'))

from cort.analysis import error_extractors
from cort.analysis import spanning_tree_algorithms

print(3)
extractor = error_extractors.ErrorExtractor(
    reference,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)

print(1)

extractor.add_system(latent)

print(2)

errors = extractor.get_errors()

print(errors)
print(3)

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

errors_by_type = errors.categorize(lambda err: is_anaphor_extracted(err[0], err[1]))


print(errors_by_type)

print(4)

errors_by_type.visualize("latent")


print("!")
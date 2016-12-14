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

for system in systems:
    for data_set in data_sets:
        print("Visualizing", system, "on", data_set)
        # if data_set == "dev-english":
        #     model = "E:\\buML\\cort\\data\\models\\gold-english-train\\model-" + system + "-train.obj"
        # else:
        #     model = "E:\\buML\\cort\\data\\models\\gold-english-train+dev\\model-" + system + "-train+dev.obj"

        model = "E:\\buML\\cort\\data\\models\\gold-english-train\\model-" + system + "-train.obj"
        print("model = ", model)
        subprocess.call([
            "C:\\Anaconda\\python.exe",
            "E:\\buML\\cort\\src\\bin\\cort-visualize",
            # "-corenlp", "smsht",
            "-h"])
            # "-in", "E:\\buML\\cort\\data\\sets\\" + data_set + ".auto",
            # "-gold", "E:\\buML\\cort\\data\\sets\\" + data_set + ".gold",
            # "-model", model,
            # "-out", "model-" + system + "-output",
            # "-ante", "output.antecedents",
            # "-extractor", get_extractor(data_set, system),
            # "-perceptron", get_perceptron(system),
            # "-clusterer", get_clusterer(system)])

from cort.core import corpora
# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')

reference = corpora.Corpus.from_file("reference", open("E:\\buML\\cort\\data\\sets\\dev-english.gold"))
latent = corpora.Corpus.from_file("latent", open("model-latent-train+dev-output"))
# tree = corpora.Corpus.from_file("tree", open("tree-output.conll"))

# optional -- not needed when you only want to compute recall errors
# pair.read_antecedents(open('pair.antecedents'))

from cort.analysis import error_extractors
from cort.analysis import spanning_tree_algorithms

extractor = error_extractors.ErrorExtractor(
    reference,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)

extractor.add_system(latent)

errors = extractor.get_errors()

errors_by_type = errors.categorize(
    lambda error: error[0].attributes['type']
)

errors_by_type.visualize("latent")


print("!")
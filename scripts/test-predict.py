from __future__ import print_function

import subprocess


__author__ = 'smartschat'


def get_extractor(data_set, system):
    if system == "closest" or system == "latent":
        return "cort.coreference.approaches.mention_ranking.extract_substructures"
    elif system == "tree":
        return "cort.coreference.approaches.antecedent_trees.extract_substructures"
    elif system == "pair":
        if data_set == "train":
            return "cort.coreference.approaches.mention_pairs" \
                   ".extract_training_substructures"
        else:
            return "cort.coreference.approaches.mention_pairs" \
                   ".extract_testing_substructures"


def get_perceptron(system):
    if system == "pair":
        return "cort.coreference.approaches.mention_pairs.MentionPairsPerceptron"
    elif system == "closest":
        return "cort.coreference.approaches.mention_ranking.RankingPerceptronClosest"
    elif system == "latent":
        return "cort.coreference.approaches.mention_ranking.RankingPerceptron"
    elif system == "tree":
        return "cort.coreference.approaches.antecedent_trees.AntecedentTreePerceptron"


def get_cost_function(system):
    if system == "pair":
        return "cort.coreference.cost_functions.null_cost"
    else:
        return "cort.coreference.cost_functions.cost_based_on_consistency"


def get_clusterer(system):
    if system == "pair":
        return "cort.coreference.clusterer.best_first"
    else:
        return "cort.coreference.clusterer.all_ante"


# systems = ["pair", "closest", "latent", "tree"]
# data_sets = ["dev", "test"]

systems = ["latent"]
# systems = ["pair", "closest", "tree"]
# systems = ["pair", "closest", "latent", "tree"]
# systems = ["tree"]
# data_sets = ["dev-english", "test-english"]
data_sets = ["new_test_compreno"]
# data_sets = ["test_compreno"]
# data_sets = ["train-english", "dev-english", "test-english"]

# "C:\\Users\\ebudnikov\\AppData\\Local\\Continuum\\Miniconda3_1\\python.exe",
# "-in", "/home/redll/cort/my_test/try.conll",

for system in systems:
#     print("Training", system, "on train.")
#     subprocess.call([
#         "C:\\Anaconda\\python.exe",
#         "E:\\buML\\cort\\src\\bin\\cort-train",
#         # "-in", "E:\\buML\\cort\\data\\sets\\train+dev-english.gold",
#         # "-out", "E:\\buML\\cort\\data\\models\\model-" + system + "-train+dev.obj",
#         "-in", "E:\\buML\\cort\\data\\sets\\train-english.gold",
#         "-out", "E:\\buML\\cort\\data\\models\\model-" + system + "-train.obj",
#         "-extractor", get_extractor("train", system),
#         "-perceptron", get_perceptron(system),
#         "-cost_function", get_cost_function(system),
#         "-cost_scaling", "100"])


    for data_set in data_sets:
        print("Predicting", system, "on", data_set)
        if data_set in ["dev-english"]:
            model = "E:\\buML\\cort\\data\\models\\gold-english-train\\model-" + system + "-train.obj"
        elif data_set in ["test-english"]:
            model = "E:\\buML\\cort\\data\\models\\gold-english-train+dev\\model-" + system + "-train+dev.obj"
        elif data_set in ["dev_compreno"]:
            model = "E:\\buML\\cort\\data\\models\\compreno-train\\model-" + system + "-train_compreno.obj"
        elif data_set in ["test_compreno"]:
            model = "E:\\buML\\cort\\data\\models\\compreno-train+dev\\model-" + system + "-train+dev_compreno.obj"
        elif data_set in ["new_dev_compreno", "new_test_compreno"]:
            model = "E:\\buML\\cort\\data\\models\\new_compreno-train\\new_model-" + system + "-train_compreno.obj"
        elif data_set in ["new_test_compreno"]:
            model = "E:\\buML\\cort\\data\\models\\new_compreno-train+dev\\new_model-" + system + "-train+dev_compreno.obj"

        # model = "E:\\buML\\cort\\data\\models\\gold-english-train\\model-" + system + "-train.obj"
        print("model = ", model)
        subprocess.call([
            "C:\\Anaconda\\python.exe",
            "E:\\buML\\cort\\src\\bin\\cort-predict-conll",
            # "-in", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
            # "-gold", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
            "-in", "E:\\buML\\cort\\data\\sets\\" + data_set + ".gold",
            "-gold", "E:\\buML\\cort\\data\\sets\\" + data_set + ".gold",
            "-model", model,
            "-out", "new_compreno_model-" + system + "-output",
            "-ante", "new_compreno_model-" + system + "-output.antecedents",
            "-extractor", get_extractor(data_set, system),
            "-perceptron", get_perceptron(system),
            "-clusterer", get_clusterer(system)])

        # model = "E:\\buML\\cort\\data\\models\\gold-english-train+dev\\model-" + system + "-train+dev.obj"
        # print("model = ", model)
        # subprocess.call([
        #     "C:\\Anaconda\\python.exe",
        #     "E:\\buML\\cort\\src\\bin\\cort-predict-conll",
        #     # "-in", "/home/redll/cort/my_test/" + data_set + ".auto",
        #     # "-gold", "/home/redll/cort/my_test/" + data_set + ".gold",
        #     "-in", "E:\\buML\\cort\\data\\sets\\" + data_set + ".auto",
        #     "-gold", "E:\\buML\\cort\\data\\sets\\" + data_set + ".gold",
        #     "-model", model,
        #     "-out", "model-" + system + "-train+dev-output",
        #     "-ante", "output.antecedents",
        #     "-extractor", get_extractor(data_set, system),
        #     "-perceptron", get_perceptron(system),
        #     "-clusterer", get_clusterer(system)])

# subprocess.run([
#     "/home/redll/cort/bin/cort-predict-conll",
#     # "-in", "/home/redll/cort/my_test/" + data_set + ".auto",
#     # "-gold", "/home/redll/cort/my_test/" + data_set + ".gold",
#     "-in", "/home/redll/cort/my_test/" + data_set + "_short.auto",
#     "-gold", "/home/redll/cort/my_test/" + data_set + "_short.gold",
#     "-model", model,
#     "-out", "model-" + system + "-output",
#     "-extractor", get_extractor(data_set, system),
#     "-perceptron", get_perceptron(system),
#     "-clusterer", get_clusterer(system)])

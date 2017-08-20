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
# systems = ["pair", "closest", "latent", "tree"]
# systems = ["tree"]
# data_sets = ["dev-english", "test-english"]
data_sets = ["dev-english"]

# "C:\\Users\\ebudnikov\\AppData\\Local\\Continuum\\Miniconda3_1\\python.exe",
# "-in", "/home/redll/cort/my_test/try.conll",

for system in systems:
    print("Training", system, "on train.")
    subprocess.call([
        "C:\\Anaconda\\python.exe",
        "E:\\buML\\cort\\copy2\\bin\\cort-train",
        "-in", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
        "-out", "E:\\buML\\cort\\data\\models\\deleting_document\\short_new_model-" + system + "-train_compreno.obj",
        "-ante", "train.output.antecedents",
        "-extractor", get_extractor("train", system),
        "-perceptron", get_perceptron(system),
        "-clusterer", get_clusterer(system),
        "-n_iter", "5",
        "-dropout_input", "0",
        "-dropout_hidden_layer", "0",
        "-cost_function", get_cost_function(system),
        "-cost_scaling", "100"])

    # subprocess.call([
    #     "C:\\Anaconda\\python.exe",
    #     "E:\\buML\\cort\\copy2\\bin\\cort-predict-conll",
    #     # "-in", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
    #     # "-gold", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
    #     "-in", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
    #     "-gold", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
    #     "-model", "E:\\buML\\cort\\data\\models\\deleting_document\\short_new_model-" + system + "-train_compreno.obj",
    #     "-out", "E:\\buML\\cort\\data\\models\\deleting_document\\new_compreno_model-" + system + "-output",
    #     "-ante", "E:\\buML\\cort\\data\\models\\deleting_document\\new_compreno_model-" + system + "-output.antecedents",
    #     "-extractor", get_extractor("dev-english", system),
    #     "-perceptron", get_perceptron(system),
    #     "-clusterer", get_clusterer(system)])
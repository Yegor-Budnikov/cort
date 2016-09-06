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

systems = ["pair"]
# systems = ["latent"]
# systems = ["tree"]
# data_sets = ["dev-english", "test-english"]
data_sets = ["dev-english"]

for system in systems:
    # print("Training", system, "on train.")
    # subprocess.run([
    #     "/usr/bin/python3.5",
    #     "/home/redll/cort/bin/cort-train",
    #     # "-in", "/home/redll/cort/my_test/try.conll",
    #     "-in", "/home/redll/cort/my_test/try-short.conll",
    #     "-out", "model-" + system + "-train.obj",
    #     "-extractor", get_extractor("train", system),
    #     "-perceptron", get_perceptron(system),
    #     "-cost_function", get_cost_function(system),
    #     "-cost_scaling", "100"])


    for data_set in data_sets:
        print("Predicting", system, "on", data_set)
        if data_set == "dev-english":
            model = "model-" + system + "-train.obj"
        else:
            model = "model-" + system + "-train+dev.obj"

        subprocess.run([
            "/home/redll/cort/bin/cort-predict-conll",
            # "-in", "/home/redll/cort/my_test/" + data_set + ".auto",
            # "-gold", "/home/redll/cort/my_test/" + data_set + ".gold",
            "-in", "/home/redll/cort/my_test/" + data_set + "_short.auto",
            "-gold", "/home/redll/cort/my_test/" + data_set + "_short.gold",
            "-model", model,
            "-out", "model-" + system + "-output",
            "-extractor", get_extractor(data_set, system),
            "-perceptron", get_perceptron(system),
            "-clusterer", get_clusterer(system)])

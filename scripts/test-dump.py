from __future__ import print_function

import subprocess


__author__ = 'yegor-budnikov'


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


systems = ["pair"]
data_sets = ["new_test_compreno"]



subprocess.call([
    "C:\\Anaconda\\python.exe",
    "E:\\buML\\cort\\src\\scripts\\cort-dump.py",
    "-in", "E:\\buML\\cort\\data\\sets\\short_new_train_compreno.gold",
    "-out", "dump-" + "pair" + "-output.pickle",
    "-extractor", get_extractor("train", "pair"),
    "-perceptron", get_perceptron("pair"),
    "-clusterer", get_clusterer("pair")])
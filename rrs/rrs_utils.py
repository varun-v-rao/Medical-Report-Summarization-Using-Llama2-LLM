import pandas as pd
import os
import concurrent.futures
from tqdm import tqdm
import evaluate
import numpy as np
import time
from rouge import Rouge
import rouge

def generate_prompt_message(text, near_samples=None, interactive=False, former_good_response=None, former_bad_response=None):

    dynamic_messages = [
        {"role": "system", "content": "You are a chest radiologist that identify the main findings and diagnosis or impression based on the given\
        FINDINGS section of the chest X-ray report, which details the radiologists' assessment of the chest X-ray image. \
        Please ensure that your response is concise and does not exceed the length of the FINDINGS."}]

    if near_samples is not None:
        dynamic_messages.append({"role": "user", "content": "Here are some examples. Please learn how to write IMPRESSION in these examples, and \
        pay particular attention to the consistent use of phrases in these below examples."})
        for near in near_samples:
            dynamic_messages.append({"role": "user", "content": "What are the main findings and diagnosis or impression based on the given Finding in chest X-ray report:\
                \nFINDINGS:\n{}".format(near["finding"])})
            dynamic_messages.append({"role": "assistant", "content": "IMPRESSION:\n{}".format(near["impression"])})


    if not interactive:
        dynamic_messages.append({"role": "user", "content": "What are the main findings and diagnosis or impression based on the given Finding in \
            chest X-ray report: \nFINDINGS: {}".format(text)})


    else:
        dynamic_messages.append({"role": "user", "content": "What are the main findings and diagnosis or impression based on the given Finding in chest \
                        X-ray report: \nFINDINGS:\n{}".format(text)})

        if former_bad_response is not None:
            for index, item in enumerate(former_bad_response):
                dynamic_messages.append({"role": "user", "content": "Below is a bad impression of the FINDINGS above:"})
                dynamic_messages.append({"role": "assistant", "content": "IMPRESSION:\n{}".format(item["summarize"])})

            if len(former_good_response) == 0:
                dynamic_messages.append({"role": "user", "content": "Please give another impression of the FINDINGS above. \
                It is important to note that your answer should avoid being consistent with the bad impression above. \
                Please pay particular attention to the consistent use of phrases in the above examples"})
            # else:
            #   dynamic_messages.append({"role": "user", "content": "Please give another impression based on the above FINDINGS.\
            #         Try to refer to the good response above and avoid the same bad response as above."})

        if former_good_response is not None:
            for index, item in enumerate(former_good_response):
                dynamic_messages.append({"role": "user", "content": "Below is an excellent impression of the FINDINGS above:"})
                dynamic_messages.append({"role": "assistant", "content": "IMPRESSION:\n{}".format(item["summarize"])})
                # dynamic_messages.append({"role": "user", "content": "This is a excellent impression, please give another clearer impression in this format."})
            if len(former_bad_response) == 0:
                dynamic_messages.append({"role": "user", "content": "This is an excellent impression, \
                    please give another impression in this format, and do not exceed the length of this impression.\
                    Please pay particular attention to the consistent use of phrases in the above examples"})
            else:
                dynamic_messages.append({"role": "user", "content": "Please give another impression of the FINDINGS above. \
                    It is important to note that your answer should avoid being consistent with the bad impression above, \
                    but should be consistent with the excellent impression above, and do not exceed the length of the excellent impression. \
                    And please pay particular attention to the consistent use of phrases in the above examples"})

    return dynamic_messages


def random_sample_selection_v2(test_sample_label, label_space, m=2000, k=20):
    # MIMIC-CXR
    dist = []
    # study_ids = []
    randon_choice_corpus = label_space
    # randon_choice_corpus = label_space.sample(n=m)
    for index, row in randon_choice_corpus.iterrows():
        # print(row[2:])
        # if (np.array(row[2:]) == example_label).all() and row["study_id"] != test_study_id:
        label = np.array(row[2:])
        # label[np.isnan(label)] = 2
        eucliDist = np.sqrt(sum(np.power((label - test_sample_label), 2)))
        dist.append(eucliDist)

    np_dist_scores = np.array(dist)
    sort_index = np.argsort(np_dist_scores)

    bottom_index = sort_index[:k]
    best_examples, best_dist, best_label = [], [], []
    for index in bottom_index:
        best_examples.append(randon_choice_corpus.iloc[index]["study_id"])
        best_dist.append(dist[index])
        # best_label.append(np.array(randon_choice_corpus.iloc[index][2:]))

    # print(best_dist)
    return best_examples
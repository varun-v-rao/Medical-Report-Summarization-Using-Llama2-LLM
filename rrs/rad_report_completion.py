# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker

# Import helper functions for radiology report generation
from .rrs_utils import generate_prompt_message, random_sample_selection_v2

# Import dependencies
import time
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from rouge import Rouge
import rouge


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    
    root = 'C:/Users/varu/github/llama-radiology-report-summarization/rrs/'
    
    ########################################################################
    #                             LOAD DATA                                #
    ########################################################################

    # MIMIC-CXR offical training split.
    train_data = pd.read_csv("data/mimic_report_sections_train_v4.csv")
    # CheXpert is used to obtain the train data labels.
    train_label_space = pd.read_csv('data/train_v4_labels.csv')

    # MIMIC-CXR offical testing split. CheXpert is used to obtain the label.
    test_label_space = pd.read_csv('data/baseline_test_V2_Clean_labels.csv')


    test_label_study_ids = []
    for index, row in test_label_space.iterrows():
        test_label_study_ids.append(row["study_id"])

    ########################################################################
    #                     LOAD MODEL AND TOKENIZER                         #
    ########################################################################

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )

    # Pre-defined parameters
    interactive = True
    interactive_times = 17
    rouge_thre = 0.7
    one_near_k_samples = 200
    two_near_k_samples = 14
    rouge_type = "rouge-1"
    rouge= Rouge()

    def process_row_with_random_choice_samples(args):
        row_index, row = args

        test_subject_id, test_study_id = row['subject_id'], row['study_id']
        test_findings, test_impression = row["findings"], row["impression"]

        # Test label vector
        test_sample_label = np.array(test_label_space.iloc[test_label_study_ids.index(test_study_id)][2:])

        # Similar Search. Stage-1
        # Return Top-<one_near_k_samples> samples' id based on label vector.  m is the size of our corpus. randomly sample from all training data.
        near_study_ids = random_sample_selection_v2(test_sample_label, train_label_space, m=10000, k=one_near_k_samples)

        # Get the similar report Findings and Impression based on 'near_study_ids'
        near_samples = []
        for index, train_row in train_data.iterrows():
            if len(near_study_ids) == 0:
                break
            if int(train_row["study_id"]) in near_study_ids:
                near_samples.append({"finding": train_data.iloc[index]["finding"], "impression": train_data.iloc[index]["impression"]})
                near_study_ids.remove(train_row["study_id"])

        # Similar Search. Stage-2
        # Choose Top-<two_near_k_samples> samples from [near_samples] based on Rouge-1 score compared with test findings.
        similar_samples = []
        similar_samples_score = []
        similar_score = []
        for sample in near_samples:
            s_scores = rouge.get_scores([test_findings], [sample["finding"]])
            similar_score.append(s_scores[0][rouge_type]["f"])

        np_similar_score = np.array(similar_score)
        similar_scores_index = np.argsort(-np_similar_score)
        choose_index = similar_scores_index[:two_near_k_samples]

        # Now we have some [similar_samples] filterd by two-stage Similar Search.
        for c_index in choose_index:
            similar_samples.append(near_samples[c_index])

        good_reponse = []
        bad_reponse = []
        former_socre = 0
        former_bad_socre = 0
        all_response_score, all_response = [], []
        try_count = 0
        while True:
            if len(good_reponse) == 0 and len(bad_reponse) == 0:
                prompt_message = generate_prompt_message(test_findings, near_samples=similar_samples)
            
                prompt_tokens = format_tokens(prompt_message, tokenizer)
            
                with torch.no_grad():
                    tokens= torch.tensor(prompt_tokens).long()
                    tokens= tokens.unsqueeze(0)
                    tokens= tokens.to("cuda:0")
                    outputs = model.generate(
                        input_ids=tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        use_cache=use_cache,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        **kwargs
                    )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            else:
                prompt_message = generate_prompt_message(test_findings, near_samples=similar_samples, interactive=interactive, former_good_response=good_reponse, former_bad_response=bad_reponse)

                prompt_tokens = format_tokens(prompt_message, tokenizer)
            
                with torch.no_grad():
                    tokens= torch.tensor(prompt_tokens).long()
                    tokens= tokens.unsqueeze(0)
                    tokens= tokens.to("cuda:0")
                    outputs = model.generate(
                        input_ids=tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        use_cache=use_cache,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        **kwargs
                    )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)


            if reponse == -2:
                print("exceed length, pop 2 similar examples")
                # list_len = len(similar_samples)
                for i in range(2):
                    similar_samples.pop()
                continue

            fotmatted_response = reponse.replace("IMPRESSION:", "")

            compare_scores = []
            for near_sa in similar_samples:
                scores = rouge.get_scores([fotmatted_response], [near_sa["impression"]])
                single_score = scores[0][rouge_type]["f"]
                compare_scores.append(single_score)

            # print(compare_scores)
            score = np.mean(np.array(compare_scores))

            all_response_score.append(score)
            all_response.append(fotmatted_response)
            try_count += 1
            # if it is better than former
            if score >= rouge_thre and score > former_socre:
                former_socre = score
                good_reponse.clear()  # Only keep one good response
                good_reponse.append({"summarize": fotmatted_response, "score": score})
            # if it is worse than former
            if score < rouge_thre and score < former_socre:
                former_bad_socre = score
                # Keep more than one bad response. Length limit is 8 here.
                if len(bad_reponse) > 8:
                    bad_reponse = bad_reponse[-8:]
                # bad_reponse.clear()
                bad_reponse.append({"summarize": fotmatted_response, "score": score})
            if try_count > interactive_times:
                break

        max_score_index = all_response_score.index(max(all_response_score))
        max_score = all_response_score[max_score_index]
        final_best_response = all_response[max_score_index]
        # print("max score=", max_score, "\n", final_best_response)
        # print("max score=", max_score, "\n")
        return row_index, final_best_response

    ################## OLD CODE ##############################
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print(f"User dialogs:\n{dialogs}")
    print("\n==================================\n")


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    
    chats = format_tokens(dialogs, tokenizer)

    with torch.no_grad():
        for idx, chat in enumerate(chats):
            safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_saleforce_content_safety,
                                        )
            # Safety check of the user prompt
            safety_results = [check(dialogs[idx][0]["content"]) for check in safety_checker]
            are_safe = all([r[1] for r in safety_results])
            if are_safe:
                print(f"User prompt deemed safe.")
                print("User prompt:\n", dialogs[idx][0]["content"])
                print("\n==================================\n")
            else:
                print("User prompt deemed unsafe.")
                for method, is_safe, report in safety_results:
                    if not is_safe:
                        print(method)
                        print(report)
                print("Skipping the inferece as the prompt is not safe.")
                sys.exit(1)  # Exit the program with an error status
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            outputs = model.generate(
                input_ids=tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Safety check of the model output
            safety_results = [check(output_text) for check in safety_checker]
            are_safe = all([r[1] for r in safety_results])
            if are_safe:
                print("User input and model output deemed safe.")
                print(f"Model output:\n{output_text}")
                print("\n==================================\n")

            else:
                print("Model output deemed unsafe.")
                for method, is_safe, report in safety_results:
                    if not is_safe:
                        print(method)
                        print(report)



if __name__ == "__main__":
    fire.Fire(main)

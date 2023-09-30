# Import Dependencies

import fire
import torch
import datasets
import numpy as np
from transformers import TrainerCallback
from contextlib import nullcontext
from nltk.tokenize import wordpunct_tokenize
from radgraph import F1RadGraph

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import default_data_collator, Trainer, TrainingArguments

from utils.dataset_utils import preprocess_dataset, preprocess_test_dataset
from utils.config_utils import create_peft_config

def main(
	model_name,
    quantization: bool=True,
    max_new_tokens: int=256, #The maximum numbers of tokens to generate
    min_new_tokens: int=0, #The minimum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    use_cache: bool=False,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    enable_profiler: bool= False,
    mini_test: bool=False,
    output_dir: str="tmp/llama-finetune-output"
):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    ### LOAD MODEL AND TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",    # I've also tried removing this line
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,    # I've also tried removing this line
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    B_FIND, E_FIND = "<FIND>", "</FIND>"
    B_IMPR, E_IMPR = "<IMPR>", "</IMPR>"
    tokenizer.add_tokens([B_FIND, E_FIND, B_IMPR, E_IMPR], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    ### LOAD AND PREPROCESS DATA 
    train_dataset = preprocess_dataset('mimic-cxr',tokenizer,'train')
    print(f'Number of training samples: {len(train_dataset)}')

    #valid_dataset = preprocess_dataset('mimic-cxr',tokenizer,'test')
    #print(f'Number of validation samples: {len(valid_dataset)}')

    if mini_test:
        num_samples = int(0.25*len(train_dataset))
        train_dataset = train_dataset.select(range(num_samples))
        print(f'Number of training samples: {len(train_dataset)}')

    ### PREPARE MODEL FOR PEFT
    model.train()
    model, lora_config = create_peft_config(model) # create peft config

    ### Define train config
    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
        'gradient_accumulation_steps': 2,
        'per_device_train_batch_size': 2,
        'gradient_checkpointing': True,
    }

    ### DEFINE AN OPTIONAL PROFILER
    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    
        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler
            
            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()

    ### FINETUNE MODEL 

    # Define training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        fp16=True,  # Use FP16 if available
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        optim="adamw_torch",
        max_steps=total_steps if enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
            callbacks=[profiler_callback] if enable_profiler else [],
        )

        # Start training
        trainer.train()

    ### SAVE MODEL
    model.save_pretrained(output_dir)
    #tokenizer.save_pretrained(output_dir)

    ### EVALUATE TEST SET
    def generate_summary(sample):
        model_input = tokenizer(sample['text'], return_tensors="pt").to("cuda")
        model.eval()
        with torch.no_grad():
            response = tokenizer.decode(model.generate(**model_input, max_new_tokens=max_new_tokens)[0], skip_special_tokens=True)
        formatted_input_text = sample['text'].replace(B_FIND,"").replace(f'{E_FIND}{B_IMPR}',"")
        formatted_response = response.replace(formatted_input_text,"").strip()
        return {
            "text": formatted_input_text,
            "summary":sample["summary"],
            "pred": formatted_response
        }

    test_dataset = preprocess_test_dataset('mimic-cxr',tokenizer,'test')
    
    if mini_test:
        num_samples = int(0.25*len(test_dataset))
        test_dataset = test_dataset.select(range(num_samples))
        print(f'Number of test samples: {len(test_dataset)}')

    
    results = test_dataset.map(generate_summary, remove_columns=list(test_dataset.features))

    def process_impression(impression):
        impression = impression.lower()
        return ' '.join(wordpunct_tokenize(impression))

    pred_str = results["pred"]
    pred_str = list(map(process_impression,pred_str))
    label_str = results["summary"]

    ###################################
    rouge = datasets.load_metric("rouge")
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    res = {key: value.mid.fmeasure * 100 for key, value in rouge_output.items()}
    print('ROUGE:')
    print({k: round(v, 4) for k, v in res.items()})

    ##################################
    bertscore = datasets.load_metric("bertscore")
    bertscore_output = bertscore.compute(predictions=pred_str, references=label_str, lang='en')
    res = {key: np.asarray(value).mean()*100 for key, value in bertscore_output.items() if key != 'hashcode'}
    print('BertScore:')
    print({k: round(v,4) for k, v in res.items()})

    #################################
    f1radgraph = F1RadGraph(reward_level="partial")
    score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=pred_str,
                                                                                refs=label_str)
    print("F1RadGraph:")
    print(score*100)
    
    # Print Samples

if __name__ == "__main__":
    fire.Fire(main)
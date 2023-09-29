from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset
from pathlib import Path
import datasets

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result

#dataset_config = 'mimic-cxr','mimic-iii'  
#split = 'train','validate',test
B_FIND, E_FIND = "<FIND>", "</FIND>"
B_IMPR, E_IMPR = "<IMPR>", "</IMPR>"
def preprocess_dataset(dataset_config, tokenizer, split):
    data_path = '/nfs/turbo/umms-vgvinodv/data/bioNLP23-Task-1B/data/'
    findings_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.findings.tok')
    impression_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.impression.tok')


    findings = [line.strip() for line in open(findings_file_path).readlines()]
    impression = [line.strip() for line in open(impression_file_path).readlines()]

    dataset = datasets.Dataset.from_dict({"text":findings,"summary":impression}) 

    prompt = (
        f"{B_FIND}{{text}}{E_FIND}{B_IMPR}{{summary}}{E_IMPR}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                text=sample["text"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset

def preprocess_test_dataset(dataset_config, tokenizer, split):
    data_path = '/nfs/turbo/umms-vgvinodv/data/bioNLP23-Task-1B/data/'
    findings_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.findings.tok')
    impression_file_path = Path(data_path).joinpath(dataset_config).joinpath(split+'.impression.tok')


    findings = [line.strip() for line in open(findings_file_path).readlines()]
    impression = [line.strip() for line in open(impression_file_path).readlines()]

    dataset = datasets.Dataset.from_dict({"text":findings,"summary":impression}) 

    prompt = (
        f"{B_FIND}{{text}}{E_FIND}{B_IMPR}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                text=sample["text"],
            ),
            "summary":sample["summary"],
            "pred":""
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    return dataset

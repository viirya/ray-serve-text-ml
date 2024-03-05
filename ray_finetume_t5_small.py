
import os

from pprint import pprint
import ray

# KubeRay controls cluster resources using YAML files. We can't set the resources here.
ray.init()
pprint(ray.cluster_resources())

from ray.train import ScalingConfig, RunConfig
from ray import train
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer

from datasets import load_dataset, load_from_disk
import nltk
# Isolated K8s cluster doesn't have internet access. nltk.download('punkt') will fail.
# We install it in docker image.
# nltk.download('punkt')
from transformers import AutoTokenizer
import evaluate
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.models.t5 import T5Tokenizer

from pyarrow.fs import S3FileSystem

import ray.data

# The base path is the path to the directory where the dataset and the model are stored.
base_path = "/text_ml"
s3_bucket = "ray-example-data"

# Ray cluster launched by KubeRay doesn't respect resource rayStartParams field in yaml file
# https://github.com/ray-project/ray/issues/43261
# So 2+ `cpus_per_worker` will cause issue in Ray.
cpus_per_worker = int(os.environ.get("CPUS_PER_WORKER", "1"))
gpus_per_worker = int(os.environ.get("GPUS_PER_WORKER", "0"))
batch_size = int(os.environ.get("BATCH_SIZE", "8"))
# SPREAD or PACK or STRICT_PACK or STRICT_SPREAD
placement = os.environ.get("PLACEMENT", "SPREAD")

# Setting num_workers to 6+ (the cpu limit in yaml file) will cause issue in Ray. See
# https://github.com/ray-project/ray/issues/43265
num_workers = int(os.environ.get("NUM_WORKERS", "5"))
use_gpu = os.environ.get("USE_GPU", "False").lower() == "true"
epochs = int(os.environ.get("EPOCHS", "10"))

sampled_num = os.environ.get("SAMPLED_NUM")
find_unused_parameters = os.environ.get("FIND_UNUSED_PARAMETERS", "False").lower() == "true"

def train_func(config):
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 10)
    steps_per_epoch = config.get("steps_per_epoch", 100)

    # Use absolute path to load the model and tokenizer as Ray will run the training script in a different directory.
    # tokenizer = AutoTokenizer.from_pretrained(f"{base_path}/t5-small", use_fast=False)
    # Seems fast tokenizer causes deadlock in Ray. So we use the slow Python tokenizer.
    tokenizer = T5Tokenizer.from_pretrained(f"{base_path}/t5-small-tokenizer")

    model_name = "t5-finetune-test"
    model_dir = f"{base_path}/{model_name}"

    # Ray doesn't support MPS backend of PyTorch, so we need to set `use_cpu` to True if running on local Macbook:
    # https://github.com/ray-project/ray/issues/28321
    # Alternatively, we can choose tensorflow-metal which RLlib supports. But now we are using PyTorch.
    # Otherwise, we will get the following error:
    # RuntimeError: ProcessGroupGloo::allgather: invalid tensor type at index 0 (expected TensorOptions(dtype=long long, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)),
    # got TensorOptions(dtype=long long, device=mps:0, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)))
    args = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=use_gpu, # if using cpu, set it to False
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        use_cpu=not use_gpu,
        max_steps=steps_per_epoch * epochs,
        ddp_find_unused_parameters=find_unused_parameters
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    import numpy as np

    # metric = evaluate.load("rouge")
    # For k8s which cannot access internet, we need to use local rouge.
    # Just checkout evaluate github repo (https://github.com/huggingface/evaluate) and
    # copy to docker image.
    metric = evaluate.load(f"{base_path}/evaluate/metrics/rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                         for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                          for label in decoded_labels]
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)
        print(result)
        # Extract ROUGE f1 scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                           for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(f"{base_path}/t5-small")

    train_ds = train.get_dataset_shard("train")
    eval_ds = train.get_dataset_shard("validation")

    train_ds_iterable = train_ds.iter_torch_batches(batch_size=batch_size)
    eval_ds_iterable = eval_ds.iter_torch_batches(batch_size=batch_size)

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

# load dataset https://huggingface.co/datasets/b-mc2/sql-create-context?row=3
# dataset = load_dataset("b-mc2/sql-create-context")
dataset = load_from_disk(f"{base_path}/sql-create-context")

datasets_train_test = dataset['train'].train_test_split(test_size=3000)
datasets_train_validation = dataset['train'].train_test_split(test_size=3000)

dataset['train'] = datasets_train_validation["train"]
dataset['validation'] = datasets_train_validation["test"]
dataset["test"] = datasets_train_test["test"]

# Use absolute path to load the model and tokenizer as Ray will run the training script in a different directory.
# tokenizer = AutoTokenizer.from_pretrained(f"{base_path}/t5-small", use_fast=False)
# Seems fast tokenizer causes deadlock in Ray. So we use the slow Python tokenizer.
tokenizer = T5Tokenizer.from_pretrained(f"{base_path}/t5-small-tokenizer")
prefix_question = "question: "
prefix_context = "context: "

max_input_length = 512
max_target_length = 200

def preprocess_data(examples):
    inputs = [prefix_question + q + "\n" + prefix_context + c for q, c in
              zip(examples['question'], examples['context'])]
    # Note: Padding dataset to the same length across all batches is required for Ray to work.
    # Otherwise, because Ray will convert dataset to panda format, later we will get the following error:
    # RuntimeError: Numpy array of object dtype cannot be converted to a Torch Tensor. This may because the numpy array is a ragged tensor--it contains items of different sizes.
    # For training locally without Ray, it may not be necessary to pad the dataset.
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_input_length, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['answer'], truncation=True, max_length=max_target_length, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Remove original string columns from the dataset.
# Otherwise, because Ray will convert dataset to panda format, later we will get the following error:
# can't convert np.ndarray of type numpy.str_. the only supported types are:
# float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool
tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=["question", "context", "answer"])

# Select a subset of the dataset for quick testing
if sampled_num:
    tokenized_datasets["train"] = tokenized_datasets["train"].shuffle().select(range(int(sampled_num)))
    # tokenized_datasets["validation"] = tokenized_datasets["validation"].shuffle().select(range(int(sampled_num)))
    # tokenized_datasets["test"] = tokenized_datasets["test"].shuffle().select(range(int(sampled_num)))

# Distributed preprocessing and data ingestion using Ray Data
ray_datasets = {
    "train": ray.data.from_huggingface(tokenized_datasets["train"]),
    "validation": ray.data.from_huggingface(tokenized_datasets["validation"])
}

def map_batches(batch):
    return batch
processed_datasets = {
    key: ds.map_batches(map_batches, batch_format="pandas").random_shuffle(seed=42).repartition(4)
    for key, ds in ray_datasets.items()
}

# Setup training checkpointing to S3
s3fs = S3FileSystem()

train_ds_size = processed_datasets["train"].count()
steps_per_epoch = train_ds_size // (batch_size * num_workers)

trainer = TorchTrainer(
    train_func,
    train_loop_config={
        "epochs": epochs,
        "batch_size": batch_size,  # per device
        # Need to provide steps_per_epoch to avoid the following error:
        # https://github.com/huggingface/datasets/issues/5773
        # https://discuss.huggingface.co/t/streaming-dataset-into-trainer-does-not-implement-len-max-steps-has-to-be-specified/32893
        "steps_per_epoch": steps_per_epoch
    },
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu, resources_per_worker={"GPU": gpus_per_worker, "CPU": cpus_per_worker}, placement_strategy=placement),
    run_config=RunConfig(storage_filesystem=s3fs, storage_path=f"{s3_bucket}/ml_ray_results", verbose=2),
    datasets=processed_datasets
)

trainer.fit()

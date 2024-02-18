
from pprint import pprint
import ray

ray.init()
pprint(ray.cluster_resources())

from ray.train import ScalingConfig
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer

from datasets import load_dataset, load_from_disk
import nltk
# nltk.download('punkt')
from transformers import AutoTokenizer
import evaluate
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# The base path is the path to the directory where the dataset and the model are stored.
base_path = "/text_ml"
def train_func(config):
    # load dataset https://huggingface.co/datasets/b-mc2/sql-create-context?row=3

    # dataset = load_dataset("b-mc2/sql-create-context")
    dataset = load_from_disk(f"{base_path}/sql-create-context")

    datasets_train_test = dataset['train'].train_test_split(test_size=3000)
    datasets_train_validation = dataset['train'].train_test_split(test_size=3000)

    dataset['train'] = datasets_train_validation["train"]
    dataset['validation'] = datasets_train_validation["test"]
    dataset["test"] = datasets_train_test["test"]

    # Use absolute path to load the model and tokenizer as Ray will run the training script in a different directory.
    tokenizer = AutoTokenizer.from_pretrained(f"{base_path}/t5-small")
    prefix_question = "question: "
    prefix_context = "context: "

    def preprocess_data(examples):
        inputs = [prefix_question + q + "\n" + prefix_context + c for q, c in
                  zip(examples['question'], examples['context'])]
        model_inputs = tokenizer(inputs, truncation=True, padding=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['answer'], truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_data, batched=True)

    batch_size = 8
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
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True, # if using cpu, set it to False
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        use_cpu=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    import numpy as np

    # metric = evaluate.load("rouge")
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

    # Select a subset of the dataset for quick testing
    # tokenized_datasets["train"] = tokenized_datasets["train"].shuffle().select(range(100))
    # tokenized_datasets["validation"] = tokenized_datasets["train"].shuffle().select(range(100))
    # tokenized_datasets["test"] = tokenized_datasets["test"].shuffle().select(range(100))

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()


trainer = TorchTrainer(
    train_func, scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
)

trainer.fit()

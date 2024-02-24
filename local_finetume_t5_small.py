from datasets import load_dataset

# load dataset https://huggingface.co/datasets/b-mc2/sql-create-context?row=3

dataset = load_dataset("b-mc2/sql-create-context")

datasets_train_test = dataset['train'].train_test_split(test_size=3000)
datasets_train_validation = dataset['train'].train_test_split(test_size=3000)

dataset['train'] = datasets_train_validation["train"]
dataset['validation'] = datasets_train_validation["test"]
dataset["test"] = datasets_train_test["test"]

# preprocessing dataset

import nltk
nltk.download('punkt')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

prefix_question = "question: "
prefix_context = "context: "

# for each example {'question': '..', 'context': '..', 'answer': '..'},
# add prefix text and concatenate 'question' and 'context' string as training data.
# 'answer' is taken as label of training data.

def preprocess_data(examples):
  inputs = [prefix_question + q + "\n" + prefix_context + c for q, c in zip(examples['question'], examples['context'])]
  model_inputs = tokenizer(inputs, truncation=True, padding=True)
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples['answer'], truncation=True, padding=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)

# preparing finetuning training

import evaluate

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

batch_size = 8
model_name = "t5-finetune-test"
model_dir = f"./{model_name}"
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
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    use_cpu=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

import numpy as np

metric = evaluate.load("rouge")

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


# load pre-trained model as initial model
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# for quickly checking training process, randomly choose partial data
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle().select(range(100))
tokenized_datasets["validation"] = tokenized_datasets["train"].shuffle().select(range(100))
tokenized_datasets["test"] = tokenized_datasets["test"].shuffle().select(range(100))

# starting finetuning
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


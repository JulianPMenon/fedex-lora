import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils import *
from models import *
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
import numpy as np
import wandb
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import torch
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from data_utils import *
import os
from copy import deepcopy


def train_client(model, dataloader, args, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.local_epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    model.train()
    for epoch in range(args.local_epochs):

        for step, data in enumerate(tqdm(dataloader)):
            data = {k: v.to(device) for k, v in data.items()}

            with autocast(enabled=use_amp):
                outputs = model(**data)
                loss = outputs.loss

            #wandb.log({"client_loss": loss.detach().cpu().numpy()})

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return model.state_dict()


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task == "mrpc":
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    elif task == "qqp":
        return accuracy_score(all_true_labels, all_predictions), f1_score(
            all_true_labels, all_predictions
        )
    elif task in ["mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_global_model(
    global_model, dataloader, args, max_metric1, max_metric2, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    global_model.to(device)

    global_model.eval()
    eval_loss = 0
    loss_steps = 0
    has_labels_for_metrics = True
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels")
        has_invalid_labels = labels is not None and (labels < 0).any().item()
        if has_invalid_labels:
            has_labels_for_metrics = False
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        else:
            model_inputs = batch
        with torch.no_grad():

            outputs = global_model(**model_inputs)

            if labels is not None and not has_invalid_labels and outputs.loss is not None:
                eval_loss += outputs.loss.detach().cpu().numpy()
                loss_steps += 1

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            if labels is not None and not has_invalid_labels:
                all_true_labels.extend(labels.cpu().numpy())

    if loss_steps > 0:
        eval_loss /= loss_steps

    if not has_labels_for_metrics or len(all_true_labels) == 0:
        print(
            f"{args.task} - Eval Loss: {eval_loss:.4f} (no labels for metrics in this split)"
        )
        return max_metric1, max_metric2

    # Calculate the metrics for the specific task
    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    # wandb.log(
    #     {
    #         f"eval_loss": eval_loss,
    #         f"metric1": metric1,
    #         f"metric2": metric2 if metric2 is not None else 0,
    #         f"max_metric1": max_metric1,
    #         f"max_metric2": max_metric2 if max_metric2 is not None else 0,
    #     }
    # )

    return max_metric1, max_metric2


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_client_e2e(model, train_dataset, val_dataset, tokenizer, args):
    num_epochs = args.local_epochs  # or whatever number of epochs you want
    per_device_train_batch_size = args.batch_size
    num_training_steps = len(train_dataset) * num_epochs // per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup

    optimizer = torch.optim.AdamW(model.parameters())

    # Define training arguments
    training_args = TrainingArguments(
        # Directory to save the model
        output_dir="./models_trained/gpt4/dump/models/gpt2-e2e-lora_gpt4",
        overwrite_output_dir=True,
        logging_dir="./models_trained/gpt4/dump/logs/gpt2-e2e-lora_gpt4",  # Directory for logs
        per_device_train_batch_size=args.batch_size,  # Adjust based on your GPU capacity
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",
        num_train_epochs=num_epochs,  # Number of training epochs
        learning_rate=args.lr,  # Learning rate for LoRA parameters
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        #report_to="wandb",
        run_name="fed-lora",
        logging_steps=100,  # Log every 100 steps
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(
            optimizer,
            get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps),
        ),
    )

    # Train the model
    trainer.train()
    return model.state_dict()


def gen_and_save(model, dataloader, tokenizer, args):
    device = args.device
    model.to(device)
    model.eval()

    all_predictions = []

    all_inputs = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):

            inputs = {k: v.to(device) for k, v in batch.items()}

            # Generate predictions (starting from after the MR)
            generated = model.generate(
                input_ids=inputs["input_ids"],  # Input MR as prompt
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1]
                + 50,  # Allow space for generation after MR
                num_return_sequences=1,
                no_repeat_ngram_size=4,
                do_sample=True,
                num_beams=10,
                penalty_alpha=0.9,
                pad_token_id=tokenizer.eos_token_id,  # Ensure padding works correctly
            )
            # Decode the generated predictions, excluding the input MR tokens
            # We slice the generated tokens to remove the input MR part

            input_seq = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            predictions = [
                tokenizer.decode(
                    generated[i][len(inputs["input_ids"][i]) :],
                    skip_special_tokens=True,
                )
                for i in range(generated.shape[0])
            ]
            # Collect predictions and references
            all_inputs.extend(input_seq)
            all_predictions.extend(predictions)
            # all_references.extend(references)

    return all_predictions, all_inputs


def process_lists(input_list, second_list, third_list):
    result1 = []
    result2 = []
    result3 = []
    current_group = []
    current_item = None
    second_list_index = 0

    for item in input_list:
        if item != current_item:
            if current_group:
                result1.append(current_group)
                result2.append(current_item)
                result3.append(third_list[second_list_index - 1])
            current_item = item
            current_group = [second_list[second_list_index]]
            second_list_index += 1
        else:
            if second_list_index < len(second_list):
                current_group.append(second_list[second_list_index])
                second_list_index += 1

    if current_group:
        result1.append(current_group)

    return result1, result2, result3


def evaluate_e2e_save_text(model, test_data, tokenizer, args):

    def preprocess_function2(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Combine the input-output pair into a single text
        model_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        # Tokenize the combined inputs
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels are the same as input_ids but shift them for next-token prediction
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Set the labels to -100 where attention mask is 0 (this will ignore padding in loss computation)
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100

        return tokenized_inputs

    tokenized_test_dataset = test_data.map(preprocess_function2, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(
        ["meaning_representation", "human_reference"]
    )
    tokenized_test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataloader = create_dataloader(tokenized_test_dataset, args)
    all_predictions, all_inputs = gen_and_save(model, test_dataloader, tokenizer, args)
    all_references = test_data[0 : len(all_predictions)]["human_reference"]

    all_references_new, all_inputs_new, all_predictions_new = process_lists(
        all_inputs, all_references, all_predictions
    )

    path_pred = args.run_dir + "/predictions.txt"
    path_ref = args.run_dir + "/refs_exact.txt"

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    with open(path_pred, "w") as file:
        for item in all_predictions_new:
            file.write(item.strip() + "\n")

    with open(path_ref, "w") as file:
        for str_list in all_references_new:
            for item in str_list:
                file.write(item.strip() + "\n")

            file.write("\n")

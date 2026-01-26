import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
    VeraConfig,
)
from data_utils import *
import argparse
from copy import deepcopy


def create_peft_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    vera = getattr(args, 'vera', False)
    if vera:
        peft_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=getattr(args, 'r', 256),
            target_modules=["query", "value"],
            projection_prng_key=getattr(args, 'projection_prng_key', 0),
            save_projection=getattr(args, 'save_projection', True),
            vera_dropout=getattr(args, 'vera_dropout', 0.0),
            d_initial=getattr(args, 'd_initial', 0.1),
            fan_in_fan_out=getattr(args, 'fan_in_fan_out', False),
            bias=getattr(args, 'bias', 'none'),
            modules_to_save=getattr(args, 'modules_to_save', None),
            init_weights=getattr(args, 'init_weights', True),
            layers_to_transform=getattr(args, 'layers_to_transform', None),
            layers_pattern=getattr(args, 'layers_pattern', None),
            inference_mode=False,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.rslora,
            target_modules=["query", "value"],
        )

    model = get_peft_model(model, peft_config)
    return model


def create_peft_FFA_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    vera = getattr(args, 'vera', False)
    if vera:
        peft_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=getattr(args, 'r', 256),
            target_modules=["query", "value"],
            projection_prng_key=getattr(args, 'projection_prng_key', 0),
            save_projection=getattr(args, 'save_projection', True),
            vera_dropout=getattr(args, 'vera_dropout', 0.0),
            d_initial=getattr(args, 'd_initial', 0.1),
            fan_in_fan_out=getattr(args, 'fan_in_fan_out', False),
            bias=getattr(args, 'bias', 'none'),
            modules_to_save=getattr(args, 'modules_to_save', None),
            init_weights=getattr(args, 'init_weights', True),
            layers_to_transform=getattr(args, 'layers_to_transform', None),
            layers_pattern=getattr(args, 'layers_pattern', None),
            inference_mode=False,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.rslora,
            target_modules=["query", "value"],
        )
    model = get_peft_model(model, peft_config)

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def create_peft_gpt2_model_e2e(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model


def create_peft_gpt2_model_e2e_ffa(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model

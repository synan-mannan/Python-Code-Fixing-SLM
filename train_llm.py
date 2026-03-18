import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os

def load_config():
    with open('training_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Model & Tokenizer (Mistral)
    model_name = config['model']['llm_name']
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['bitsandbytes']['load_in_4bit'],
        bnb_4bit_quant_type=config['bitsandbytes']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config['bitsandbytes']['bnb_4bit_use_double_quant'],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    
    # LoRA
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    
    # Dataset
    dataset = load_dataset('json', data_files=config['dataset']['train'], split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    
    def formatting_prompts_func(example):
        # Mistral chat template
        text = f"<s>[INST] {example['prompt']} [/INST]"
        return {'text': text}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # Training args (smaller batch for larger model)
    training_args = TrainingArguments(
        output_dir=config['output_dir']['llm'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=2,  # Smaller for 7B
        gradient_accumulation_steps=8,
        learning_rate=config['training']['learning_rate'],
        fp16=True,
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        warmup_steps=config['training']['warmup_steps'],
        report_to='wandb',
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['training']['max_seq_length'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config['output_dir']['llm'])

if __name__ == '__main__':
    main()


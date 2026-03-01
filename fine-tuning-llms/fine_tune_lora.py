"""
Fine-tune Llama-3.1-8B with QLoRA (4-bit quantization + LoRA adapters).

Blog post: https://dadops.dev/blog/fine-tuning-llms/
Code Block 5.

REQUIRES: GPU with sufficient VRAM, transformers, peft, trl, bitsandbytes

Blog claims:
  - Trainable params: 13.6M / all params: 8.03B / trainable%: 0.17%
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Load base model with 4-bit quantization (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,         # QLoRA: 4-bit quantization
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# LoRA configuration — these are the key hyperparameters
lora_config = LoraConfig(
    r=16,                      # rank: 16 is a good default
    lora_alpha=16,             # scaling factor: usually equal to rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13.6M || all params: 8.03B || trainable%: 0.17%

# Training config
training_config = SFTConfig(
    output_dir="./ticket-classifier-lora",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    learning_rate=2e-4,             # the most critical hyperparameter
    warmup_steps=10,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
# Note: train_dataset must be prepared from your JSONL data
# trainer = SFTTrainer(
#     model=model,
#     args=training_config,
#     train_dataset=train_dataset,     # your formatted dataset
#     tokenizer=tokenizer,
# )
# trainer.train()
print("LoRA config loaded. Provide train_dataset to begin training.")

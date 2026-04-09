import os

replacements = [
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\text_trainer.py',
        'config = AutoConfig.from_pretrained(model_name)',
        'config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\model_utility.py',
        'config = AutoConfig.from_pretrained(model_path)',
        'config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\monkeypatch.py',
        'model_config = transformers.AutoConfig.from_pretrained(pretrained_model)',
        'model_config = transformers.AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\add_random_noise.py',
        'model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")',
        'model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\add_random_noise.py',
        'tokenizer = AutoTokenizer.from_pretrained(model_path)',
        'tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_grpo.py',
        'tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])',
        'tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"], trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_grpo.py',
        'model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)',
        'model = model_class.from_pretrained(train_request["model_path"], trust_remote_code=True, **model_kwargs)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_dpo.py',
        'tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])',
        'tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"], trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_dpo.py',
        'model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)',
        'model = model_class.from_pretrained(train_request["model_path"], trust_remote_code=True, **model_kwargs)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_instruct.py',
        'tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])',
        'tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"], trust_remote_code=True)'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_instruct.py',
        'model = model_class.from_pretrained(\n        model_path,\n        # trust_remote_code=True, remove this because we already filter the model architecture, it will not be used with liger-kernel \n        torch_dtype=torch.bfloat16,\n        attn_implementation=attn_implementation,\n    )',
        'model = model_class.from_pretrained(\n        model_path,\n        trust_remote_code=True,\n        torch_dtype=torch.bfloat16,\n        attn_implementation=attn_implementation,\n    )'
    ),
    (
        r'c:\Users\ASUS\Documents\commit\master-text\scripts\train_instruct.py',
        'model = model_class.from_pretrained(\n        model_path,\n        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",\n        torch_dtype=torch.bfloat16,',
        'model = model_class.from_pretrained(\n        model_path,\n        trust_remote_code=True,\n        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",\n        torch_dtype=torch.bfloat16,'
    )
]

for fpath, old_str, new_str in replacements:
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        
        if old_str in content:
            content = content.replace(old_str, new_str)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Patched: {os.path.basename(fpath)}")
print("Patching complete!")

# Training scripts with [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory)

Usage:

1. follow installation of [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory)

2. place e.g. `train-dpo.sh` in your git-cloned 360-LLaMA-Factory's root directory (same hierarchy as [360-example.sh](https://github.com/Qihoo360/360-LLaMA-Factory/blob/sp/360-example.sh))

3. register your dataset (e.g. [Light-R1-DPO](https://huggingface.co/datasets/qihoo360/Light-R1-DPO)) in [dataset_info.json](https://github.com/Qihoo360/360-LLaMA-Factory/blob/sp/data/dataset_info.json)
```json
  "light-r1-dpo": {
    "file_name": "/path/to/dpo-pairs.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
```

4. fill in the missing arguments in `train-dpo.sh` and `sh train-dpo.sh`

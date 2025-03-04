# Light-R1: Surpassing R1-Distill from Scratch\* with 70k Math Data through Curriculum SFT & DPO

*\*from models without long COT*


| Model | AIME24 (pass@1) | AIME25 |
| --- | --- | --- |
| DeepSeek-R1-Distill-Llama-70B | 70.0 |  | 
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 54.9 |
| TinyR1-32B [(Math-Model, from R1-Distill)](https://huggingface.co/qihoo360/TinyR1-32B-Preview#evaluation) | 73.1 | N/A |
| LIMO (32B) | 56.3 |  |
| s1.1-32B | 64.7 |  |
| OpenThinker-32B | 66.0 |  |
| [**Light-R1-32B (ours)** 🤗](TODO) | **76.6** | **64.6** | 

While much work has been open-sourced trying to reproduce DeepSeek-R1 on models of 72B or less, none achieves similar performance on the hard math competition AIME24 as DeepSeek-R1-Distill-Qwen-32B's score 72.6.

We introduce Light-R1-32B, which achieves 76.6 on AIME24 training from Qwen2.5-32B-Instruct. Starting from models without long COT (*from scratch* in terms of R1) and training on decontaminated math data, we distilled DeepSeek-R1 with curriculum SFT & DPO to surpass DeepSeek-R1-Distill-Qwen-32B on AIME24, and improved further with model merging.

More importantly, 
besides the state-of-the-art from-scratch model Light-R1-32B, we also released on Day 1 all training datasets of our curriculum SFT & DPO and training code based on [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory).

We believe Light-R1 represents a practical way of training strong long COT models from scratch (from models without long COT). While we are working to further improve our models with RL, curriculum SFT & DPO facilitates more control along the pipeline and is more cost-friendly.

With the rapid development of training and inference techniques, we hope to see more accessible long-COT models in the near future and Light-R1 provides a validated transparent way to train them in at least specialized domains.


## Release Details

- Light-R1-32B model on [🤗 huggingface](TODO)

- Curriculum SFT & DPO datasets on [🤗 huggingface]()

- Training scripts based on [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) in [scripts]()

- Evaluation code based on [DeepScaleR](https://github.com/agentica-project/deepscaler) in [deepscaler-release]()
    - along with evaluation logs of our checkpoints
    - all our reported scores are averaged over 64 runs; public models' scores are taken from their evaluation results and if not present, averaged over 64 runs

- Technical report work in progress

## Inference Notes

Light-R1-32B does not always think as its thinking capabilities are trained only with math data.

We forced Light-R1 to think by hard-coding `<think>` in the chat template right before the model is supposed to generate output, as suggested by [DeepSeek](https://x.com/deepseek_ai/status/1890324295181824107).

[vLLM](https://github.com/vllm-project/vllm) or [SGLang] are suggested for inference.
Light-R1-32B inherits Qwen models' chat template with `<think>` and `</think>` added as special tokens and `<think>` hard-coded to force thinking.


## Post-Training through Curriculum SFT & DPO

|  | AIME24 pass@1 (64 average) | AIME25 | GPQA Diamond |
| --- | --- | --- | --- |
| Qwen2.5-32B-Instruct |  |  |  |
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 54.9 | 62.1 |
| Light-R1-SFT-stage1 | 0.701（16@1） | 0.567(16@1) |  |
| Light-R1-SFT-stage2 | 73.0 | 64.3 | 60.6 |
| Light-R1-DPO | 75.8 | 63.4 | 61.8 |
| Light-R1-32B | 76.6 | 64.6 | 61.8 | 

We adopted a curriculum learning approach with SFT and DPO.

Training questions are collected from public math datasets including [open-r1/OpenR1-Math-220k](open-r1/OpenR1-Math-220k), [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and AIME (up to 2023).
We decontaminated the questions against common Reasoning benchmarks such as AIME24/25, MATH-500 and GPQA Diamond.

We collected responses from DeepSeek-R1 on these questions and filtered them based on verification and difficulty levels rated by sampling DeepSeek-R1-Distill-Qwen-1.5B, forming a 70k dataset fot SFT stage1.

After SFT stage1, a more difficult set, mostly filtered from the 70k dataset, was constructed with 3k data for SFT stage2.

Then we sampled Light-R1-SFT-stage2's responses after SFT stage2, filtered correct and incorrect ones for each question and construct DPO pairs based on verification results and DeepSeek-R1's responses.

DPO is performed on top of SFT stage2 with sequence parallelism in [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory).

Finally, we merged models of SFT-stage2, DPO and another DPO version with AIME24 score 74.7.
The two DPO versions differ in that one of the data has special tokens skipped in rejected responses. Interestingly the resulting version also exhibit improvement.

We observed stepwise improvement in our approach and intermediate evaluation results of each stage are listed in the table above.


## Data Decontamination

We carefully evaluated data contamination of several open-sourced datasets.

While certain contamination may be [inevitable during pre-training](https://x.com/DimitrisPapail/status/1888325914603516214),
it is unacceptable for post-training to compare on benchmarks.
MATH-500 is somewhat compromised with tens of questions that are identical or only numbers changed. AIME 24 and 25 stay intact but we have to pay special attention when we incorporate AIME data up to 2023.


## License & Acknowledgements

All released materials of this project follows the open-source license Apache 2.0.

Our training experiments are powered by [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory).
Our evaluation scripts are based on [DeepScaleR](https://github.com/agentica-project/deepscaler) and therefore [verl](https://github.com/volcengine/veRL).

Light-R1-32B is trained from [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct).
Training data are collected from various public sources.


## Citation

```bibtex
@misc{lightr1proj,
      title={Light-R1: Surpassing R1-Distill from Scratch with 70k Data through Curriculum SFT & DPO}, 
      author={Liang Wen, Fenrui Xiao, Xin He,  Yunke Cai， Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, Xiangzheng Zhang},
      year={2025},
      eprint={},
      archivePrefix={},
      url={https://github.com/Qihoo360/Light-R1}, 
}
```
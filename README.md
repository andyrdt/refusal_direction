# Refusal in Language Models Is Mediated by a Single Direction

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains code and results accompanying the paper "Refusal in Language Models Is Mediated by a Single Direction".
In the spirit of scientific reproducibility, we provide code to reproduce the main results from the paper.

- [Paper](https://arxiv.org/abs/2406.11717)
- [Blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## Setup

```bash
git clone https://github.com/andyrdt/refusal_direction.git
cd refusal_direction
source setup.sh
```

The setup script will prompt you for a HuggingFace token (required to access gated models) and a Together AI token (required to access the Together AI API, which is used for evaluating jailbreak safety scores).
It will then set up a virtual environment and install the required packages.

## Qwen3-14B HarmBench head-ablation compliance experiment

To compare a local Qwen3-14B model with and without the selected 60 attention heads ablated, install the project dependencies, set a DeepSeek API key, then run:

```bash
export DEEPSEEK_API_KEY="..."
python3 run_harmbench_head_ablation_compliance.py
```

The command defaults to the locally cached Qwen3-14B snapshot at `/root/autodl-tmp/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`; override it with `--model_path /path/to/Qwen3-14B` if needed. It evaluates the first 100 examples in `walledai/HarmBench` (`standard`, `train`) using Qwen3's thinking chat template. Qwen3 generation uses batches of 4 and the DeepSeek judge sends up to 10 simultaneous requests; override these with `--batch_size` and `--judge_concurrency`. It writes timestamped `baseline.jsonl`, `ablated.jsonl`, `summary.json`, `config.json`, and `run.log` files under `results/harmbench_head_ablation_compliance/`. The DeepSeek judge is `deepseek-v4-flash` in non-thinking mode.

`walledai/HarmBench` is gated on Hugging Face: accept its access conditions and authenticate with Hugging Face before the first run. The script only accepts an already-downloaded local model directory and does not download model weights.

## Reproducing main results

To reproduce the main results from the paper, run the following command:

```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```
where `{model_path}` is the path to a HuggingFace model. For example, for Llama-3 8B Instruct, the model path would be `meta-llama/Meta-Llama-3-8B-Instruct`.

The pipeline performs the following steps:
1. Extract candiate refusal directions
    - Artifacts will be saved in `pipeline/runs/{model_alias}/generate_directions`
2. Select the most effective refusal direction
    - Artifacts will be saved in `pipeline/runs/{model_alias}/select_direction`
    - The selected refusal direction will be saved as `pipeline/runs/{model_alias}/direction.pt`
3. Generate completions over harmful prompts, and evaluate refusal metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/completions`
4. Generate completions over harmless prompts, and evaluate refusal metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/completions`
5. Evaluate CE loss metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/loss_evals`

For convenience, we have included pipeline artifacts for the smallest model in each model family:
- [`qwen/qwen-1_8b-chat`](/pipeline/runs/qwen-1_8b-chat/)
- [`google/gemma-2b-it`](/pipeline/runs/gemma-2b-it/)
- [`01-ai/yi-6b-chat`](/pipeline/runs/yi-6b-chat/)
- [`meta-llama/llama-2-7b-chat-hf`](/pipeline/runs/llama-2-7b-chat-hf/)
- [`meta-llama/meta-llama-3-8b-instruct`](/pipeline/runs/meta-llama-3-8b-instruct/)

## Minimal demo Colab

As part of our [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), we included a minimal demo of bypassing refusal. This demo is available as a [Colab notebook](https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw).

## As featured in

Since publishing our initial [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) in April 2024, our methodology has been independently reproduced and used many times. In particular, we acknowledge [Fail](https://huggingface.co/failspy)[Spy](https://x.com/failspy) for their work in reproducing and extending our methodology.

Our work has been featured in:
- [HackerNews](https://news.ycombinator.com/item?id=40242939)
- [Last Week in AI podcast](https://open.spotify.com/episode/2E3Fc50GVfPpBvJUmEwlOU)
- [Llama 3 hackathon](https://x.com/AlexReibman/status/1789895080754491686)
- [Applying refusal-vector ablation to a Llama 3 70B agent](https://www.lesswrong.com/posts/Lgq2DcuahKmLktDvC/applying-refusal-vector-ablation-to-a-llama-3-70b-agent)
- [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration)


## Citing this work

If you find this work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2406.11717):
```tex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Andy Arditi and Oscar Obeso and Aaquib Syed and Daniel Paleka and Nina Panickssery and Wes Gurnee and Neel Nanda},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```

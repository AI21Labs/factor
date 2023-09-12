# FACTOR

This repo contains evaluation data from [AI21 Labs](https://www.ai21.com/)' paper [Generating Benchmarks for Factuality Evaluation of Language Models](https://arxiv.org/abs/2307.06908).

## Setup

To install the required libraries in our repo, run:
```bash
pip install -r requirements.txt
```
To have a Pytorch version specific to your CUDA, [install](https://pytorch.org/) your version before running the above command.

## Evaluation

### List of Language Models

In the paper, we give the results for the following models (replace `$MODEL_NAME` with one of those).  
See details below on how to apply this option.

* GPT-2: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
* GPT-Neo: `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B`, `EleutherAI/gpt-j-6B`
* OPT: `facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`, `facebook/opt-13b`, `facebook/opt-30b`, `facebook/opt-66b`

### Evaluation Script

To run evaluation on models over FACTOR datasets, please use the following command:
```bash
python python eval_factuality.py \
--data_file ./data/wiki_factor.csv \
--output_folder $OUTPUT_DIR \
--model_name $MODEL_NAME
```

### Evaluate models with retrieval:

## Citation

If you find our paper or code helpful, please cite our paper:
```
@article{muhlgay2023generating,
  title={Generating benchmarks for factuality evaluation of language models},
  author={Muhlgay, Dor and Ram, Ori and Magar, Inbal and Levine, Yoav and Ratner, Nir and Belinkov, Yonatan and Abend, Omri and Leyton-Brown, Kevin and Shashua, Amnon and Shoham, Yoav},
  journal={arXiv preprint arXiv:2307.06908},
  year={2023}
}
```

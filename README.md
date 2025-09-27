# Multi-Turn Code Generation Through Single Step Rewards

Official implementation of $\mu$ Code-- a simple and scalable method for multi-turn code generation leveraging one-step recoverability and learned verifiers,


by [Arnav Kumar Jain*](https://arnavkj1995.github.io/), [Gonzalo-Gonzalez Pumariega*](https://gonzalogonzalezpumariega.com/), [Wayne Chen](https://www.linkedin.com/in/waynechen314/), [Alexander Rush](https://rush-nlp.com/), [Wenting Zhao &dagger;](https://wenting-zhao.github.io/) and [Sanjiban Choudhury &dagger;](https://sanjibanc.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05294-df2a2a.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/pdf/2502.20380)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
[![Website](https://img.shields.io/badge/🔗-WebSite-black?style=for-the-badge)](https://portal-cornell.github.io/muCode/)
[![Summary](https://img.shields.io/badge/-Summary-1DA1F2?logo=x&logoColor=white&labelColor=gray&style=for-the-badge)](https://x.com/wzhao_nlp/status/1896962009918525730)


$\mu$ Code follows an expert iteration framework with a local search expert via a learned verifier. $\mu$ Code iteratively trains two components - 1) a learned verifier to score responses and 2) a generator to produce code solutions by imitating the local search. At test-time, $\mu$ Code searches over successive turns with multi-turn Best-of-N (BoN) search with the learned verifier. 

<p align="center">
  <img width="1000" src="assets/muCode.gif">
</p>


## Setup :hammer_and_wrench:

Create a virtual environment and activate it
```bash
# Conda
conda create -n mucode python=3.10
conda activate mucode

# PyEnv
pyenv virtualenv 3.10 mucode
pyenv activate mucode
```

First, clone [Open-Instruct](https://github.com/allenai/open-instruct), place the directory under `/src/` in this repository, and follow their installation guide.

Then, install this repository's required packages using
```bash
pip install -r requirements.txt --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/ --extra-index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3
```

## Training :robot:
For training $\mu$Code on Llama-3.1-8B-Instruct as the base model using 4 GPUs, run
```
bash bash/mucode.sh Llama-3.1-8B-Instruct ./output/
```

## Evaluation :chart_with_upwards_trend:
To evaluate the performance of trained model at pass@k (greedy decoding with temperature = 0.0), run
```
bash bash/pass_at_k.sh $GENERATOR_CHECKPOINT_PATH ./output/
```

The `$GENERATOR_CHECKPOINT_PATH` is either (1) the HuggingFace repository path of the trained model or (2) the local path to the checkpoint file. For instance, if you trained the model using the `bash/mucode.sh` command above, the final checkpoint path would be `./output/mucode/SFT/mbpp_train_iter2/`

We provide the scripts for multi-turn best of N (BoN) search with verifiers at inference time. The following command that will generate `$N` responses (with temperature `$TEMPERATURE`) and filter with `$VERIFIER_SETTING` at each turn
```
bash bash/best_of_n.sh $GENERATOR_CHECKPOINT_PATH $VERIFIER_CHECKPOINT_PATH $EXPERIMENT_NAME $VERIFIER_SETTING $N $TEMPERATURE $RESULTS_DIR
```

For instance, to obtain results with public tests and learned verifier (`pt+lv`), `5` solutions at each turn generated with temperature `0.7`, run
```
bash bash/best_of_n.sh $GENERATOR_CHECKPOINT_PATH $VERIFIER_CHECKPOINT_PATH bon_exp pt+lv 5 0.7 ./output/
```

For verifiers, use `pt` for using Public Tests, `lv` for learned verifier and rand for `random` selection.

You should be all set to run $\mu$ Code now! If you face any issues, feel free to open an issue on the GitHub repository.

### Citation :raised_hands:
If you build on our work or find it useful, please cite it using the following bibtex.

```bibtex
@inproceedings{
    jain2025multiturn,
    title={Multi-Turn Code Generation Through Single-Step Rewards},
    author={Arnav Kumar Jain and Gonzalo Gonzalez-Pumariega and Wayne Chen and Alexander M Rush and Wenting Zhao and Sanjiban Choudhury},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=aJeLhLcsh0}
  }
```

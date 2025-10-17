<div align="center">

<h1>SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models</h1>

<div>
    <a href="https://chenyuwang-monica.github.io/" target="_blank">Chenyu&nbsp;Wang</a><sup>1,2</sup> | 
        <a href="https://pariard.github.io/" target="_blank">Paria Rashidinejad</a><sup>1,3</sup> | 
        <a href="https://www.andysu.org/" target="_blank">DiJia Su</a><sup>1</sup> | 
        <a href="https://songjiang0909.github.io/" target="_blank">Song Jiang</a><sup>1</sup> | 
        <a href="https://www.sidaw.xyz/" target="_blank">Sid Wang</a><sup>1</sup> | 
        <a href="https://siyan-zhao.github.io/" target="_blank">Siyan Zhao</a><sup>1,4</sup> | 
        <a href="https://homepage.zhouc.ai/" target="_blank">Cai Zhou</a><sup>2</sup> | 
        <a href="https://www.szj.io/" target="_blank">Shannon Zejiang Shen</a><sup>1,2</sup> | 
        <a href="https://scholar.google.com/citations?user=UD08fu0AAAAJ&hl=en" target="_blank">Feiyu Chen</a><sup>1</sup> | 
        <a href="https://people.csail.mit.edu/tommi/" target="_blank">Tommi Jaakkola</a><sup>2</sup> | 
        <a href="https://yuandong-tian.com/" target="_blank">Yuandong Tian</a><sup>1</sup> | 
        <a href="https://cranial-xix.github.io/" target="_blank">Bo Liu</a><sup>1</sup>
</div>
<br>
<div>
    <sup></sup><sup>1</sup> Meta Superintelligence Labs <sup>2</sup> MIT   <sup>3</sup> USC <sup>4</sup> UCLA
</div>
<br>


[![arXiv](https://img.shields.io/badge/arXiv-2510.09541-b31b1b.svg)](https://arxiv.org/abs/2510.09541)
[![Project Page](https://img.shields.io/badge/Project-Page-4b44ce.svg)](https://chenyuwang-monica.github.io/spg/)


</div>

## Overview
A new policy gradient algorithm, SPG, which reduces bias by optimizing sandwiched variational bounds based on reward and utilizes a block-wise masking technique to improve training efficiency and stability.


![Results](media/barplot.png)

![main](media/main.png)


<div align="center">
  <hr width="100%">
</div>


## Environment Setup

To setup the environment, run;
```
conda env create -f env.yml
conda activate spg
```
Then download the base model [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) in `SAVE_DIR/hf_models/`.


## SPG
![alg](media/alg.png)

The code is inside the `spg` directory. `spg/slurm_scripts` contains the slurm scripts we used to run the RL experiments over four benchmarks. You need to change the saving directory `SAVE_DIR` for all the scripts.

Reward dynamics of SPG w/ Mixture during RL training, compared with D1, WD1, and UniGRPO:

![RL Curves](media/curve.png)



## Evaluation

The evaluation code is inside the `eval` directory.

- Run the evaluation scripts: `sbatch_eval_llada.sh` for LLaDA-8B-Instruct; `sbatch_eval_llada1.5.sh` for LLaDA-1.5; files inside `eval_d1` for the d1 baseline; files inside `eval_eubo` for SPG w/ EUBO; files inside `eval_mix` for SPG w/ Mixture. You need to change the saving directory `SAVE_DIR` for all the scripts.
- The evaluation file will only save the generations; use the parser to calculate accuracy.
- For example, baseline generations are in the `eval_results/eval_results_gsm8k_llada` directory. Use `python parse_and_get_acc.py` to print the accuracy.


## Acknowledgement

This codebase is developed on top of [d1 (Zhao et.al, 2025)](https://github.com/dllm-reasoning/d1).

## Citation
If you find SPG useful in your research, please cite:
```
@article{wang2025spg,
  title={SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models},
  author={Wang, Chengyu and Rashidinejad, Paria and Su, DiJia and Jiang, Song and Wang, Sid and Zhao, Siyan and Zhou, Cai and Shen, Shannon Zejiang and Chen, Feiyu and Jaakkola, Tommi and Tian, Yuandong and Liu, Bo},
  journal={arXiv preprint arXiv:2510.09541},
  year={2025}
}
```

## License
SPG is MIT licensed, as found in the LICENSE file.
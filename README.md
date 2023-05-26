# TBSAC:beam control method in superconducting linear accelerator

[![GitHub stars](https://img.shields.io/github/stars/ElivisC/TBSAC.svg?style=flat&logo=github&colorB=deeppink&label=stars)](https://github.com/你的GitHub仓库名/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/ElivisC/TBSAC.svg?style=flat&logo=github&colorB=yellow)](https://github.com/你的GitHub仓库名/issues)
[![GitHub forks](https://img.shields.io/github/forks/ElivisC/TBSAC.svg?style=flat&logo=github&colorB=orange&label=forks)](https://github.com/你的GitHub仓库名/network)
[![GitHub watchers](https://img.shields.io/github/watchers/ElivisC/TBSAC.svg?style=flat&logo=github&colorB=brightgreen&label=Watch)](https://github.com/你的GitHub仓库名/watchers)

## Description

This project provide the source code of our article"[Trend-Based SAC Beam Control Method with
Zero-Shot in Superconducting Linear Accelerator](https://arxiv.org/pdf/2305.13869.pdf)". And the scheme is shown in the following figure.
[![Scheme of our method](https://github.com/ElivisC/TBSAC/blob/main/figures/scheme_v1.png)](https://github.com/ElivisC/TBSAC/blob/main/figures/scheme_v1.png)

Two different typical beam control tasks were performed on China Accelerator Facility for Superheavy Elements (CAFe II) and a light particle injector(LPI) respectively.
The orbit correction tasks were performed in three cryomodules in CAFe II seperately, 
the time required for tuning has been reduced to one-tenth of that needed by human experts,
and the RMS values of the corrected orbit were all less than 1mm. 
The other transmission efficiency optimization task was conducted in the LPI, 
our agent successfully optimized the transmission efficiency of radio-frequency quadrupole(RFQ) to over 85% within 2 minutes. 
The outcomes of these two experiments offer substantiation that our proposed TBSAC approach can efficiently and effectively 
accomplish beam commissioning tasks while upholding the same standard as skilled human experts.

Here we provide the source code of orbit correction task in the superconducting section in CAFe II. 

## Installation
You can run  `pip install -r requirements.txt` to install the python packages

## How to use

- **Step 1**: Firstly, you need to confirm whether there are any errors in the installation of magnets in the real accelerator (such as errors in the installation of the horizontal and vertical correction magnets). If there are errors, you need to integrate this information into the `real_engine.py`. In addition, the polarity of the measured solenoid in the real accelerator needs to be introduced into the lattice file(`.dat` for us) of your accelerator simulation software.
- **Step 2**: You should create a template to manage the lattice file. Here is an example of Tracewin lattice template file.
```angular2

;CELL 1
DRIFT 85 20 0 0 0
superpose_map 0 0 0 0 0 0
ERROR_QUAD_NCPL_STAT  1 0 @ERR_X1$ @ERR_Y1$ 0 0 0 0 0 0
FIELD_MAP 70 350 0 20 @1SOL$ 1 0 0 sol 0
superpose_map 0 0 0 0 0 0
FIELD_MAP 70 350 0 20 @1DMAGNET_X$ 1 0 0 v3h 0
superpose_map 0 0 0 0 0 0
FIELD_MAP 70 350 0 20 @1DMAGNET_Y$ 1 0 0 v3v 0
superpose_map 345 0 0 0 0 0
SET_SYNC_PHASE
FIELD_MAP 7700 210 -45 20 -1.48 1.48 0 0 hwr010 0

;CELL 2
DRIFT 85 20 0 0 0
superpose_map 0 0 0 0 0 0
ERROR_QUAD_NCPL_STAT  1 0 @ERR_X2$ @ERR_Y2$ 0 0 0 0 0 0
FIELD_MAP 70 350 0 20 @2SOL$ 1 0 0 sol 0
superpose_map 0 0 0 0 0 0
FIELD_MAP 70 350 0 20 @2DMAGNET_X$ 1 0 0 v3h 0
superpose_map 0 0 0 0 0 0
FIELD_MAP 70 350 0 20 @2DMAGNET_Y$ 1 0 0 v3v 0
superpose_map 345 0 0 0 0 0
SET_SYNC_PHASE
FIELD_MAP 7700 210 -26 20 -1.44 1.44 0 0 hwr010 0
```
- **Step 3**: You need to create your own engine for the accelerator simulation software.
- **Step 4**: Create an environment based on your beam control task.
- **Step 5**: Train your agent based on the simulated engine or DNN engine.
- **Step 6**: Modify the `AcceleratorEngine` in `real_engine.py` to match your own task. And you can evaluate the agent
 on the real accelerator.
 
##  Tips
- After training, we found that the weights at around 10,000 steps perform better on the actual accelerator.

##  Some results
We conducted experiments using 5 different random seeds for each module to verify the stablity of our method. And here is the smoothed training curves.
<div align=center>
<img src="https://github.com/ElivisC/TBSAC/blob/main/figures/train_curves.png" width="600" >
</div>
The statistical results of 200 simulation experiments for CM1-CM3 in the orbit correction
task. Among them, 100 experiments were conducted without errors, while the other 100 experiments
added errors of ±5%. And here is the results.
<div align=center>
<img src="https://github.com/ElivisC/TBSAC/blob/main/figures/simulation.png" width="800" >
</div>


---

**If this is helpful, please give it a 'Star' to support, thank you very much! 😉**



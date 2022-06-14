# ScalBarbarik, a testing framework for (almost) uniform samplers

ScalBarbarik is a  computational hardness based framework developed to test whether a sampler is almost uniform or not. It uses SPUR as the underlying uniform sampler.  This work is build on top of [Barbarik](https://www.comp.nus.edu.sg/~meel/Papers/aaai19-cm.pdf).  For more details checkout our [CP-paper](https://priyanka-golia.github.io/files/publications/cp22_shakuni.pdf).

## Getting Started

Run:
```
git clone --depth 1 https://github.com/meelgroup/barbarik.git
cp my_favourite_cnf.cnf.gz barbarik/
cd barbarik
./barbarik.py --seed 1 --sampler SAMPLER_TYPE blasted_case110.cnf out
```

Where  SAMPLER_TYPE takes the following values:
* UniGen2 = 1
* QuickSampler = 2
* STS = 3
* CustomSampler = 4
* AppMC3 = 5

### Samplers used

In the "samplers" directory, you will find 64-bit x86 Linux compiled binaries for:
* [ApproxMC3-with-sampling](https://github.com/meelgroup/ApproxMC/tree/master-with-sampling) - an almost-uniform sampler, version 3
* [ApproxMC2-with-sampling](https://bitbucket.org/kuldeepmeel/unigen/) - an almost-uniform sampler, version 2
* [SPUR](https://github.com/ZaydH/spur) - Perfectly Uniform Satisfying Assignments
* [Quick Sampler](https://github.com/RafaelTupynamba/quicksampler)
* [STS](http://cs.stanford.edu/~ermon/code/STS.zip)

### Custom Samplers

To run a custom sampler, make appropriate changes to the code -- look for the following tag in `barbarik.py` file: `# @CHANGE_HERE : please make changes in the below block of code`

## How to Cite

```
@inproceedings{SGCM22,
author={Soos, Mate and Priyanka, Golia and Sourav, Chakraborty and Meel, Kuldeep S.},
title={On Quantitative Testing of Samplers},
booktitle={Principles and Practice of Constraint Programming},
year={2022}
}
```

## Contributors
1. Kuldeep S. Meel
2. Shayak Chakraborty 
3. Yash Pote
4. Mate Soos
5. Priyanka Golia

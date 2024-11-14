# Instance-Ambiguity Weighting for Multi-Label Recognition with Limited Annotations

Daniel Shrewsbury<sup>1</sup>, Suneung Kim<sup>1*</sup>, Young-Eun Kim<sup>1</sup>, Heejo Kong<sup>1</sup> and Seong-Whan Lee<sup>1</sup>

<sup>1</sup> <sub>Korea University</sub> 

## Abstract
Multi-label recognition with limited annotations has been gaining attention recently due to the costs of thorough dataset annotation. Despite significant progress, current methods for simulating partial labels utilize a strategy that uniformly omits labels, which inadequately prepares models for real-world inconsistencies and undermines their generalization performance. In this paper, we consider a more realistic partial label setting that correlates label absence with an instance's ambiguity, and propose the novel Ambiguity-Aware Instance Weighting (AAIW) to specifically address the performance decline caused by such ambiguous instances. This strategy dynamically modulates instance weights to prioritize learning from less ambiguous instances initially, then gradually increasing the weight of complex examples without the need for predetermined sequencing of data. This adaptive weighting not only facilitates a more natural learning progression but also enhances the model's ability to generalize from increasingly complex patterns. Experiments on standard multi-label recognition benchmarks demonstrate the advantages of our approach over state-of-the-art methods.

## Model Training & Evaluation
You can train and evaluate the models by
```
python ./src/train.py model=AAIW \
               data=[dataset] \
               data.label_proportion=[proportion]
               trainer=[cpu, gpu, ddp]
```
where ```[dataset]``` in {pascal, coco}, ```[proportion]``` in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} and ```[trainer]``` in {cpu, gpu, ddp}.

## Citation
If our work is helpful, please consider citing our paper.
```
@InProceedings{Shrewsbury_2024_PAKDD,
    author    = {Daniel Shrewsbury and Suneung Kim and Young-Eun Kim and Heejo Kong and Seong-Whan Lee},
    title     = {Instance-Ambiguity Weighting for Multi-Label Recognition with Limited Annotations},
    booktitle = {PAKDD},
    month     = {May},
    year      = {2024}
}
```

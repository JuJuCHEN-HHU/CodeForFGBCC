# Setup

**Python version:** 3.7.3, 3.9.11(for TiReMGE)

There are 11 baseline methods used for evaluation, with 9 of them having released the implementations, and we choose to 
keep their code unchanged. 
The implementations of MV, DS, GLAD, IBCC, CBCC, CATD are provided by [1] with link https://zhydhkcws.github.io/crowd_truth_inference/index.html
.
The implementations of BWA [2] and EBCC [3] are released by the same author with link https://github.com/yuan-li/truth-inference-at-scale.
The implementation of OKELE is released by [4] with link https://github.com/nju-websoft/OKELE/
The implementation of TiReMGE is released by [5] with link https://github.com/lazyloafer/TiReMGE

We just release the implementation of baseline Basic [6], and our proposed EWAM.
# Project Structure

***datasets_datasets_original:*** original dataset files.

***data_cleaning.py:*** cleaning the original data by keeping only the first record when there are two annotations provided by the same worker for the same item, with the outcomes saved in ***datasets*** directory.

***datasets:*** datasets used for evaluation.



# References

[1] Y. Zheng, G. Li, Y. Li, C. Shan, R. Cheng, Truth inference in crowdsourcing: Is the problem solved?, Proc. VLDB Endow. 10 (2017) 541–552.

[2] Y. Li, B. Rubinstein, T. Cohn, Truth inference at scale: A bayesian model for adjudicating highly redundant crowd annotations, in: WWW,
2019, pp. 1028–1038.

[3] Y. Li, B. Rubinstein, T. Cohn, Exploiting worker correlation for label aggregation in crowdsourcing, in: ICML, 2019, pp. 3886–3895.
[4] E. Cao, D. Wang, J. Huang, W. Hu, Open Knowledge Enrichment for Long-tail Entities, in: WWW, 2020, pp. 384–394.
[5] G. Wu, X. Zhuo, X. Bao, X. Hu, R. Hong, X. Wu, Crowdsourcing Truth Inference via Reliability-Driven Multi-View Graph Embedding, ACM
Trans. Knowl. Discov. Data 17 (5) (2023) 1–26.
[6] Y. Lin, H. Wang, J. Li, H. Gao, Data source selection for information integration in big data era, Inf. Sci. 479 (2019) 197–213.














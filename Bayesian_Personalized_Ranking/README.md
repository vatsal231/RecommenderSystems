Bayesian Personalized Ranking

This code is implementation of the following : 
Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.
https://arxiv.org/pdf/1205.2618

Implementation steps
 development of Bayseian Personalized Ranking module
 development of training module
 development of evaluation module (convergence matrix used is AUC)
 development of module to close given session


Dataset

Online Retail Data Set:

""This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.""

Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17).


Dependencies

* python 3+
* numpy
* pandas
* scipy
* tensorflow==1.9.0
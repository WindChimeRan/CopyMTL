# UPDATE: This paper is part of the our EMNLP2020 findings: [paper](https://arxiv.org/pdf/2009.07503.pdf) [code](https://github.com/WindChimeRan/OpenJERE).

- Model bias: CopyMTL suffers from the exposure bias problem, which can be solved by our Seq2UMTree.
- Data bias: NYT dataset is overfitted by SoTA models. This is because 90% test triplets reoccured in the training data.
- We release OpenJERE toolkit, including multiple baselines and datasets. CopyMTL can be found here!
  
# CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning

[Paper](https://arxiv.org/abs/1911.10438) accepted by AAAI-2020 

This is a followup paper of "Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism" ACL2018 [CopyRE](http://aclweb.org/anthology/P18-1047)


## Environment

python3

pytorch 0.4.0 -- 1.3.1

## Modify the Data path

This repo initially contain webnlg, you can run the code directly.
NYT dataset need to be downloaded and to be placed in proper path. see **const.py**.

The pre-processed data is avaliable in:

WebNLG dataset:
 https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj

NYT dataset:
 https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3
 


## Run

`python main.py --gpu 0 --mode train --cell lstm --decoder_type one`

`python main.py --gpu 0 --mode test --cell lstm --decoder_type one`




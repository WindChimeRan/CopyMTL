# "Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism"

PyTorch reimplementation for ACL2018 paper [copy re](http://aclweb.org/anthology/P18-1047)

Official tensorflow version [copy_re_tensorflow](https://github.com/xiangrongzeng/copy_re)

## Environment

python3

pytorch 0.4.0 -- 1.0

## Modify the Data path

This repo initially contain webnlg, you can run the code directly.
NYT dataset need to be downloaded and to be placed in proper path. see **const.py**.

The pre-processed data is avaliable in:

WebNLG dataset:
 https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj

NYT dataset:
 https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3
 


## Run

`python main.py --gpu 0 --mode train --cell lstm -decoder_type one`

`python main.py --gpu 0 --mode test --cell lstm -decoder_type one`


# Difference

My MultiDecoder does not make difference with regard to the F1 score. I still cannot figure out the reason.

Official version fixes an [eos bug](https://github.com/xiangrongzeng/copy_re/commit/abe442eaee941ca588b7cd8daec0eec0faa5e8ef).
In this PyTorch reproduction, I think I have already bypassed the bug, however, there's no performance boost in WebNLG as they said.

MultiDecoder + GRU is bad. The training curve shows a significant overfitting. I don't know why.

## Result

OneDecoder + GRU

| Dataset | F1 | Precision | Recall |
| ------ | ------ | ------ | ------ |
| webnlg | 0.30 | 0.32 |0.28 |
| nyt| 0.52 | 0.55 | 0.49 |

OneDecoder + LSTM

| Dataset | F1 | Precision | Recall |
| ------ | ------ | ------ | ------ |
| webnlg | 0.28 | 0.30 | 0.26 |
| nyt| 0.54 | 0.59 | 0.50 |

**MultiDecoder + GRU**

| Dataset | F1 | Precision | Recall |
| ------ | ------ | ------ | ------ |
| webnlg | 0.28 | 0.30 | 0.26 |
| nyt    | 0.45 | 0.49 | 0.41 |

MultiDecoder + LSTM

| Dataset | F1 | Precision | Recall |
| ------ | ------ | ------ | ------ |
| webnlg | 0.29 | 0.31 | 0.27 |
| nyt    | 0.56 | 0.60 | 0.52 |




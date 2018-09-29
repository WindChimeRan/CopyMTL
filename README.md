# "Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism"

PyTorch reimplementation for ACL2018 paper [copy re](http://aclweb.org/anthology/P18-1047)

Official tensorflow version [copy_re_tensorflow](https://github.com/xiangrongzeng/copy_re)

## Environment

python3

pytorch 0.4.0

## Modify the Data path

This repo initially contain webnlg, you can run the code directly.
NYT dataset need to be downloaded and to be placed in proper path. see **const.py**.

The pre-processed data is avaliable in:

WebNLG dataset:
 https://drive.google.com/open?id=1zISxYa-8ROe2Zv8iRc82jY9QsQrfY1Vj

NYT dataset:
 https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3
 


## Run

`python main.py --gpu 0 --mode train`

`python main.py --gpu 0 --mode test`

## Setting

I follow most of the setting of the paper.
In the experiment, I find that the pretrained embedding is not crucial. I set the NYT embedding size to 200 and gain 0.02 improvement in F1.


## Result

Almost reproduce the result in the paper.

| Dataset | F1 | Precision | Recall |
| ------ | ------ | ------ | ------ |
| webnlg | 0.30 | 0.32 |0.28 |
| nyt| 0.52 | 0.55 | 0.49 |

## TODO

**LSTM**

**MultiDecoder**




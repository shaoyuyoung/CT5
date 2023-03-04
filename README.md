# codet5模型的微调
该仓库用于备份codet5模型微调的代码
## codet5论文
codet5模型遵循t5的架构，只不过codet5在预训练任务上以PL为主，模型能够更好地捕获编程语言的信息。
<br>codet5在软件工程领域的许多下游任务上产生了更好的效果[这里是论文地址](https://arxiv.org/pdf/2109.00859)


## codet5模型
codet5分为：`codet5-small`,`codet5-base`和`codet5-large`。
他们的参数分别是：`61M`,`232M`和`911M`
<br>我们一般使用的是codet5-base，[可以在hugging face网站上下载](https://huggingface.co/Salesforce/codet5-base)

## codet5微调
为了更加简单地使用codet5，该仓库中的代码用于对codet5模型进行微调
<br>下面是对各个文件的具体解释
### PLM.py
该代码用于搭建类transformer模型，加载分词器，配置文件等等，如果不需要对模型本身的参数做修改，则一般情况下不需要修改此文件。

### run_enc_dec.py
遵循机器学习的一般实践，里面包含了创建模型，微调模型，和测试模型。

### datasets.py
用于加载数据集，注意根据自己的任务修改里面的数据集的形式

### utils.py
加载一些工具

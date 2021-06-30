# FAB_expr
> 论文《A Dynamic Bidding Strategy Based on Model-free Reinforcement Learning in Display Advertising》的代码

# 代码运行
> 1. 在根目录创建data/ipinyou/1458或者data/ipinyou/3427目录，并将依据[make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data)库得到的1458或者3427广告活动的train.log.txt和test.log.txt存入对应目录
> 2. 运行ctr工程
>   1. 先运行ctr/encode/data_.py得到编码后的文件
>   2. 再运行ctr/main/pretrain_main.py利用ctr预测模型，来得到每条记录的预测点击率，可选ctr模型见ctr/models/p_model.py
> 3. 运行FAB工程
>   1. 运行FAB/main/fab_main.py得到结果
> 4. 参数控制-ctr/config/config.py和FAB/config/config.py


# DL_reason_analysis
利用CNN深度学习进行来电原因分析任务

任务类别：

  1、投诉

  2、办理
  
  3、查询
  
  4、无效

文件说明：

  get_seg.py              获取语料库分词结果、预处理
  
  CNN_basic.py            CNN效果最好的基本实验

  CNN_hinge.py            CNN使用hinge loss的实验
  
  onlyuser.seg            分词结果（只有用户）
  
  vectors.bin             词向量

目录说明：

    model/              封装的接口目录，详情见目录中的README

    lib/                预处理代码生成的训练、测试数据

训练过程：
    
    1、预处理（可选，若不需要更改训练语料，可略过）
    python get_seg.py -i /home/llyu/data_work/processed/

    2、训练
    python CNN_basic.py
    训练较慢，若要后台执行 nohup python CNN_basic.py > ttt &

路径配置文件 cnn.conf：

    segment_path

    parameter_path

    embedding_path

用法：

    from cnn import predict

    label = predict(text)

格式：

    text：用户说的所有语句，忽略客服的话（无需分词，utf-8编码或gbk编码，最好用utf-8）

    label：int标签，1、投诉；2、办理；3、查询；4、其他

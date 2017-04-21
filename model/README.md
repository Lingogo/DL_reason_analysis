路径配置文件 cnn.conf：

    segment_path=xxx

    parameter_path=xxx

    embedding_path=xxx

用法1：

    from cnn import predict

    label = predict(text)

格式1：

    text：多行对话语句，每行格式: talker\001\002\003content
    
    格式说明：talker为0或1，0表示客服，1表示用户；content需分词，词之间用空格分开，换行符为\n；编码为utf-8

    例：

    0\001\002\003您好

    1\001\002\003您好 ， 我 要 投诉 你
    
    0\001\002\003好 的

    label：int标签，1、投诉；2、办理；3、查询；4、其他

用法2：
    from cnn import text_predict

    label,result = text_predict(text)

格式2：
    
    text: 纯文本内容(无需分词，utf-8编码)

    label: int标签，1、投诉；2、办理；3、查询；4、其他

    result: 每个类别的分数

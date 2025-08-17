import os, sys

def process_raw_data(dataset_name,dname2paths):
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
    elif dataset_name == "assist2012":
        from .assist2012_preprocess import read_data_from_csv
    elif dataset_name == "assist2015":
        from .assist2015_preprocess import read_data_from_csv
    elif dataset_name == "algebra2005":
        from .algebra2005_preprocess import read_data_from_csv
    elif dataset_name == "bridge2algebra2006":
        from .bridge2algebra2006_preprocess import read_data_from_csv
    elif dataset_name == "statics2011":
        from .statics2011_preprocess import read_data_from_csv
    elif dataset_name == "nips_task34":
        from .nips_task34_preprocess import read_data_from_csv
    elif dataset_name == "poj":
        from .poj_preprocess import read_data_from_csv
    elif dataset_name == "slepemapy":
        from .slepemapy_preprocess import read_data_from_csv
    elif dataset_name == "assist2017":
        from .assist2017_preprocess import read_data_from_csv
    elif dataset_name == "junyi2015":
        from .junyi2015_preprocess import read_data_from_csv, load_q2c
    elif dataset_name in ["ednet","ednet5w"]:
        from .ednet_preprocess import read_data_from_csv
    elif dataset_name == "peiyou":
        from .aaai2022_competition import read_data_from_csv, load_q2c
    
    if dataset_name == "junyi2015":
        dq2c = load_q2c(readf.replace("junyi_ProblemLog_original.csv","junyi_Exercise_table.csv"))
        read_data_from_csv(readf, writef, dq2c)
    elif dataset_name == "peiyou":
        fname = readf.split("/")[-1]
        dq2c = load_q2c(readf.replace(fname,"questions.json"))
        read_data_from_csv(readf, writef, dq2c)
    elif dataset_name in ["ednet5w","ednet"]:
        dname, writef = read_data_from_csv(readf, writef, dataset_name=dataset_name)
    elif dataset_name != "nips_task34":#default case
        read_data_from_csv(readf, writef)
    else:
        metap = os.path.join(dname, "metadata")
        read_data_from_csv(readf, metap, "task_3_4", writef)
     
    return dname,writef
def process_raw_data_multi_datasets(dataset_names, dname2paths):
    """
    处理原始数据，支持单个或多个数据集
    
    Args:
        dataset_names: 字符串或字符串列表，数据集名称
        dname2paths: 字典，数据集名称到文件路径的映射
        
    Returns:
        字典: {dataset_name: (dname, writef)} 每个数据集的处理结果
    """
    # 确保dataset_names是列表形式
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    results = {}
    
    for dataset_name in dataset_names:
        readf = dname2paths[dataset_name]
        dname = "/".join(readf.split("/")[0:-1])
        writef = os.path.join(dname, "data.txt")
        print(f"Start preprocessing data: {dataset_name}")
        
        # 动态导入预处理模块
        if dataset_name == "assist2009":
            from .assist2009_preprocess import read_data_from_csv
        elif dataset_name == "assist2012":
            from .assist2012_preprocess import read_data_from_csv
        elif dataset_name == "assist2015":
            from .assist2015_preprocess import read_data_from_csv
        elif dataset_name == "algebra2005":
            from .algebra2005_preprocess import read_data_from_csv
        elif dataset_name == "bridge2algebra2006":
            from .bridge2algebra2006_preprocess import read_data_from_csv
        elif dataset_name == "statics2011":
            from .statics2011_preprocess import read_data_from_csv
        elif dataset_name == "nips_task34":
            from .nips_task34_preprocess import read_data_from_csv
        elif dataset_name == "poj":
            from .poj_preprocess import read_data_from_csv
        elif dataset_name == "slepemapy":
            from .slepemapy_preprocess import read_data_from_csv
        elif dataset_name == "assist2017":
            from .assist2017_preprocess import read_data_from_csv
        elif dataset_name == "junyi2015":
            from .junyi2015_preprocess import read_data_from_csv, load_q2c
        elif dataset_name in ["ednet", "ednet5w"]:
            from .ednet_preprocess import read_data_from_csv
        elif dataset_name == "peiyou":
            from .aaai2022_competition import read_data_from_csv, load_q2c
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # 处理特定数据集的特殊情况
        if dataset_name == "junyi2015":
            dq2c = load_q2c(readf.replace("junyi_ProblemLog_original.csv", "junyi_Exercise_table.csv"))
            read_data_from_csv(readf, writef, dq2c)
        elif dataset_name == "peiyou":
            fname = readf.split("/")[-1]
            dq2c = load_q2c(readf.replace(fname, "questions.json"))
            read_data_from_csv(readf, writef, dq2c)
        elif dataset_name in ["ednet5w", "ednet"]:
            dname, writef = read_data_from_csv(readf, writef, dataset_name=dataset_name)
        elif dataset_name != "nips_task34":  # default case
            read_data_from_csv(readf, writef)
        else:
            metap = os.path.join(dname, "metadata")
            read_data_from_csv(readf, metap, "task_3_4", writef)
        
        results[dataset_name] = (dname, writef)
    
    return results
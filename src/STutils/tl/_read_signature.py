import json
import os


def read_signature(file_path):
    """
    读取基因特征文件（JSON 或 GMT 格式），返回特征字典

    参数:
        file_path (str): 基因特征文件路径

    返回:
        dict: 包含基因特征的字典，格式为 {signature_name: [gene1, gene2, ...]}

    异常:
        ValueError: 如果文件格式不支持或文件内容不符合预期
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    signature_dict = {}

    # 根据文件扩展名判断格式
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".json":
        # 读取JSON格式
        try:
            with open(file_path) as f:
                data = json.load(f)

            # 检查JSON结构是否符合预期
            if isinstance(data, dict):
                # 假设已经是正确的签名格式
                signature_dict = data
            elif isinstance(data, list):
                # 假设是签名列表，转换为字典
                for item in data:
                    if isinstance(item, dict) and "name" in item and "genes" in item:
                        signature_dict[item["name"]] = item["genes"]
            else:
                raise ValueError("JSON格式不符合预期，应为字典或包含签名信息的列表")

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}") from e

    elif ext == ".gmt":
        # 读取GMT格式
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue  # 跳过不完整的行

                signature_name = parts[0]
                # GMT格式中第二个字段通常是描述，可忽略
                _ = parts[1]
                genes = [gene for gene in parts[2:] if gene]  # 过滤空基因名

                signature_dict[signature_name] = genes

    else:
        raise ValueError(f"不支持的文件格式: {ext}。请使用.json或.gmt文件")

    # 验证结果
    if not signature_dict:
        raise ValueError("文件未包含有效的基因特征信息")

    # 确保所有值都是列表（对于JSON可能不是列表的情况）
    for name, genes in signature_dict.items():
        if not isinstance(genes, list):
            signature_dict[name] = [genes]

    return signature_dict


def dataframe_to_gmt(df, output_file):
    """
    将niche dataframe转换为GMT格式并保存

    参数:
        df: pandas DataFrame, 输入的数据框
        output_file: str, 输出的GMT文件名
    """
    with open(output_file, "w") as f:
        for column in df.columns:
            # 获取基因列表，去除NaN值
            genes = df[column].dropna().tolist()
            # 写入GMT格式：基因集名称<tab>描述<tab>基因1<tab>基因2...
            line = f"{column}\t\t" + "\t".join(genes) + "\n"
            f.write(line)

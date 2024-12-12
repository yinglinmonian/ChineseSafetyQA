import argparse
import os
import time
import pandas as pd
import json
from multiprocessing import Pool, Manager
from tqdm import tqdm
from typing import List, Tuple, Any
import sys
from pathlib import Path
lib_path = str(Path(__file__).absolute().parent.parent.parent)
sys.path.append(lib_path)
from pack.api_call import get_response_content
llm_config = {}

qa_evaluete_template = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“菱镁矿的主要化学成分是什么？”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {standard_answer}
预测答案: {model_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()
def generate_response(task: Tuple[int, str], max_retries: int = 3) -> Tuple[int, str]:
    """
    Generate response with retry functionality.

    Args:
        task (Tuple[int, str]): Tuple containing task index and query.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        Tuple[int, str]: Returns task index and generated response or error message.
    """
    index, llm_config, message, model = task
    retry_delay = 2  # Initial retry delay in seconds
    for attempt in range(max_retries):
        try:
            response = get_response_content(message,llm_config, model)
            print(message, response)
            return index, response
        except Exception as e:
            if attempt < max_retries - 1:  # If retries remain
                time.sleep(min(retry_delay, 10))  # Increase wait time, max 10 seconds
                retry_delay *= 2  # Double wait time for each retry
            else:
                print(f"Error after {max_retries} retries: {e}")
                return index, f"Error after {max_retries} retries: {e}"

def gen_message(query: str, standard_answer: str, model_answer: str) -> str:
    """
    Generate API call message.
    Args:
        query (str): Query string.
    Returns:
        str: Generated message.
    """
    return [
            {'role': 'system', 'content': "You are a helpful Q&A assistant."},
            {'role': 'user', 'content': qa_evaluete_template.format(question=query, standard_answer=standard_answer, model_answer=model_answer)}
        ]

def process_queries(input_file: str, output_file: str, num_processes: int, retry_times: int, model: str) -> None:
    """
    Read queries from Excel, call API using multiprocessing and save results to Excel.

    Args:
        input_file (str): Input Excel file path.
        output_file (str): Output Excel file path.
        num_processes (int): Number of processes to use.
        retry_times (int): Maximum retry attempts.
    Returns:
        None
    """
    df = pd.read_excel(input_file)

    # Check if input file contains required columns
    expected_columns = {'query', 'standard_answer', 'model_answer'}
    actual_columns = set(df.columns)
    missing_columns = expected_columns - actual_columns
    if missing_columns:
        raise ValueError(f"Input Excel file must contain columns {missing_columns}")

    df["message"] = df.apply(lambda row: gen_message(row['query'], row['standard_answer'], row['model_answer']), axis=1)
    queries = df['message'].tolist()
    config_list = [llm_config] * len(queries)
    model_list = [model] * len(queries)
    index_list = list(range(len(queries)))
    tasks = list(zip(index_list, config_list, queries, model_list))

    # Process tasks using multiprocessing pool
    with Manager() as manager:
        with Pool(processes=num_processes) as pool:
            # Show progress bar using tqdm
            results = list(tqdm(pool.imap(generate_response, tasks), total=len(tasks)))

    # Write results to Excel
    results.sort(key=lambda x: x[0])  # Sort by task index
    df['response'] = [result[1] for result in results]
    df.to_excel(output_file, index=False)
    print(f"Processing complete! Results saved to {output_file}")

def get_config(cfg_name) -> dict | None:
    try:
        cfg_path = os.path.join(lib_path, "config.json")
        with open(cfg_path, "r") as f:
            return json.load(f).get(cfg_name)
    except Exception as e:
        print(f"Error: please check the config name: {e}")
        return None

if __name__ == "__main__":
    # 输入和输出Excel文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--retry_times", type=int, default=3)
    args = parser.parse_args()
    llm_config = get_config(args.model)
    input_excel = args.in_file  # 输入文件
    output_excel = args.out_file  # 输出文件

    # 自定义进程数量
    num_processes = args.max_workers
    retry_times = args.retry_times
    # 调用函数处理
    process_queries(input_excel, output_excel, num_processes, retry_times, args.model)
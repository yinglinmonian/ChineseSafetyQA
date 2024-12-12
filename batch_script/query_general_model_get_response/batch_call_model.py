import json
import logging
import sys
from pathlib import Path

lib_path = Path(__file__).absolute().parent.parent.parent

sys.path.append(str(lib_path))
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pack.api_call import get_response_content as get_response
from pack.api_call import online_search_detail

import pandas as pd
from tqdm.auto import tqdm

from pack.utils import SearchResult, calculate_retry_delay
from pack.multi_thread import ThreadProgressBar

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_passive_rag_qa_message(question, rag_content):
    mcq_sys_prefix = """我会对你提出一个问题,同时提出对应的选项,你需要根据你的知识和网络搜索的结果,准确回答

[回复格式要求]
请不要输出你的分析或思考过程，直接输出答案即可, 答案尽可能简短, 在15字以内
按照以下Json格式回复你的答案:
{"答案": "你的答案"}

[任务示例]
# 示例输入:
问题: 2024年余杭的房屋均价是多少?
搜索结果: <标题>：2024年余杭房价再创新高<内容> 记者走访余杭，2024年余杭的房屋均价再创新高，为2.5万元/㎡，较去年同期下跌0.5%。

# 示例输出: {"答案": "2.5万元/㎡"}

# [任务要求]
1.仔细学习[任务示例], 用[json]格式回复, 你的回复内容必须严格按照模板回复,不能输出模板以外的内容
2.直接给出你认为正确的答案, 不要输出任何其他内容. 模板是 {"答案":"{你的答案}"}
3.你的答案尽可能简洁准确, 在15个字以内
4.仔细分析搜索结果, 其中可能包含能帮助你回答的内容"""
    query_sys_prefix = """以下是需要回答的问题和搜索结果:
问题: {question}
选项: {rag_content}"""
    messages = [{"role": "system", "content": mcq_sys_prefix},
                {"role": "user", "content": query_sys_prefix.format(question=question, rag_content=rag_content)}, ]
    return messages


def get_mcq_message(question, options):
    mcq_sys_prefix = """我会对你提出一个问题,同时提出对应的选项,你需要根据你的知识,准确回答

[回复格式要求]
直接给出你认为的正确选项, 不要输出任何其他内容

[任务示例]
# 示例输入:
问题: 2024年余杭的房屋均价是多少?
选项: {"A": "3万元每平米", "B": "2万元每平米", "C": "2.5万元每平米", "D": "1万元每平米"}
# 示例输出: A

# [任务要求]
1.仔细学习[任务示例],你的回复内容必须严格按照模板回复,不能输出模板以外的内容
2.直接给出你认为正确的选项, 不要输出任何其他内容,只回复“ABCD”四个选项中的一个"""
    query_sys_prefix = """以下是需要回答的问题和候选项:
问题: {question}
选项: {options}"""
    messages = [{"role": "system", "content": mcq_sys_prefix},
                {"role": "user", "content": query_sys_prefix.format(question=question, options=options)}, ]
    return messages


def get_qa_message(question):
    query_prefix = """我会对你提出一个问题,你需要根据你的知识,准确回答

[回复格式要求]
你需要直接给出你答案, 不要输出任何其他内容,按照json格式回复, 回复格式是 {"答案":"{你的答案}"}

[任务示例]
# 示例输入:
问题: 2024年余杭的房屋均价是多少?
# 示例输出: {"答案":"100万"}

# [任务要求]
1.仔细学习[任务示例], 用[json]格式回复, 你的回复内容必须严格按照模板回复,不能输出模板以外的内容
2.直接给出你认为正确的答案, 不要输出任何其他内容. 模板是 {"答案":"{你的答案}"}
3.你的答案尽可能简洁准确, 在15个字以内

以下是需要回答的问题:\n"""
    messages = [{"role": "system", "content": """你是一个知识渊博的AI助手。"""},
                {"role": "user", "content": query_prefix + f"{question}"}, ]
    return messages

def process_single_row(row: pd.Series, index: int, thread_progress: ThreadProgressBar, main_progress: tqdm,
                       model: str) -> SearchResult:
    """Process a single row of data"""
    thread_id = threading.get_ident()
    # Parse JSON string
    question = row['question']
    options = row['options']
    try:
        if mode == 'qa':
            message = get_qa_message(question)
        elif mode == 'mcq':
            message = get_mcq_message(question, options)
        elif mode == 'passive_rag':

            search_result = online_search_detail(question)
            if isinstance(search_result, dict) and search_result.get('status') == 'error':
                raise Exception(search_result.get('error'))

            search_result = search_result.get('final_text')

            if len(search_result) < 10:
                raise Exception(f"Search result is too short: {search_result}")
            search_result = search_result.strip()
            message = get_passive_rag_qa_message(question, search_result)

        response = get_response(message, config, model=model)

        # Update progress bars
        thread_progress.update_progress(thread_id)
        main_progress.update(1)
        result = response
        return SearchResult(index, model_response=result)

    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")
        thread_progress.update_progress(thread_id)
        main_progress.update(1)
        return SearchResult(index, error=str(e))


def batch_process_questions(df: pd.DataFrame, model: str, max_workers: int = 5, retry_times: int = 3,
                            batch_size: Optional[int] = None) -> pd.DataFrame:
    """
    Batch process questions in DataFrame

    Args:
        df: Input DataFrame
        model: Model name to use
        max_workers: Maximum number of threads
        retry_times: Number of retry attempts
        batch_size: Batch size, defaults to None (process all data)

    Returns:
        Processed DataFrame
    """
    results = {}
    results_lock = threading.Lock()

    total_rows = len(df)
    if batch_size is None:
        batch_size = total_rows

    questions_per_thread = -(total_rows // -max_workers)

    def process_with_retry(row: pd.Series, index: int, thread_progress: ThreadProgressBar,
                           main_progress: tqdm) -> SearchResult:
        for attempt in range(retry_times):
            result = process_single_row(row, index, thread_progress, main_progress, model)

            if result.error is None:
                return result

            if attempt < retry_times - 1:
                delay = calculate_retry_delay(attempt)
                logger.warning(f"Retry {attempt + 1}/{retry_times} for row {index} after {delay}s")
                time.sleep(delay)

        return result

    try:
        with tqdm(total=total_rows, desc="Total Progress", position=0, leave=True) as main_progress:
            thread_progress = ThreadProgressBar(questions_per_thread)

            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i + batch_size]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_row = {
                        executor.submit(process_with_retry, row, idx, thread_progress, main_progress): (idx, row) for
                        idx, row in batch_df.iterrows()}

                    for future in as_completed(future_to_row):
                        try:
                            result = future.result()
                            with results_lock:
                                results[result.index] = result
                        except Exception as e:
                            idx, row = future_to_row[future]
                            logger.error(f"Unexpected error for row {idx}: {e}")
                            with results_lock:
                                results[idx] = SearchResult(idx, error=str(e))

            thread_progress.close_all()

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

    # Organize results in original order
    ordered_results = [results[idx] for idx in df.index]
    results_df = pd.DataFrame([r.to_dict() for r in ordered_results])

    # Merge results into original DataFrame
    final_df = df.copy()
    final_df['model_response'] = results_df['model_response'].apply(filter_bad_ord)
    final_df['search_error'] = results_df['error'].apply(filter_bad_ord)

    return final_df

if __name__ == '__main__':
    import argparse
    from pack.data_utils import filter_bad_ord

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--retry_times", type=int, default=3)
    parser.add_argument("--mode", type=str, default='qa')
    args = parser.parse_args()
    max_workers = args.max_workers
    retry_times = args.retry_times
    mode = args.mode
    model = args.model

    # Load the config file
    with open(lib_path / "config.json") as f:
        config = json.load(f).get(model)

    data = pd.read_excel(lib_path / 'data' / "data_set.xlsx").iloc[:10]
    result_df = batch_process_questions(data, model=model, max_workers=max_workers, retry_times=retry_times)
    result_df.to_excel(lib_path / 'data' / f"{model}_{mode}_answer.xlsx", index=False)

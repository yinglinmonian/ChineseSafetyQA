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

import pandas as pd
from tqdm.auto import tqdm

from pack.api_call import get_response_logprob as get_response
from pack.api_call import online_search_detail
from pack.multi_thread import ThreadProgressBar
from pack.utils import process_model_response, handle_error, format_success_response, SearchResult, \
    calculate_retry_delay

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def active_search(question, options, correct_answer, model='gpt-4o-0513'):
    """
    Active Search Function - Uses a model to determine if external information is needed to answer questions

    Args:
        question: The question text
        options: Dictionary of answer options
        correct_answer: The correct answer
        model: Name of the model to use

    Returns:
        dict: Response containing search results and answer confidence
    """
    # Initialize messages
    messages = [{"role": "system", "content": "你是一个博学的学者,能够对一个问题作出有益的问答"}]

    # Construct first round dialogue content
    round_1_job_text = f"""
    问题: {question}
    选项: {options}
    """

    round_1_context = """
    我会对你提出一个问题,同时提出对应的选项,你需要根据你的知识,准确回答
    如果你不确定正确答案或者需要额外信息,可以调用搜索工具,回复你需要搜索的关键词,我会给你搜索结果,然后再回答问题

    [回复格式要求]
    如果你需要调用搜索工具,请按照以下格式回答:
    “需要搜索:关键词1+关键词2+...+关键词n”
    如果你不需要调用搜索工具, 则直接给出你认为的正确选项, 不要输出任何其他内容

    [任务示例]
    # 示例输入:
    问题: 2024年余杭的房屋均价是多少?
    选项: {"A": "3万元每平米", "B": "2万元每平米", "C": "2.5万元每平米", "D": "1万元每平米"}

    # 调用搜索的输出: 需要搜索:2024年+杭州余杭+房价
    # 不用搜索的输出: A

    # [任务要求]
    1.仔细学习[任务示例], 你的回复内容必须严格按照模板回复,不能输出模板以外的内容
    2.如果需要搜索,你需要自己提取搜索关键词,然后按照模板提供搜索关键词, 模板是“需要搜索:关键词1+关键词2+...+关键词n”
    3.请注意,你只有一次搜索机会,请仔细分析问题,准确提取能够帮助你回答的搜索关键词
    4.如果不需要搜索,则直接给出你认为的正确选项, 不要输出任何其他内容,只回复“ABCD”四个选项中的一个

    以下是需要回答的问题和候选项:
    """

    messages.append({"role": "user", "content": round_1_context + round_1_job_text})

    try:
        # Get first round response
        round_1_response = get_response(messages, config, model=model)
        if isinstance(round_1_response, dict) and round_1_response.get('status') == 'error':
            raise Exception(round_1_response.get('error'))

        messages.append({"role": "assistant", "content": round_1_response['message']['content']})

        # Determine if search is needed
        if '需要搜索:' in round_1_response['message']['content']:
            return handle_search_path(messages, round_1_response['message']['content'], round_1_job_text, question,
                                      correct_answer, model)
        else:
            # Direct answer path
            confidence_result = process_model_response(question, round_1_response, correct_answer)
            return format_success_response(search_used=False, **confidence_result)

    except Exception as e:
        return handle_error(e, "active_search")


def handle_search_path(messages, round_1_content, round_1_job_text, question, correct_answer, model):
    """
    Handle the search path when external information is needed
    Processes search requests, retrieves results, and generates final answer
    """
    try:
        # Extract search keywords and perform search
        search_keyword = round_1_content.split('Need to search:')[-1]
        search_result = online_search_detail(search_keyword)

        if isinstance(search_result, dict) and search_result.get('status') == 'error':
            raise Exception(search_result.get('error'))

        search_text = search_result.get('final_text')

        formatted_search_result = f"搜索结果:\n{search_text}\n以上是根据你的搜索关键词搜索到的结果,请根据以上结果回答问题"

        # Construct second round dialogue
        round_2_context = """
        请直接给出你认为的正确选项, 不要输出任何其他内容,只回复“ABCD”四个选项中你确认对的一个

        [任务示例]
        # 示例输入:
        问题: 2024年余杭的房屋均价是多少?
        选项: {"A": "3万元每平米", "B": "2万元每平米", "C": "2.5万元每平米", "D": "1万元每平米"}
        # 示例输出: A
        """
        messages.append({"role": "user", "content": formatted_search_result + round_2_context + round_1_job_text})

        # Get second round response
        round_2_response = get_response(messages, config, model=model)
        confidence_result = process_model_response(question, round_2_response, correct_answer)

        return format_success_response(search_used=True, search_result=formatted_search_result,
                                       search_keyword=search_keyword, **confidence_result)

    except Exception as e:
        return handle_error(e, "search_path")


def process_single_row(row: pd.Series, index: int, thread_progress: ThreadProgressBar, main_progress: tqdm,
                       model: str) -> SearchResult:
    """
    Process a single row of data with progress tracking

    Args:
        row: Single row of DataFrame
        index: Row index
        thread_progress: Progress bar for thread
        main_progress: Main progress bar
        model: Model name to use

    Returns:
        SearchResult object containing processing results or error
    """
    thread_id = threading.get_ident()
    try:
        question = row['question']
        options = row['options']
        correct_answer = row['correct_answer']

        # Call search function
        result = active_search(question=question, options=options, correct_answer=correct_answer, model=model)

        # Update progress bars
        thread_progress.update_progress(thread_id)
        main_progress.update(1)

        return SearchResult(index, model_response=result)

    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")
        thread_progress.update_progress(thread_id)
        main_progress.update(1)
        return SearchResult(index, error=str(e))


def batch_process_questions(df: pd.DataFrame, model: str = 'gpt-4-0803', max_workers: int = 5, retry_times: int = 3,
                            batch_size: Optional[int] = None) -> pd.DataFrame:
    """
    Batch process questions in DataFrame using multiple threads

    Args:
        df: Input DataFrame containing questions
        model: Model name to use
        max_workers: Maximum number of concurrent threads
        retry_times: Number of retry attempts for failed requests
        batch_size: Size of batches to process (None = process all at once)

    Returns:
        Processed DataFrame with results and errors
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

    # Merge results with original DataFrame
    final_df = df.copy()
    final_df['model_response'] = results_df['model_response']
    final_df['error'] = results_df['error']

    return final_df


if __name__ == '__main__':
    import argparse
    from pack.data_utils import filter_bad_ord

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--retry_times", type=int, default=3)
    args = parser.parse_args()
    model = args.model
    max_workers = args.max_workers
    retry_times = args.retry_times

    # Load the config file
    with open(lib_path / "config.json") as f:
        config = json.load(f).get(model)

    data = pd.read_excel(lib_path / 'data' / "data_set.xlsx").iloc[:10]
    result_df = batch_process_questions(data, model=model, max_workers=max_workers, retry_times=retry_times)
    result_df['model_response'] = result_df['model_response'].apply(lambda x: filter_bad_ord(str(x)))
    result_df['error'] = result_df['error'].apply(lambda x: filter_bad_ord(str(x)))
    result_df.to_excel(lib_path / 'data' / f"uncertainty_active_rag_{model}.xlsx", index=False)

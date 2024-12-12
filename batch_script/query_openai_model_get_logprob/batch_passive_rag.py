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

from pack.api_call import online_search_detail

import pandas as pd
from tqdm.auto import tqdm

from pack.api_call import get_response_logprob as get_response
from pack.utils import process_model_response, handle_error, format_success_response, SearchResult, \
    calculate_retry_delay
from pack.multi_thread import ThreadProgressBar

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def passive_search(question, options, correct_answer, search_result, model='gpt-4o-0513'):
    # Initialize messages list
    messages = [
        {"role": "system", "content": "You are a knowledgeable scholar who can provide helpful answers to questions"}]
    # Truncate search result to avoid token limit
    search_result = search_result[:100000]
    if len(search_result) < 10:
        raise Exception(f"Search result is too short: {search_result}")
    # Construct conversation content, keeping the Chinese prompt intact
    job_text = f"""
    问题: {question}
    选项: {options}
    搜索结果: {search_result}
    """

    job_context = """
    我会对你提出一个问题,同时提出对应的选项,你需要根据你的知识和网络搜索的结果,准确回答

    [回复格式要求]
    请直接给出你认为的正确选项, 不要输出任何其他内容,只回复"ABCD"四个选项中你确认对的一个

    [任务示例]
    # 示例输入:
    问题: 2024年余杭的房屋均价是多少?
    选项: {"A": "3万元每平米", "B": "2万元每平米", "C": "2.5万元每平米", "D": "1万元每平米"}
    搜索结果: <标题>：2024年余杭房价再创新高<内容> 记者走访余杭，2024年余杭的房屋均价再创新高，为2.5万元/㎡，较去年同期下跌0.5%。

    # 示例输出: C

    # [任务要求]
    1.仔细学习[任务示例], 你的回复内容必须严格按照模板回复,不能输出模板以外的内容
    4.请直接给出你认为的正确选项, 不要输出任何其他内容,只回复"ABCD"四个选项中你确认对的一个

    以下是需要回答的问题和候选项:
    """

    messages.append({"role": "user", "content": job_context + job_text})

    try:
        # Get first round response from model
        response = get_response(messages, config, model=model)
        if isinstance(response, dict) and response.get('status') == 'error':
            raise Exception(response.get('error'))

        # Process model response and calculate confidence
        confidence_result = process_model_response(question, response, correct_answer)
        return format_success_response(search_used=True, **confidence_result)

    except Exception as e:
        return handle_error(e, "passive_search")


def process_single_row(row: pd.Series, index: int, thread_progress: ThreadProgressBar, main_progress: tqdm,
                       model: str) -> SearchResult:
    """
    Process a single row of data

    Args:
        row: Input data row
        index: Row index
        thread_progress: Thread progress tracker
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

        # Perform online search for the question
        search_result = online_search_detail(question)
        if isinstance(search_result, dict) and search_result.get('status') == 'error':
            raise Exception(search_result.get('error'))

        search_result = search_result.get('final_text')

        # Validate search result
        if len(search_result) < 10:
            raise Exception(f"Search result is too short: {search_result}")
        search_result = search_result.strip()

        # Call LLM function with search results
        result = passive_search(question=question, options=options, correct_answer=correct_answer,
                                search_result=search_result, model=model)

        # Update progress trackers
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
    Batch process questions in DataFrame using parallel execution

    Args:
        df: Input DataFrame containing questions
        model: Model name to use
        max_workers: Maximum number of parallel threads
        retry_times: Number of retry attempts for failed requests
        batch_size: Size of batches to process, None means process all at once

    Returns:
        DataFrame with processing results and any errors
    """
    results = {}
    results_lock = threading.Lock()

    total_rows = len(df)
    if batch_size is None:
        batch_size = total_rows

    # Calculate questions per thread for progress tracking
    questions_per_thread = -(total_rows // -max_workers)

    def process_with_retry(row: pd.Series, index: int, thread_progress: ThreadProgressBar,
                           main_progress: tqdm) -> SearchResult:
        """Helper function to handle retries for failed requests"""
        result = None
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

            # Process data in batches
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i + batch_size]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_row = {
                        executor.submit(process_with_retry, row, idx, thread_progress, main_progress): (idx, row) for
                        idx, row in batch_df.iterrows()}

                    # Collect results as they complete
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

    # Order results and prepare final DataFrame
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

    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--retry_times", type=int, default=3)
    args = parser.parse_args()

    model = args.model
    max_workers = args.max_workers
    retry_times = args.retry_times

    # Load configuration
    with open(lib_path / "config.json") as f:
        config = json.load(f).get(model)

    # Process sample data
    data = pd.read_excel(lib_path / 'data' / "data_set.xlsx").iloc[:10]
    result_df = batch_process_questions(data, model=model, max_workers=max_workers, retry_times=retry_times)

    # Clean and save results
    result_df['model_response'] = result_df['model_response'].apply(lambda x: filter_bad_ord(str(x)))
    result_df['error'] = result_df['error'].apply(lambda x: filter_bad_ord(str(x)))
    result_df.to_excel(lib_path / 'data' / f"uncertainty_passive_rag_{model}.xlsx", index=False)

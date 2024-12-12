import json
import math
from typing import Dict, Any, Optional


def get_confidence(query, ori_answer, true_answer):
    """
    Calculate the confidence score for a given query answer.

    :param query: Query string
    :param ori_answer: Dictionary containing original answer and its probability information
    :param true_answer: True answer string
    :return: Dictionary containing status, message and calculation results
    """

    if not isinstance(ori_answer, dict) or 'logprobs' not in ori_answer:
        """
        Validate if the input original answer is a valid dictionary and contains necessary keys.
        """
        return {"status": "error", "message": f"Invalid ori_answer format for query: {query}"}

    try:
        logprobs_content = ori_answer['logprobs']["content"]
        if not logprobs_content:
            """
            Raise exception because log probability content is empty.
            """
            raise ValueError("Empty logprobs content")

        logprobs = logprobs_content[0]

        # Check required fields
        required_fields = ["token", "logprob", "top_logprobs"]
        if not all(field in logprobs for field in required_fields):
            """
            Raise exception because required log probability fields are missing.
            """
            raise ValueError("Missing required fields in logprobs")

        model_answer = logprobs["token"]
        model_answer_probs = math.exp(logprobs["logprob"])

        # Use next and generator expression to find correct answer probability
        right_logprob_probs = next((math.exp(candidate["logprob"]) for candidate in logprobs["top_logprobs"] if
                                    candidate["token"] == true_answer), None)

        return {"status": "success", "query": query, "true_answer": true_answer,
                "right_logprob_probs": right_logprob_probs, "model_answer": model_answer,
                "model_answer_probs": model_answer_probs, "answer_right": model_answer == true_answer}

    except Exception as e:
        error_msg = f"Error processing query '{query}': {str(e)}"
        """
        Return a dictionary containing error status and detailed error message.
        """
        return {"status": "error", "message": error_msg}


def validate_answer(content):
    """Validate if the answer format is correct"""
    return content.lower().strip() in ['a', 'b', 'c', 'd']


def process_model_response(question, response, correct_answer):
    """Process model response and get confidence score"""
    if not validate_answer(response['message']['content']):
        raise ValueError(f"Invalid answer format: {response['message']['content']}")

    confidence_result = get_confidence(question, response, correct_answer)
    if isinstance(confidence_result, dict) and confidence_result.get('status') == 'error':
        raise Exception(confidence_result.get('error'))

    return confidence_result


def handle_error(error_msg, error_source):
    """Unified error handling"""
    return {"status": "error", "error": f"{error_source} error with {error_msg}"}


def format_success_response(search_used=False, **kwargs):
    """Unified success response format"""
    response = {"status": "success", "search": search_used}
    response.update(kwargs)
    return response


class SearchResult:
    def __init__(self, index: int, model_response: Optional[Dict] = None, error: Optional[str] = None):
        self.index = index
        self.model_response = model_response
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index,
                "model_response": json.dumps(self.model_response, ensure_ascii=False) if self.model_response else None,
                "error": self.error}


def calculate_retry_delay(attempt: int, base_delay: float = 2.0, max_delay: float = 10.0) -> float:
    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay
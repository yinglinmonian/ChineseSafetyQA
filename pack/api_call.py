# -*- coding: utf-8 -*-

from openai import OpenAI


def get_response_logprob(message, config, model):
    """
    Get response with log probabilities from OpenAI API.

    Args:
        message: The input message to send to the API
        config: Configuration dictionary containing API credentials
        model: The OpenAI model to use

    Returns:
        dict: Response containing log probabilities or error details
    """
    try:
        # Initialize OpenAI client with provided credentials
        client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])

        # Make API call with log probability parameters
        completion = client.chat.completions.create(model=model,
                                                    messages=message,
                                                    frequency_penalty=1,
                                                    temperature=1,
                                                    top_p=0.01,
                                                    logprobs=True,
                                                    top_logprobs=5,
                                                    n=1)

        return completion.choices[0].to_dict()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_response_content(message, config, model):
    """
    Get plain response content from OpenAI API without log probabilities.

    Args:
        message: The input message to send to the API
        config: Configuration dictionary containing API credentials
        model: The OpenAI model to use

    Returns:
        str: Response content or error details
    """
    try:
        # Initialize OpenAI client with provided credentials
        client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])

        # Make API call for content only
        completion = client.chat.completions.create(model=model,
                                                    messages=message,
                                                    frequency_penalty=1,
                                                    temperature=1,
                                                    top_p=0.01,
                                                    n=1)

        return completion.choices[0].message.content
    except Exception as e:
        return {"status": "error", "error": str(e)}


def online_search_detail(text):
    """
    Perform an online search and return detailed results.

    Args:
        text (str): The search query text

    Returns:
        dict: A dictionary containing either:
            - On success: {"final_text": str} where final_text contains formatted search results
              Each result should be formatted as: "<Title>: {title} <Content>: {content}\n"
            - On error: {"status": "error", "error": str} with error details

    Requirements:
        1. The function should perform a search using the input text
        2. Results better be ranked by relevance
        3. The total length of final_text better not exceed 30000 characters according to model context length limit
        4. Each result entry should be cleaned of invalid characters
        5. Handle all exceptions and return appropriate error messages
    """
    pass


if __name__ == '__main__':
    # Test the online search function with a sample query
    print(online_search_detail("test query").get('final_text'))
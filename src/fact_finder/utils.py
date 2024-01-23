from typing import List, Dict


def concatenate_with_headers(answers: List[Dict[str, str]]) -> str:
    result = ""
    for answer in answers:
        for header, text in answer.items():
            result += header + "\n" + text + "\n\n"
    return result
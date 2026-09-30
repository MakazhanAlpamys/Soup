import time
import pytest
import re
from soup_cli.utils.errors import format_friendly_error
from soup_cli.utils.errors import _AUTH_STATUS_TEXT


phrasings = ["HTTP 401", "HTTP  401", "HTTP Error 401", "HTTP Error 401: Unauthorized", "HTTP =401", "HTTP = 401", "HTTP: 401", "HTTP Error: 401", "HTTP 4010", "HTTP401", "HTTP Error  =  401", r"HTTP\t401", r"HTTP Error\n401", "Got HTTP 401 from server", "HTTP : 401"]


_AUTH_STATUS_TEXT_OLD = (
    (re.compile(r"HTTP\s+(?:Error\s+)?[:=]?\s*401\b"), 401),
    (re.compile(r"HTTP\s+(?:Error\s+)?[:=]?\s*403\b"), 403),
)


def test_quadratic_backtracking_is_not_present() -> None:
    test_input = ValueError("HTTP" + " " * 50_000)
    start_time = time.perf_counter()
    format_friendly_error(test_input)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    time_bound = 0.5
    assert total_time < time_bound, "Quadratic backtracking present in one or more error messages."

@pytest.mark.parametrize("error_code", ["401", "403"])
@pytest.mark.parametrize("test_input", phrasings)
def test_new_pattern_agrees_with_old_pattern(error_code: str, test_input: str) -> None:
    index_old = 0
    index_current = 1
    if error_code == "403":
        test_input = test_input.replace("401", error_code)
        index_old = 1
        index_current = 4
    old_output = _AUTH_STATUS_TEXT_OLD[index_old][0].search(test_input)
    current_output = _AUTH_STATUS_TEXT[index_current][0].search(test_input)
    assert (old_output is None) == (current_output is None), "Error message patterns do not match."
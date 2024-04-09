import glob
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)


class VerificationFailureItem(BaseModel):
    type: str = "verification_failure"
    test_code: str = Field(default=None, description="Code of the test")
    output: str = Field(description="Output of the verification process")

    test_method: Optional[str] = Field(default=None, description="Test method")
    test_class: Optional[str] = Field(default=None, description="Test class")
    test_file: Optional[str] = Field(default=None, description="Test file")
    test_linenumber: Optional[int] = Field(default=None, description="Test line number")


def get_number_of_tests(output: str) -> (int, int):
    failed_match = re.search(r'(\d+) failed', output)
    passed_match = re.search(r'(\d+) passed', output)
    error_match = re.search(r'(\d+) errors', output)

    if failed_match:
        failed_count = int(failed_match.group(1))
    else:
        failed_count = 0

    if error_match:
        error_count = int(error_match.group(1))
    else:
        error_count = 0

    if passed_match:
        passed_count = int(passed_match.group(1))
    else:
        passed_count = 0

    return failed_count, passed_count, error_count


def find_timed_out(output: str) -> List[VerificationFailureItem]:
    match = re.search(r'(\w+.py)::(\w+)::(\w+)', output)
    if match:
        file_name = match.group(1)
        class_name = match.group(2)
        method_name = match.group(3)
        return [VerificationFailureItem(test_file=file_name, test_class=class_name, test_method=method_name, output="Timed out")]

    match = re.search(r'(\w+.py)\s\.\.\.', output)
    if match:
        file_name = match.group(1)
        return [VerificationFailureItem(test_file=file_name, output="Timed out")]

    return [VerificationFailureItem(output="Timed out")]

def find_failed_tests(output: str) -> List[VerificationFailureItem]:
    sections = output.split("======================================================================")
    failed_tests = [extract_test_details(section) for section in sections]
    return [test for test in failed_tests if test is not None]

def extract_test_details(section: str) -> Optional[VerificationFailureItem]:
    header_pattern = r'(FAIL|ERROR): ([^\s]+) \(([^.]+)\.([^\.]+)\.([^\)]+)\)'
    body_pattern = r'[\s\S]+?.*File "([^"]+)", (.*)'

    file_pattern = re.compile(r'File "([^"]+)", ')

    # test_files = glob.glob(str(self.current_dir) + "/" + self.test_file_pattern, recursive=True)

    splitted = section.split("----------------------------------------------------------------------")
    if len(splitted) >= 2:
        if "unittest.loader" in splitted[0]:
            test_load_fail = True
        else:
            test_load_fail = False

        file_path = None
        #all_matches = file_pattern.findall(splitted[1])
        #for match in all_matches:
        #    if match in test_files:
        #        file_path = match
        #        break

        header_match = re.search(header_pattern, splitted[0], re.DOTALL)
        body_match = re.search(body_pattern, splitted[1], re.DOTALL)
        if header_match:
            if test_load_fail:
                test_result, test_method, _, _, _ = header_match.groups()
                traceback = splitted[1]
                test_class = None
                test_file = None
            else:
                test_result, test_method, module, test_class, _ = header_match.groups()

                if file_path:
                    traceback = splitted[1].split(file_path)[1][2:]
                else:
                    file_path, traceback = body_match.groups()

                #test_file = file_path.replace(str(self.current_dir) + "/", "")

            test_file = None
            #traceback = traceback.replace(str(self.current_dir) + "/", "")

            return VerificationFailureItem(
                test_method=test_method,
                test_class=test_class,
                test_file=test_file,
                output=traceback.strip()
            )

    return None
def find_failed_tests(data: str) -> List[VerificationFailureItem]:
    test_output = ""
    test_file = None
    test_class = None
    test_method = None
    test_code = None
    test_linenumber = None

    failures = []
    extract_output = False
    extract = False
    lines = data.split("\n")
    for line in lines:
        if "= FAILURES =" in line:
            extract = True
            continue

        if extract:
            test_file_search = re.search(r'^(\w+\.py):(\d+)', line)

            if line.startswith("="):
                failures.append(
                    VerificationFailureItem(test_file=test_file,
                                            test_linenumber=test_linenumber,
                                            test_class=test_class,
                                            test_method=test_method,
                                            output=test_output,
                                            test_code=test_code))
                return failures
            elif line.startswith("__"):
                match = re.search(r'(\w+)\.(\w+)', line)
                single_match = re.search(r'_\s(\w+)\s_', line)

                if match or single_match:
                    if test_file:
                        failures.append(
                            VerificationFailureItem(test_file=test_file,
                                                    test_linenumber=test_linenumber,
                                                    test_class=test_class,
                                                    test_method=test_method,
                                                    output=test_output,
                                                    test_code=test_code))
                        test_output = ""
                        test_code = None
                        test_file = None

                    if match:
                        test_class = match.group(1)
                        test_method = match.group(2)
                    else:
                        test_class = None
                        test_method = single_match.group(1)
            elif not test_file and test_file_search:
                test_file = test_file_search.group(1)
                test_linenumber = test_file_search.group(2)
                extract_output = False
            elif line.startswith("E ") or (line.startswith("> ") and "???" not in line):
                test_output += line + "\n"
                extract_output = True
            elif extract_output and not test_file_search:
                test_output += line + "\n"
            elif test_file_search:
                extract_output = False
            elif not line.startswith("self") and line.strip() and not test_file:
                if not test_code:
                    test_code = line
                else:
                    test_code += "\n" + line

    if test_file:
        failures.append(VerificationFailureItem(test_file=test_file,
                                                test_linenumber=test_linenumber,
                                                test_class=test_class,
                                                test_method=test_method,
                                                output=test_output,
                                                test_code=test_code))

    return failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

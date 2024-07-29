from moatless.codeblocks.parser.python import PythonParser


def scikit_learn_10297():
    with open(
        "../data/python/regressions/scikit-learn__scikit-learn-10297/original.py", "r"
    ) as f:
        content = f.read()
    parser = PythonParser(debug=False)

    return parser.parse(content)

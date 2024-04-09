from moatless.code_graph import CodeGraph
from moatless.codeblocks import CodeBlock
from moatless.codeblocks.parser.python import PythonParser


def test_relationships():
    content = """from datetime import datetime
class Event:
    def __init__(self, name):
        self.name = name

class EventLogger:
    def __init__(self):
        self.events = []
    def log_event(self, description):
        timestamp = datetime.now()
        event = f"{timestamp}: {description}"
        self.events.append(event)
        print(f"Event logged: {event}")
        # Calling show_last_event to display the last event logged
        self.show_last_event()
    
    def show_last_event(self):
        if self.events:
            print(f"Last event: {self.events[-1]}")
        else:
            print("No events have been logged.")
"""

    code_graph = CodeGraph()

    file = "event_logger.py"

    def add_to_graph(codeblock: CodeBlock):
        code_graph.add_to_graph(file, codeblock)

    parser = PythonParser(index_callback=add_to_graph)

    codeblock = parser.parse(content)

    print(codeblock.to_tree(include_references=True))

    rels = code_graph.find_relationships(file, ["EventLogger", "log_event"])

    for rel in rels:
        print(rel.full_path())

import argparse
import json
import logging
import os

from moatless.loop import AgenticLoop
from moatless.streamlit.list_visualization import create_linear_table
import streamlit as st
from dotenv import load_dotenv
from moatless.streamlit.investigate_node import investigate_node

from moatless.benchmark.report import generate_report
from moatless.benchmark.utils import get_moatless_instance
from moatless.search_tree import SearchTree
from moatless.streamlit.shared import trajectory_table
from moatless.streamlit.tree_visualization import update_visualization

logger = logging.getLogger(__name__)


def main():
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Add argument parsing at the start of main()
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Path to the file")
    parser.add_argument("--linear", action="store_true", help="Force linear visualization")
    args, unknown = parser.parse_known_args()

    st.set_page_config(
        layout="wide",
        page_title="Moatless Visualizer",
        initial_sidebar_state="collapsed",
    )

    # Store the linear preference in session state
    if "force_linear" not in st.session_state:
        st.session_state.force_linear = args.linear

    container = st.container()

    # Get file path from command line args if provided
    if "path" in st.query_params:
        file_path = st.query_params["path"]
    elif args.path:
        file_path = args.path
        # is directory
        if os.path.isdir(file_path):
            logger.info("Generating report for directory")
            generate_report(file_path)
            file_path = os.path.join(file_path, "report.json")

        st.query_params["path"] = file_path
    else:
        file_path = None

    if not file_path:
        st.sidebar.text("Please provide a file path.")
        file_path = st.text_input("Enter the full path to your JSON file:")
        load_button = st.button("Load File")

        if load_button and file_path:
            st.query_params["path"] = file_path
            st.rerun()
    else:
        st.sidebar.text(f"Loading file: {file_path}")

    if file_path:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                # Generate report if directory is provided
                report_path = os.path.join(file_path, "report.json")

                # Add refresh button in sidebar
                st.sidebar.text(f"Loading file: {file_path}")
                if st.sidebar.button("🔄 Regenerate Report"):
                    with st.spinner("Regenerating report..."):
                        generate_report(file_path)
                    st.rerun()

                # Generate initial report if it doesn't exist
                with st.spinner("Generating report from directory..."):
                    generate_report(file_path)
                file_path = report_path

            file_name = os.path.basename(file_path).lower()

            if file_name == "report.json":
                with st.spinner("Loading report..."):
                    trajectory_table(file_path)
                    report_dir = os.path.dirname(file_path)
                    if st.button("Regenerate Report"):
                        with st.spinner("Regenerating report..."):
                            generate_report(report_dir, split="verified")
                            st.rerun()

            else:
                instance = None
                if not "search_tree" in st.session_state or st.session_state.search_tree.persist_path != file_path:
                    with st.spinner("Loading search tree from trajectory file"):
                        # Need to load twice to get the instance id...
                        with open(file_path, "r") as f:
                            search_tree = json.load(f)
                            if "metadata" in search_tree and "instance_id" in search_tree["metadata"]:
                                instance_id = search_tree["metadata"]["instance_id"]
                            else:
                                instance_id = None

                        if instance_id:
                            instance = get_moatless_instance(instance_id)
                        else:
                            instance = None

                        # repository = create_repository(instance)
                        # code_index = create_index(instance, repository=repository)

                        # try:
                        #    from moatless.runtime.testbed import TestbedEnvironment

                        #    runtime = TestbedEnvironment(
                        #        repository=repository,
                        #        instance=instance
                        #    )
                        # except:
                        #    runtime = None

                        st.session_state.search_tree = AgenticLoop.from_file(
                            file_path,
                            # repository=repository,
                            # runtime=runtime,
                            # code_index=code_index,
                        )
                        # st.session_state.search_tree.agent.set_actions(create_all_actions(repository, code_index, runtime=runtime, completion_model=st.session_state.search_tree.agent.completion))

                if "node_id" in st.query_params:
                    node_id = int(st.query_params["node_id"])
                    investigate_node(st.session_state.search_tree, node_id)
                else:
                    eval_result = None
                    directory_path = os.path.dirname(file_path)
                    eval_path = f"{directory_path}/eval_result.json"
                    if os.path.exists(eval_path):
                        with open(f"{directory_path}/eval_result.json", "r") as f:
                            eval_result = json.load(f)

                    if not instance and search_tree.metadata.get("instance_id"):
                        instance = get_moatless_instance(search_tree.metadata["instance_id"])

                    create_linear_table(
                        st.session_state.search_tree.root.get_all_nodes(),
                        eval_result=eval_result,
                        instance=instance,
                    )

        else:
            st.error("The specified file does not exist. Please check the path and try again.")

    if not file_path:
        st.info("Please provide a valid file path and click 'Load File' to begin.")


if __name__ == "__main__":
    main()

from typing import Optional, List

import streamlit as st

from moatless.node import Node
from moatless.streamlit.shared import show_completion
from testbeds.schema import TestStatus


def create_linear_table(
    nodes: List[Node],
    max_node_id: int,
    eval_result: Optional[dict] = None,
    instance: Optional[dict] = None,
) -> None:
    """Create a table visualization for linear trajectories."""
    # Add timeline visualization at the top

    # Show instance summary first
    if instance or eval_result:
        left_col, right_col = st.columns([1, 1])

        with left_col:
            summary_tabs = st.tabs(["Summary", "Patch", "Test Details"])

            with summary_tabs[0]:
                if eval_result and eval_result.get("error"):
                    st.error("🛑 Evaluation Error")
                    st.code(eval_result["error"])
                elif eval_result and eval_result.get("node_results"):
                    final_node = max(map(int, eval_result["node_results"].keys()))
                    final_result = eval_result["node_results"][str(final_node)]
                    if final_result.get("resolved") is True:
                        st.success("✅ Instance Resolved")
                    elif final_result.get("resolved") is False:
                        st.error("❌ Instance Not Resolved")

                # Enhanced summary section
                if nodes[-1].file_context:
                    # Files in context summary
                    st.markdown("#### Files in Context")
                    # if nodes[-1].file_context:
                    #    st.markdown(nodes[-1].file_context.create_summary())

                    # Test results summary
                    st.markdown("#### Test Results")
                    test_summary = nodes[-1].file_context.get_test_summary()
                    st.markdown(test_summary)

                    # Test metrics visualization
                    passed = int(test_summary.split("passed")[0].strip())
                    failed = int(test_summary.split("failed")[0].split(".")[-1].strip())
                    errors = int(test_summary.split("errors")[0].split(".")[-1].strip())

                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("✅ Passed", passed)
                    with metrics_cols[1]:
                        if failed > 0:
                            st.metric(
                                "❌ Failed", failed, delta=failed, delta_color="inverse"
                            )
                        else:
                            st.metric("✅ Failed", failed)
                    with metrics_cols[2]:
                        if errors > 0:
                            st.metric(
                                "⚠️ Errors", errors, delta=errors, delta_color="inverse"
                            )
                        else:
                            st.metric("✅ Errors", errors)

            with summary_tabs[1]:
                if nodes[-1].file_context and nodes[-1].file_context.has_patch():
                    st.code(nodes[-1].file_context.generate_git_patch())
                else:
                    st.info("No patch available yet")

            # Create new Test Details tab
            with summary_tabs[2]:
                if nodes[-1].file_context:
                    failure_details = nodes[-1].file_context.get_test_failure_details()
                    if failure_details:
                        st.markdown(failure_details)
                    else:
                        st.info("No test failures to display")

        with right_col:
            if instance:
                instance_tabs = st.tabs(["Description", "Instance", "Golden patch"])

                with instance_tabs[0]:
                    if "description" in instance:
                        st.markdown("**Description:**")
                        st.markdown(instance["description"])
                    if "expected_changes" in instance:
                        st.markdown("**Expected Changes:**")
                        st.code(instance["expected_changes"])

                with instance_tabs[1]:
                    st.json(instance)

                with instance_tabs[2]:
                    if "golden_patch" in instance:
                        st.code(instance["golden_patch"])
                    else:
                        st.info("No golden patch available")

    st.markdown("---")

    st.markdown("### Timeline")

    # Calculate number of nodes to show (excluding root node)
    visible_nodes = [n for n in nodes[1:] if n.node_id <= max_node_id]
    nodes_per_row = 8
    num_rows = (
        len(visible_nodes) + nodes_per_row - 1
    ) // nodes_per_row  # Ceiling division

    for row in range(num_rows):
        start_idx = row * nodes_per_row
        end_idx = min(start_idx + nodes_per_row, len(visible_nodes))
        row_nodes = visible_nodes[start_idx:end_idx]

        timeline_cols = st.columns(nodes_per_row)
        for idx, (node, col) in enumerate(zip(row_nodes, timeline_cols)):
            if node.node_id > max_node_id:
                continue

            with col:
                # Determine node status/color
                color = "gray"
                test_status = ""
                diff_stats = ""
                fail_reason = ""

                if node.observation:
                    if node.observation.properties.get("fail_reason"):
                        color = "red"
                        fail_reason = node.observation.properties.get("fail_reason", "")

                    if node.file_context and node.file_context.test_files:
                        test_status = node.file_context.get_test_status()
                        if test_status:
                            if test_status == TestStatus.ERROR:
                                color = "red"
                                test_status = "⚠️ Error"
                            elif test_status == TestStatus.FAILED:
                                color = "yellow"
                                test_status = "❌ Failed"
                            else:
                                color = "green"
                                test_status = "✅ Passed"

                    if node.observation.properties.get("diff"):
                        diff_lines = node.observation.properties["diff"].split("\n")
                        additions = sum(
                            1
                            for line in diff_lines
                            if line.startswith("+") and not line.startswith("+++")
                        )
                        deletions = sum(
                            1
                            for line in diff_lines
                            if line.startswith("-") and not line.startswith("---")
                        )
                        if additions or deletions:
                            diff_stats = f"+{additions}/-{deletions}"

                # Combine test status and diff stats on same line if both exist
                status_line = test_status
                if diff_stats:
                    status_line = (
                        f"{test_status} {diff_stats}" if test_status else diff_stats
                    )
                if fail_reason:
                    status_line = fail_reason
                    color = "red"

                # Create a styled div for the node
                if node.observation:
                    fail_reason = node.observation.properties.get("fail_reason", "")
                else:
                    fail_reason = "no_observation"

                st.markdown(
                    f"""
                    <div style="
                        padding: 8px;
                        border: 2px solid {color};
                        border-radius: 4px;
                        text-align: center;
                        font-size: 0.9em;
                    ">
                        <div style="font-weight: bold;">Node {node.node_id}</div>
                        <div style="font-size: 0.8em;">{node.action.name if node.action else 'Start'}</div>
                        <div style="font-size: 0.8em;">{status_line}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Add a small vertical space between rows
        if row < num_rows - 1:
            st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Create columns for the table with adjusted widths
    cols = st.columns([1, 3, 3, 2])
    headers = ["Node", "Action", "Observation", "Context"]
    for col, header in zip(cols, headers):
        col.write(f"**{header}**")

    # Add rows for each node
    for node in nodes[1:]:
        if node.node_id > max_node_id:
            continue

        cols = st.columns([1, 3, 3, 2])

        # Node column
        node_str = f"Node{node.node_id}"
        if node.action:
            node_str += f" ({node.action.name})"
        cols[0].subheader(node_str)

        # Add token usage info if available
        if node.completions:
            usage_rows = []
            has_cost = False

            for completion_type, completion in node.completions.items():
                if completion and completion.usage:
                    usage = completion.usage
                    tokens = []
                    if usage.prompt_tokens:
                        tokens.append(f"{usage.prompt_tokens}↑")
                    if usage.completion_tokens:
                        tokens.append(f"{usage.completion_tokens}↓")
                    if usage.cached_tokens:
                        tokens.append(f"{usage.cached_tokens}⚡")

                    if usage.completion_cost:
                        has_cost = True
                        cost = f"${usage.completion_cost:.4f}"
                        usage_rows.append(
                            f"|{completion_type}|{cost}|{' '.join(tokens)}|"
                        )
                    elif tokens:
                        usage_rows.append(f"|{completion_type}|{' '.join(tokens)}|")

            if usage_rows:
                if has_cost:
                    header = "|Type|Cost|Tokens|"
                    separator = "|:--|--:|:--|"
                else:
                    header = "|Type|Tokens|"
                    separator = "|:--|:--|"
                table = "\n".join([header, separator] + usage_rows)
                cols[0].markdown(table, help="↑:prompt ↓:completion ⚡:cached")
                cols[0].markdown("---")

        # Action column with tabs
        if node.action_steps or node.assistant_message:
            tab_names = ["Action", "Completion"]

            action_tabs = cols[1].tabs(tab_names)

            # Input tab
            with action_tabs[0]:
                if node.assistant_message:
                    st.markdown(node.assistant_message)

                for action_step in node.action_steps:
                    if hasattr(action_step.action, "old_str"):
                        st.markdown(f"**File path:** `{action_step.action.path}`")
                        st.markdown("**Old string:**")
                        st.code(action_step.action.old_str)
                        st.markdown("**New string:**")
                        st.code(action_step.action.new_str)
                    elif hasattr(node.action, "file_text"):
                        st.write(f"File path: {action_step.action.path}")
                        st.markdown("**File text:**")
                        st.code(action_step.action.file_text)
                    else:
                        st.json(action_step.action.model_dump(exclude_none=True))

            # Build tab
            with action_tabs[1]:
                if node.completions and node.completions.get("build_action"):
                    show_completion(node.completions["build_action"])
                else:
                    st.info("No build completion available")

        # Observation column with tabs
        if node.observation:
            tabs = cols[2].tabs(["Observation", "Message", "JSON"])

            # Properties tab (shown by default)
            with tabs[0]:
                if node.observation:
                    if "diff" in node.observation.properties:
                        st.markdown("**Diff:**")
                        st.code(node.observation.properties["diff"])

                    if "new_span_ids" in node.observation.properties:
                        st.markdown("**New span IDs:**")
                        for span_id in node.observation.properties["new_span_ids"]:
                            st.markdown(f"- `{span_id}`")

                    if "fail_reason" in node.observation.properties:
                        st.error(f"🛑 {node.observation.properties['fail_reason']}")

                    if "flags" in node.observation.properties:
                        st.warning(
                            f"⚠️ {', '.join(node.observation.properties['flags'])}"
                        )

                    if "test_results" in node.observation.properties:
                        test_results = node.observation.properties["test_results"]
                        total_tests = len(test_results)
                        failed_test_count = sum(
                            1
                            for test in test_results
                            if test["status"] in ["FAILED", "ERROR"]
                        )

                        if failed_test_count > 0:
                            st.warning(
                                f"⚠️ {failed_test_count} out of {total_tests} tests failed"
                            )
                        else:
                            st.success(f"✅ All {total_tests} tests passed")

                    if node.observation.summary:
                        st.code(node.observation.summary)

            # Message tab
            with tabs[1]:
                st.code(node.observation.message)

            # JSON tab
            with tabs[2]:
                for action_step in node.action_steps:
                    st.json(action_step.model_dump(), expanded=False)

        # Context column with tabs
        if node.file_context:
            context_col = cols[3]

            # Create tabs for context
            tab_names = ["Summary", "Tests"]
            if node.file_context.has_patch():
                tab_names.append("Patch")
            tab_names.append("JSON")

            context_tabs = context_col.tabs(tab_names)

            # Summary tab
            with context_tabs[0]:
                # Show context summary
                st.markdown(node.file_context.create_summary())

                # Show test metrics
                test_summary = node.file_context.get_test_summary()
                # Parse numbers from format like "0 passed. 1 failed. 0 errors."
                passed = int(test_summary.split("passed")[0].strip())
                failed = int(test_summary.split("failed")[0].split(".")[-1].strip())
                errors = int(test_summary.split("errors")[0].split(".")[-1].strip())
                total = passed + failed + errors

                if total > 0:
                    # Create visual test summary
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("✅ Passed", passed)
                    with cols[1]:
                        if failed > 0:
                            st.metric(
                                "❌ Failed", failed, delta=failed, delta_color="inverse"
                            )
                        else:
                            st.metric("✅ Failed", failed)
                    with cols[2]:
                        if errors > 0:
                            st.metric(
                                "⚠️ Errors", errors, delta=errors, delta_color="inverse"
                            )
                        else:
                            st.metric("✅ Errors", errors)

            # Tests tab with per-file results
            with context_tabs[1]:
                if node.file_context.test_files:
                    for test_file in node.file_context.test_files:
                        st.markdown(f"#### {test_file.file_path}")

                        # Count results for this file
                        file_passed = sum(
                            1
                            for r in test_file.test_results
                            if r.status == TestStatus.PASSED
                        )
                        file_failed = sum(
                            1
                            for r in test_file.test_results
                            if r.status == TestStatus.FAILED
                        )
                        file_errors = sum(
                            1
                            for r in test_file.test_results
                            if r.status == TestStatus.ERROR
                        )

                        # Show file metrics
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Passed", file_passed)
                        with cols[1]:
                            st.metric("Failed", file_failed)
                        with cols[2]:
                            st.metric("Errors", file_errors)

                        # Show failure details for this file if any
                        failures = [
                            r
                            for r in test_file.test_results
                            if r.status in [TestStatus.FAILED, TestStatus.ERROR]
                            and r.message
                        ]
                        if failures:
                            for result in failures:
                                error_type = (
                                    "❌ Failed"
                                    if result.status == TestStatus.FAILED
                                    else "⚠️ Error"
                                )
                                location = f"line {result.line}" if result.line else ""
                                if result.span_id:
                                    location = f"{location} {result.span_id}".strip()

                                st.markdown(f"**{error_type}** {location}")
                                st.code(result.message)
                else:
                    st.info("No test files in context")

            # Patch tab
            if node.file_context.has_patch():
                with context_tabs[2]:
                    st.code(node.file_context.generate_git_patch())

            # Add new JSON tab at the end
            with context_tabs[-1]:
                st.json(
                    node.file_context.model_dump(exclude_none=False), expanded=False
                )

        # Add a separator between rows
        st.markdown("---")

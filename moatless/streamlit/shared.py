import json
import logging
import os

import altair as alt
import pandas as pd
import streamlit as st

from moatless.benchmark.report import read_reports, to_dataframe
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


def generate_summary(df: pd.DataFrame):
    total_trajectories = len(df)
    status_counts = df["success_status"].value_counts()

    total_cost = df["total_cost"].sum()
    total_prompt_tokens = df["prompt_tokens"].sum()
    total_completion_tokens = df["completion_tokens"].sum()
    total_cached_tokens = df["cached_tokens"].sum()
    avg_cost = df["total_cost"].mean()

    # Filter out 'running', 'error', and 'rejected' statuses for avg_duration calculation
    filtered_df = df[~df["status"].isin(["running", "error", "rejected"])]
    avg_duration = filtered_df["duration"].mean()

    col_metrics, col_chart = st.columns([3, 2])

    with col_metrics:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Trajectories", total_trajectories)
            st.metric("Total Cost", f"${total_cost:.2f}")
            st.metric("Total Prompt Tokens", total_prompt_tokens)
            st.metric("Total Completion Tokens", total_completion_tokens)
            st.metric("Total Cached Tokens", total_cached_tokens)
        with metric_col2:
            st.metric("Avg Cost per Trajectory", f"${avg_cost:.2f}")
            st.metric(
                "Avg Duration (excl. running/error/rejected)", f"{avg_duration:.2f} s"
            )

    with col_chart:
        status_data = pd.DataFrame(
            {"Status": status_counts.index, "Count": status_counts.values}
        )

        status_data["Status_with_Count"] = status_data.apply(
            lambda row: f"{row['Status']} ({row['Count']})", axis=1
        )

        pie_chart = (
            alt.Chart(status_data)
            .mark_arc()
            .encode(
                theta="Count",
                color=alt.Color(
                    "Status_with_Count",
                    scale=alt.Scale(
                        domain=[
                            f'Resolved ({status_counts.get("Resolved", 0)})',
                            f'Finished ({status_counts.get("Finished", 0)})',
                            f'Partially Resolved ({status_counts.get("Partially Resolved", 0)})',
                            f'Running ({status_counts.get("Running", 0)})',
                            f'Failed ({status_counts.get("Failed", 0)})',
                            f'Rejected ({status_counts.get("Rejected", 0)})',
                            f'Error ({status_counts.get("Error", 0)})',
                        ],
                        range=[
                            "#4CAF50",
                            "#808080",
                            "#FFC107",
                            "#2196F3",
                            "#F44336",
                            "#A52A2A",
                            "#9C27B0",
                        ],
                    ),
                ),
                tooltip=["Status", "Count"],
            )
            .properties(title="Trajectory Status Distribution", width=300, height=300)
        )

        # Add text labels to the pie chart
        text = pie_chart.mark_text(radiusOffset=20).encode(
            text=alt.Text("Count:Q", format=".0f"), color=alt.value("white")
        )

        # Combine the pie chart and text labels
        final_chart = (pie_chart + text).interactive()

        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)


def trajectory_table(report_path: str):
    st.header("Trajectory List")

    if not os.path.exists(report_path):
        logger.error(f"Report file not found at {report_path}")
        return

    results = read_reports(report_path)
    logger.info(f"Loaded {len(results)} trajectory reports")
    df = to_dataframe(results)

    # Extract all unique flags from the results
    all_flags = set()
    for result in results:
        if result.flags:  # Check if flags exists and is not empty
            all_flags.update(result.flags)

    # Extract all unique actions from the results
    all_actions = set()
    for result in results:
        # Debug print to see the result structure
        if hasattr(result, "actions") and result.actions:  # Check if actions exists
            all_actions.update(result.actions.keys())

    df["success_status"] = df.apply(
        lambda row: "Resolved"
        if row["resolved"] is not None and row["resolved"]
        else "Partially Resolved"
        if row["resolved_solutions"] > 0
        else "Failed"
        if row["failed_solutions"] > 0
        else "Rejected"
        if row["status"] == "rejected"
        else "Running"
        if row["status"] == "running"
        else "Finished"
        if row["status"] == "finished"
        else "Error"
        if row["status"] == "error"
        else "Unknown",
        axis=1,
    )

    # Format llmonkeys_rate as a percentage
    df["llmonkeys_rate"] = df["llmonkeys_rate"].apply(lambda x: f"{x * 100:.2f}%")

    # Display summary
    generate_summary(df)

    # Add filtering options
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        status_filter = st.multiselect(
            "Status", df["status"].unique(), key="status_filter"
        )
    with col2:
        solution_status_filter = st.multiselect(
            "Solution Status",
            ["Has Resolved Solutions", "Has Unresolved Solutions", "No Solutions"],
            key="solution_status_filter",
        )
    with col3:
        flag_filter = st.multiselect(
            "Flags", sorted(list(all_flags)), key="flag_filter"
        )
    with col4:
        if int(df["resolved_by"].min()) < int(df["resolved_by"].max()):
            resolved_by_range = st.slider(
                "Resolved By",
                int(df["resolved_by"].min()),
                int(df["resolved_by"].max()),
                (int(df["resolved_by"].min()), int(df["resolved_by"].max())),
                key=f"resolved_by_slider",
            )
    with col5:
        action_filter = st.multiselect(
            "Actions", sorted(list(all_actions)), key="action_filter"
        )

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if status_filter:
        mask &= df["status"].isin(status_filter)
    if solution_status_filter:
        if "Has Resolved Solutions" in solution_status_filter:
            mask &= df["resolved_solutions"] > 0
        if "Has Unresolved Solutions" in solution_status_filter:
            mask &= df["failed_solutions"] > 0
        if "No Solutions" in solution_status_filter:
            mask &= (df["resolved_solutions"] == 0) & (df["failed_solutions"] == 0)
    if flag_filter:
        mask &= df.apply(
            lambda row: any(flag in row.get("flags", []) for flag in flag_filter),
            axis=1,
        )
    if action_filter:
        mask &= df.apply(
            lambda row: row.get("actions", {})
            and any(
                action in row["actions"] and row["actions"][action] > 0
                for action in action_filter
            ),
            axis=1,
        )

    if int(df["resolved_by"].min()) < int(df["resolved_by"].max()):
        mask &= df["resolved_by"].between(resolved_by_range[0], resolved_by_range[1])

    filtered_df = df[mask]

    directory_path = os.path.dirname(report_path)
    # Create a column with clickable links using trajectory_path
    filtered_df["Select"] = filtered_df.apply(
        lambda row: f'<a href="?path={directory_path}/{row["instance_id"]}/trajectory.json">View</a>',
        axis=1,
    )

    # Create a new column for fail issues that combines icons for different issues
    def create_fail_issues_column(row):
        # Get number of flags
        num_flags = len(row.get("flags", []))

        if num_flags == 0:
            return ""

        # Create tooltip with just flags
        tooltip = f"Flags: {', '.join(row.get('flags', []))}"

        # Only show number of flags in the column
        return f'<span title="{tooltip}">{num_flags}</span>'

    # Apply the function to create the Issues column
    filtered_df["Issues"] = filtered_df.apply(create_fail_issues_column, axis=1)

    # Update display columns to include duplicated_actions
    display_columns = [
        "Select",
        "instance_id",
        "Issues",
        "resolved_by",
        "llmonkeys_rate",
        "success_status",
        "solutions",
        "resolved_solutions",
        "failed_solutions",
        "resolved_max_reward",
        "failed_max_reward",
        "status",
        "all_transitions",
        "failed_actions",
        "duplicated_actions",  # Add duplicated_actions column
        "total_cost",
        "prompt_tokens",
        "completion_tokens",
        "max_build_tokens",
    ]

    filtered_df = filtered_df.sort_values(by="instance_id")

    # Function to apply color to rows based on success_status
    def color_rows(row):
        base_colors = {
            "Rejected": "rgba(165, 42, 42, 0.3)",  # Brown
            "Resolved": "rgba(76, 175, 80, 0.3)",
            "Finished": "rgba(128, 128, 128, 0.3)",
            "Partially Resolved": "rgba(255, 235, 59, 0.3)",
            "Running": "rgba(33, 150, 243, 0.3)",
            "Error": "rgba(156, 39, 176, 0.3)",  # Purple
            "Failed": "rgba(244, 67, 54, 0.3)",
        }

        color = base_colors.get(row["success_status"], "rgba(244, 67, 54, 0.3)")
        styles = [f"background-color: {color}"] * len(row)

        # Add error tooltip to status column if status is Error
        if row["success_status"] == "Error" and row.get("error"):
            # Find index of status column
            status_idx = display_columns.index("status")
            # Escape error message
            error = (
                row["error"]
                .replace('"', "&quot;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            styles[status_idx] = (
                f"background-color: {color}; cursor: help; title: '<pre>{error}</pre>'"
            )

        return styles

    # Apply the styling
    styled_df = filtered_df[display_columns].style.apply(color_rows, axis=1)

    # Display the filtered and styled dataframe
    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Determine if dark mode is active
    is_dark_mode = st.get_option("theme.base") == "dark"

    # Add CSS to style the table
    st.markdown(
        f"""
    <style>
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th, td {{
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid {'#444' if is_dark_mode else '#ddd'};
    }}
    tr:hover {{background-color: {'rgba(255,255,255,0.1)' if is_dark_mode else 'rgba(0,0,0,0.05)'}; }}
    a {{
        color: {'#58a6ff' if is_dark_mode else '#0366d6'};
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    span[title] {{
        position: relative;
    }}
    span[title]:hover::after {{
        content: attr(title);
        position: absolute;
        left: 0;
        top: 100%;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 14px;
        white-space: pre;
        z-index: 1000;
        font-family: monospace;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def show_completion(completion):
    if completion:
        st.json(
            {
                "model": completion.model,
                "usage": completion.usage.model_dump() if completion.usage else None,
            }
        )
        if completion.input:
            st.subheader("Input prompts")
            for input_idx, input_msg in enumerate(completion.input):
                try:
                    if "content" in input_msg:
                        if isinstance(input_msg["content"], str):
                            content = input_msg["content"]
                        elif (
                            isinstance(input_msg["content"], list)
                            and input_msg["role"] == "user"
                        ):
                            content_list = [
                                c.get("content") or c.get("text")
                                for c in input_msg["content"]
                            ]

                            content = "\n\n".join(content_list)
                        else:
                            content = json.dumps(input_msg["content"], indent=2)

                        tokens = count_tokens(content)
                        with st.expander(
                            f"Message {input_idx + 1} by {input_msg['role']} ({tokens} tokens)",
                            expanded=(input_idx == len(completion.input) - 1),
                        ):
                            st.code(content, language="")

                            if "tool_calls" in input_msg:
                                st.json(input_msg["tool_calls"])
                    else:
                        with st.expander(
                            f"Message {input_idx + 1} by {input_msg['role']}",
                            expanded=(input_idx == len(completion.input) - 1),
                        ):
                            st.json(input_msg)
                except Exception as e:
                    logger.exception(
                        f"Failed to parse {json.dumps(input_msg, indent=2)}"
                    )

        if completion.response:
            st.subheader("Completion response")
            st.json(completion.response)

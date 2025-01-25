import streamlit as st

from moatless.node import Node
from moatless.search_tree import SearchTree
from moatless.streamlit.shared import show_completion



def investigate_node(search_tree: SearchTree, node_id: int):
    # Initialize session state for the node if not exists
    if f"node_{node_id}" not in st.session_state:
        st.session_state[f"node_{node_id}"] = search_tree.get_node_by_id(node_id)

    # Get current node from session state
    current_node = st.session_state[f"node_{node_id}"]
    
    cols = st.columns([1, 3, 3, 2])
    
    # Node column
    node_str = f"Node{current_node.node_id}"
    if current_node.action:
        node_str += f" ({current_node.action.name})"

    cols[0].subheader(node_str)
    
    # Add re-execute button
    if current_node.action:
        if cols[0].button("🔄 Re-execute"):
            with st.spinner("Re-executing action..."):
                try:
                    # Reset node state
                    current_node.observation = None
                    current_node.file_context = current_node.parent.file_context.clone()
                    # Execute action and update node state
                    try:
                        current_node.observation = search_tree.agent._execute(current_node)
                    except Exception as e:
                        st.error(f"Error during execution: {str(e)}")
                    st.session_state[f"node_{node_id}"] = current_node
                    st.success("Action re-executed successfully")
                except Exception as e:
                    st.error(f"Error during execution: {str(e)}")

    # Action column with tabs
    if current_node.action:
        tab_names = ["Input", "Build"]
        if hasattr(current_node.action, "thoughts") and current_node.action.thoughts:
            tab_names.append("Thoughts")
        
        action_tabs = cols[1].tabs(tab_names)
        
        # Input tab
        with action_tabs[0]:
            st.json(current_node.action.model_dump(exclude={"thoughts"}))
        
        # Build tab
        with action_tabs[1]:
            if current_node.completions and current_node.completions.get("build_action"):
                show_completion(current_node.completions["build_action"])
            else:
                st.info("No build completion available")
        
        # Thoughts tab
        if "Thoughts" in tab_names:
            with action_tabs[2]:
                st.markdown(current_node.action.thoughts)
    
    # Observation column with tabs
    if current_node.observation:
        tabs = cols[2].tabs(["Observation", "Message"])
        
        # Properties tab (shown by default)
        with tabs[0]:        
            if current_node.observation and current_node.observation.properties:
                if "fail_reason" in current_node.observation.properties:
                    st.error(f"🛑 {current_node.observation.properties['fail_reason']}")
                
                if "test_results" in current_node.observation.properties:
                    test_results = current_node.observation.properties["test_results"]
                    total_tests = len(test_results)
                    failed_test_count = sum(
                        1 for test in test_results if test["status"] in ["FAILED", "ERROR"]
                    )
                    
                    if failed_test_count > 0:
                        st.warning(f"⚠️ {failed_test_count} out of {total_tests} tests failed")
                    else:
                        st.success(f"✅ All {total_tests} tests passed")

                st.write("Full JSON:")
                st.json(current_node.observation.model_dump(), expanded=False)

        # Message tab
        with tabs[1]:
            st.code(current_node.observation.message)
    
    # Context column with tabs
    if current_node.file_context:
        context_col = cols[3]
        
        # Create tabs for context
        tab_names = ["Summary"]
        if current_node.file_context.has_patch():
            tab_names.append("Patch")
        tab_names.extend([file.file_path for file in current_node.file_context.files])
        context_tabs = context_col.tabs(tab_names)
        
        # Summary tab
        with context_tabs[0]:
            # Show context summary
            st.markdown(current_node.file_context.create_summary())
        
        # Patch tab
        if current_node.file_context.has_patch():
            with context_tabs[1]:
                st.code(current_node.file_context.generate_git_patch())
            tab_offset = 2
        else:
            tab_offset = 1
        
        # File tabs
        for i, file in enumerate(current_node.file_context.files):
            with context_tabs[i + tab_offset]:
                view_tabs = st.tabs(["Processed View", "Raw View"])
                
                with view_tabs[0]:
                    st.code(
                        file.to_prompt(
                            show_span_ids=False,
                            show_line_numbers=True,
                            show_outcommented_code=True
                        ),
                        language="python"
                    )
                
                with view_tabs[1]:
                    st.code(file.content, language="python")
    

from smolagents.tools import tool
# from smolagents.memory import ActionStep


# def show_script_after_mace_call(memory_step, agent):
#     """
#     Step callback that checks if the MACE_script tool was just called
#     and, if so, displays the script that was generated.
#     """
#     if isinstance(memory_step, ActionStep) and memory_step.tool_calls:
#         for tool_call in memory_step.tool_calls:
#             if (
#                 tool_call.name == "python_interpreter"
#                 and "MACE_script(" in tool_call.arguments
#             ):
#                 print("\n" + "=" * 60)
#                 print("Generated script:")
#                 print("=" * 60)

#                 action_result = memory_step.action_output

#                 # Check if the result is a dictionary with the expected key
#                 if (
#                     isinstance(action_result, dict)
#                     and "script_content" in action_result
#                 ):
#                     script_content = action_result["script_content"]
#                     print(script_content)
#                 else:
#                     # Fallback for unexpected format
#                     print(
#                         "Callback could not find 'script_content' in the tool output. Full output:"
#                     )
#                     print(action_result)

#                 print("=" * 60)
#                 break


@tool
def show_content(script: any) -> str:
    """
    Return the given script content for display to the user.

    Args:
        script (any): Full text of a generated Slurm script.
    Returns:
        str: The same script_content for the agent to show.
    """
    if isinstance(script, dict):
        script_content = script.get("script_content", "")
        return script_content
    else:
        script_content = str(script)

    print("\n" + "=" * 60)
    print("Generated script:")
    print("=" * 60)
    print(script_content)
    return script_content

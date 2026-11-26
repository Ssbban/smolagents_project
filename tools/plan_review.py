from smolagents import LogLevel


class UserCancelled(BaseException):
    """The person cancelled at the plan review.

    A BaseException, so that neither the agent loop nor generated code catches it: it leaves
    `manager.run()` before any action step runs, and the command-line loop ends the turn.
    """


def display_plan(plan_content):
    """Display the plan in a formatted way."""
    print("\n" + "=" * 60)
    print("AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice():
    """Get user's choice for plan approval; end of input counts as Cancel."""
    while True:
        try:
            choice = input(
                "\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3): "
            ).strip()
        except EOFError:
            return 3
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_modified_plan(original_plan):
    """Allow user to modify the plan."""
    print("\n" + "-" * 40)
    print("MODIFY PLAN")
    print("-" * 40)
    print("Current plan:")
    print(original_plan)
    print("-" * 40)
    print("Enter your modified plan (press Enter twice to finish):")

    lines = []
    empty_line_count = 0

    while empty_line_count < 2:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            empty_line_count += 1
        else:
            empty_line_count = 0
        lines.append(line)

    modified_plan = "\n".join(lines).strip()
    return modified_plan or original_plan


def interrupt_after_plan(memory_step, agent):
    """
    Show each plan to the person, who approves, modifies or cancels it. Cancel raises UserCancelled.
    """
    if agent.logger.level < LogLevel.INFO:  # from INFO on, smolagents has just printed the plan itself
        display_plan(memory_step.plan)

    choice = get_user_choice()

    if choice == 1:
        print("Plan approved. Continuing execution...")
        return

    if choice == 2:
        modified_plan = get_modified_plan(memory_step.plan)
        memory_step.plan = modified_plan

        print("\nPlan updated.")
        display_plan(modified_plan)
        print("Continuing with modified plan...")
        return

    print("Execution cancelled by user.")
    raise UserCancelled("cancelled at the plan review")

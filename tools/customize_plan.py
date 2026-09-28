from smolagents import PlanningStep


def display_plan(plan_content):
    """Display the plan in a formatted way."""
    print("\n" + "=" * 60)
    print("AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice():
    """Get user's choice for plan approval."""
    while True:
        choice = input(
            "\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3): "
        ).strip()
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
        line = input()
        if line.strip() == "":
            empty_line_count += 1
        else:
            empty_line_count = 0
        lines.append(line)

    modified_plan = "\n".join(lines[:-2])
    return modified_plan if modified_plan.strip() else original_plan


def interrupt_after_plan(memory_step, agent):
    """
    Interrupt after a planning step so the user can approve, modify, or cancel it.
    """
    if not isinstance(memory_step, PlanningStep):
        return

    print("\nAgent interrupted after plan creation.")
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
    agent.interrupt()

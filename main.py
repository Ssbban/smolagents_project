"""Command-line session: each request is one turn of the manager agent."""

import contextlib

import smolagents

from agents import build_agents, build_model
from config import Settings, SettingsError, shadowed_settings
from tools.plan_review import UserCancelled
from tools.session import Session, start_session

with contextlib.suppress(ImportError):  # not on Windows
    import readline  # noqa: F401  (line editing and history for every input() prompt)

HISTORY_TURNS = 3
ANSWER_CHARS = 1000  # of each earlier answer passed on in "history"


def run_turn(manager, session: Session, request: str, history: list[dict]) -> str:
    """Run one request; print what happened and return it as one line for the history."""
    mark = session.mark()
    session.trace.emit("turn_start", request=request)
    answer = None
    try:
        result = manager.run(
            request, additional_args={"session": session.snapshot(), "history": history}, return_full_result=True
        )
        answer = result.output
        outcome = "answered" if result.state == "success" else "stopped at the step limit (fallback answer)"
    except UserCancelled:
        outcome = "cancelled at the plan review"
    except KeyboardInterrupt:
        outcome = "interrupted (Ctrl+C)"
    except Exception as error:  # AgentError, or a provider error in the planning call (smolagents does not wrap it)
        outcome = f"failed: {type(error).__name__}: {error}"
    produced = session.since(mark)
    session.trace.emit("turn_end", outcome=outcome, answer=answer, **produced)

    print(f"\nOutcome: {outcome}")
    if answer is not None:
        print(f"Agent> {answer}")
    for artifact in produced["artifacts"]:
        print(f"  file written: {artifact['path']} ({artifact['kind']})")
    for job in produced["jobs"]:
        print(f"  job submitted: {job['job_id']} from {job['script']}")
    return outcome if answer is None else f"{outcome}: {str(answer)[:ANSWER_CHARS]}"


def main():
    try:
        settings = Settings.from_env()
    except SettingsError as error:
        raise SystemExit(f"Error: {error}") from None
    for name in shadowed_settings():
        print(f"Warning: {name} is set in the environment, which overrides the value in .env")
    session = start_session(settings.runs_dir)
    session.trace.emit("session_start", settings=settings.public(), smolagents=smolagents.__version__)
    manager = build_agents(build_model(settings), settings)
    print(f"Session {session.id}: generated files and the trace are in {session.dir.as_posix()}/")
    print("Agent: Hello! I can help with MACE training and i-PI simulations. Type 'exit' to quit.")

    history: list[dict] = []
    while True:
        try:
            request = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not request:
            continue
        if request.lower() in {"exit", "quit", "q"}:
            break
        outcome = run_turn(manager, session, request, history[-HISTORY_TURNS:])
        history.append({"request": request, "outcome": outcome})
    print("\nAgent: Goodbye!")


if __name__ == "__main__":
    main()

import subprocess
from smolagents.tools import tool


@tool
def slurm_submit(script_path: str) -> str:
    """
    Submit a Slurm script via sbatch and return the sbatch output (job ID).

    Args:
        script_path (str): Path to the Slurm .sh script file to submit.
    """
    res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip())
    return res.stdout.strip()

# smolagents_project

Scientific workflow agent for MACE training and i-PI simulations on a Slurm
cluster.

The project is built with Hugging Face SmolAgents. Its goal is to translate a
user's molecular simulation intent into validated, reviewable workflow
artifacts:

- MACE training Slurm scripts
- i-PI `input.xml` files
- ASE socket-client scripts for i-PI force providers
- i-PI Slurm scripts that launch the server and ASE clients together

## Current Architecture

```text
smolagents_project/
|-- main.py
|-- prompts/
|   |-- manager_instruction.txt
|   |-- mace_instruction.txt
|   `-- ipi_instruction.txt
|-- tools/
|   |-- specs.py
|   |-- mace_cli_schema.py
|   |-- mace_script.py
|   |-- ipi_InputFile_generator.py
|   |-- ipi_runase_generator.py
|   |-- ipi_submit_generator.py
|   |-- get_server_info.py
|   |-- github_fetch.py
|   |-- RetrieverTool.py
|   |-- customize_plan.py
|   |-- show_content.py
|   `-- slurm_submit.py
|-- requirements.txt
|-- .env.example
`-- .gitignore
```

`tools/specs.py` contains the structured contracts used by the generators:

- `ClusterSpec`
- `MACEJobSpec`
- `IPIInputSpec`
- `IPIASEClientSpec`
- `IPISubmitSpec`
- `WorkflowArtifact`

The agent can still collect missing information conversationally, but script
generation now goes through deterministic spec validation before writing files.

MACE parameter synchronization is handled by `tools/mace_cli_schema.py`. It
fetches the official MACE `arg_parser.py`, parses `add_argument` calls with
Python AST, and classifies each option as either a key-value option or a switch
flag. `tools/mace_script.py` validates every generated MACE flag against that
schema before rendering the Slurm script.

i-PI XML edits are intentionally strict. `ipi_template_schema` lists exact paths
for the selected template, and `ipi_InputFile_generator` only accepts exact
XPath updates or explicit append blocks. It does not infer fields from names and
does not create unknown XML tags from simple parameters.

## Setup

```bash
conda create -n mace_agent python=3.11 -y
conda activate mace_agent
pip install --upgrade pip
pip install -r requirements.txt
```

Configure the model through environment variables. Do not put API keys in the
repository.

```bash
export SMOLAGENTS_MODEL_ID=deepseek-chat
export SMOLAGENTS_API_BASE=https://api.deepseek.com/v1
export SMOLAGENTS_API_KEY=your_api_key
```

On Windows PowerShell:

```powershell
$env:SMOLAGENTS_MODEL_ID = "deepseek-chat"
$env:SMOLAGENTS_API_BASE = "https://api.deepseek.com/v1"
$env:SMOLAGENTS_API_KEY = "your_api_key"
```

## Usage

```bash
python main.py
```

Example prompt:

```text
Run a MACE training job using my conda environment. The train file is
/path/to/train.xyz, the validation file is /path/to/valid.xyz, use E0s=average,
max_num_epochs=100, device=cuda, and preview the Slurm script before submit.
```

The manager agent routes MACE training requests to `mace_agent` and i-PI
simulation requests to `ipi_agent`.

## Repository Hygiene

Generated artifacts and large scientific data are intentionally ignored:

- `mace_scripts/`
- `ipi_scripts/`
- `data/*.model`
- `data/*.pt`
- `data/*.xyz`
- `checkpoints/`
- `logs/`
- `results/`
- `tools/__pycache__/`

Keep large models, trajectory files, checkpoints, and cluster logs outside Git.
Use a data registry, object storage, Git LFS, or documented download commands
for reproducible examples.

## Safety Notes

- `main.py` reads the API key only from `SMOLAGENTS_API_KEY`.
- Tool-generated scripts are previewed before submission.
- MACE and i-PI generators validate structured specs before writing files.
- Historical commits may still contain removed secrets or large files. Rotate
  any exposed keys and rewrite repository history before publishing a clean
  public release.

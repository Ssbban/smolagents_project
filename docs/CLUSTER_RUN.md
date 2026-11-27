# First run on a real cluster (2026-11-26)

One cluster: a Slurm 19.05.2 login node on CentOS 7 (glibc 2.17), a CPU partition and a partition of A100 MIG
slices, MACE 0.3.10 to 0.3.13 in three conda environments, i-PI 3.1.5 in one of them.

| What ran | Outcome |
|---|---|
| Without a model: cluster probe, `MACE_script`, `preflight`, `slurm_submit` (answered `y` at a terminal), `job_status` | MACE training jobs completed on CPU (one node pinned) and on a GPU; `job_status` read `sacct` |
| Without a model: two i-PI runs (`03-NVT-SVR`, 192-atom water box, MACE-MP client on CPU) on one node at once | Both completed, each on its own socket |
| Live model, request 1: "train a quick test MACE model on my bulk ice data ... on CPU on the compute-0-0 node" | Plan approved, script generated on the first `MACE_script` call, submitted after `y`, job completed, followed with `job_status` |
| Live model, request 2: "run a short i-PI NVT simulation ... using the model we just trained ... on a GPU" | Two delegations (the MACE agent located the model, the i-PI agent built the run), submitted after `y`, 200 steps completed |
| Both live submissions: the person answered the submit question about 50 s after it was asked, inside a sub-agent's code action | The turn went on; no time-out (v1 failed at 30 s) |
| Live model, after the changes below, one request: "train a small MACE model ... then use that model to run a 100-step i-PI NVT simulation ... tell me whether the simulation ran all its steps and what temperature it ended at" | The MACE agent asked which of the three MACE environments to use (answered after about 400 s, no time-out); `ipi_submit_generator` refused an environment without `i-pi` and the agent took the one with it; the person declined a client whose model path was wrong, with the right path as feedback, and the agent regenerated it; both jobs completed; read with `job_output`, the agent reported step 100 of 100 and 370.97 K, as in `simulation.out` |
| Live model, a person who has no structure file: "give me an input file for ipi that using any random mace model to run a nve simulation, submit on any lightgpu" | Before `write_structure`: both agents built 64-atom silicon in memory and could not save it, and the i-PI agent's probes of guessed paths were all accepted, a made-up one too; no job. After: the agent wrote `si64.xyz` itself, chose MACE-MP-0 small (downloaded by the job) and the one environment with i-PI, and the job completed on the GPU partition (5000 steps NVE); read with `job_output`, its report of the conserved energy (-341.4815 eV throughout) matches `simulation.out` |

What the run showed, and what changed:

| Seen | Cause | Now |
|---|---|---|
| `faiss-cpu==1.16.0` does not install | No wheel for glibc 2.17 | Pinned to 1.9.0.post1, the newest that has one |
| First model call: 401 | An old `SMOLAGENTS_API_KEY` in `~/.bashrc` beat the key in a separate settings file, next to the API base from that file | Settings now come from one place, the environment |
| The agent could not tell which partition a node belongs to | `sinfo -N` prints a node once per partition and the probe had no partition column (38 rows, 16 nodes) | One row per node with its partitions; modules from `module -t avail` |
| `mace_cli_schema` showed the MACE 0.3.17 options, labelled "environment not visible" | Three visible environments have MACE, so none was chosen and the release table was used | It asks for `conda_env` and lists the environments with their MACE versions |
| The MACE agent could not check the label keys of the training file | No tool reads it; MACE trains on with weight zero when a key is missing | `MACE_script` lists the first frame's keys in its notes and refuses a given key that is not there |
| The i-PI job ran in an environment without `i-pi` | `sbatch` exported the submitting shell's `PATH`, which had another environment's `i-pi` | `--export=NONE` in every job script; `ipi_submit_generator` checks the environment for `i-pi` and MACE |
| Both agents ended with "please check the output files yourself" | No tool read `log.i-pi`, `simulation.out` or MACE's log; `show_content` returned a path it did not know as if it were the text | `job_output` reads the files in the bundle of a job of this session; `show_content` refuses other paths |
| A wasted step on `modifications` given as a list; a model path relative to the repository; after the change, one made absolute from the home directory (the person caught it) | The job runs in its bundle directory; the paths the tools return are relative to the repository | Lists accepted; a model file must be an absolute path whose directory exists, and the message names the directory the tools' paths are relative to |
| 4 of 35 actions rejected: written as `<invoke name="python_interpreter">` or `<invoke name="final_answer">` | Claude's own tool-call markup instead of a `<code>` block | `agents.CodeActionModel` turns both into the `<code>` block; all 4 then parse |
| Each plan printed twice | smolagents logs plans at INFO, and the plan review printed it again | Printed by the review only below INFO |
| CPU jobs got one core | `--ntasks=1` and no CPU count | `cpus_per_task` in both generators |
| A person without a structure file got no job; `/nonexistent/definitely/fake.xyz` was accepted as a structure | Generated code cannot write files and no tool wrote a structure; an absolute structure path was never checked | `write_structure`; on the cluster a structure path must exist |
| `log.i-pi` read during a run was empty | Python buffers output redirected to a file until it exits | `PYTHONUNBUFFERED=1` in the i-PI job script: the log showed MD step 80 of a running job |

Not changed: one MIG slice of the GPU node (`SLURM_JOB_GPUS=12`) failed CUDA initialisation with "out of memory" in
two short test jobs while `nvidia-smi` showed it idle; jobs on the other slices ran. That is for the cluster's
administrators.

# Codex Jetson Command Rules

This repo is usually operated from Codex on Windows PowerShell while the real
runtime is on the Jetson through `ssh jetson-codex`.

PowerShell parses command strings before they reach Jetson bash. Complex inline
SSH commands can break when they contain bash syntax such as `$()`, `$var`,
nested quotes, JSON/Python dictionaries, heredocs, or Windows CRLF line endings.

Use these rules in future Codex sessions:

1. Simple read-only commands may run inline:
   `ssh jetson-codex 'cd /home/jetson/VINGS-Mono && ...'`

2. Complex bash, Python, JSON, or patch logic should be written to a temporary
   `.py` or `.sh` file, copied to Jetson, then executed there:
   `scp local_script.py jetson-codex:/tmp/local_script.py`
   `ssh jetson-codex 'python3 /tmp/local_script.py'`

3. Project runs and tests on Jetson must use the project conda environment:
   `source /home/jetson/miniconda3/etc/profile.d/conda.sh && conda activate vings_jetson`

4. Avoid placing bash `$()`, bash variables, Python dictionaries, JSON blocks,
   or heredocs inside PowerShell double-quoted SSH commands.

5. If multi-line text must be sent through stdin, remove Windows CRLF or prefer
   the temporary-file method above.

For a new or compacted Codex conversation, read this file before running Jetson
commands. Also read the current handoff file if task context is needed, but do
not store these command rules in the handoff document.

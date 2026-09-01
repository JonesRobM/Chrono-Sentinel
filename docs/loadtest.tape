# Terminal recording of the load test. Render with:
#     vhs docs/loadtest.tape
#
# Runs against a local uvicorn with CHRONO_TORCH_THREADS=1, which is the exact
# configuration the README's latency table reports. Recording against the
# emulated amd64 container instead would show the emulator, not the service.

Output docs/loadtest.gif

Set Shell bash
Set FontSize 14
Set Width 1280
Set Height 400
Set Padding 20
Set Framerate 20
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set WindowBarSize 40

Hide
Type "clear && pkill -f 'uvicorn threatsim.serving.app' >/dev/null 2>&1; CHRONO_TORCH_THREADS=1 .venv/bin/python -m uvicorn threatsim.serving.app:app --port 8077 --log-level warning >/dev/null 2>&1 & until curl -sf localhost:8077/readyz >/dev/null; do sleep 1; done; clear"
Enter
Sleep 12s
Show

Type "# How much does the Monte Carlo sample count cost?"
Enter
Sleep 1s
Type ".venv/bin/python scripts/loadtest.py --requests 1000 --concurrency 8 \"
Enter
Type "  --warmup 100 --sweep-mc 2,10,30,100"
Enter
Sleep 20s

Type "# Sub-linear: the 30 passes are one batched forward pass, not 30."
Enter
Sleep 3s

Hide
Type "pkill -f 'uvicorn threatsim.serving.app' >/dev/null 2>&1"
Enter

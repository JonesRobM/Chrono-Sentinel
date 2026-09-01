# Terminal recording of the population-drift endpoint. Render with:
#     vhs docs/drift.tape
#
# This is the part of the project that is not just serving: the service
# compares live traffic against the distribution the model was trained on and
# reports how far it has moved. Synthetic traffic looks nothing like NAB, so
# the PSI values here are deliberately large -- that is the detector working,
# not a fault.

Output docs/drift.gif

Set Shell bash
Set FontSize 14
Set Width 1280
Set Height 720
Set Padding 20
Set Framerate 20
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set WindowBarSize 40

Hide
Type "clear && docker rm -f chrono >/dev/null 2>&1; docker run -d --rm --name chrono --platform linux/amd64 -p 7860:7860 ghcr.io/jonesrobm/chrono-sentinel:0.3.0 >/dev/null; until curl -sf localhost:7860/readyz >/dev/null; do sleep 1; done; clear"
Enter
Sleep 15s
Show

Type "# Drift is withheld until enough traffic has been seen."
Enter
Sleep 1s
Type "curl -s localhost:7860/drift | jq '{observations_in_buffer, minimum_observations, features}'"
Enter
Sleep 4s

Type "# Send 60 windows of synthetic traffic."
Enter
Sleep 1s
Type "for i in $(seq 60); do curl -so /dev/null localhost:7860/score \"
Enter
Type "  -H 'Content-Type: application/json' -d @examples/noisy_window.json; done"
Enter
Sleep 6s

Type "curl -s localhost:7860/drift | jq '.features | to_entries[:3] | from_entries'"
Enter
Sleep 4s

Type "# Every input feature has shifted: this traffic is nothing like"
Enter
Type "# the NAB series the model was trained on. PSI > 0.25 = significant."
Enter
Sleep 2s
Type "curl -s localhost:7860/metrics | grep -E '^chrono_(drift_max_psi|requests_total)'"
Enter
Sleep 5s

Hide
Type "docker rm -f chrono >/dev/null 2>&1"
Enter

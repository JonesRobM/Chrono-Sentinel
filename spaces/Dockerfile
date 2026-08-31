# Hugging Face Space entry point.
#
# Spaces' Docker SDK builds a Dockerfile from the Space repository; it cannot
# pull an arbitrary registry tag on its own. This two-line file is the link:
# it deploys the exact image that CI built and tested, rather than rebuilding
# from source and hoping the result matches.
#
# Requirements:
#   * the GHCR package must be PUBLIC, or the Space builder cannot pull it
#   * pin a concrete tag, never :latest, so the deployed artefact is the one
#     the README's results table describes
#
# Copy this file, and spaces/README.md, to the root of the Space repository.

FROM ghcr.io/jonesrobm/chrono-sentinel:v0.2.1

EXPOSE 7860

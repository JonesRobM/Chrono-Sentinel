---
title: Chrono-Sentinel
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Chrono-Sentinel scoring service

Time-series anomaly detection with Monte Carlo Dropout uncertainty.

POST a window of time-series points, get back an anomaly score with an
uncertainty interval. Interactive API documentation is at `/docs`.

| Endpoint   | Purpose                                              |
| ---------- | ---------------------------------------------------- |
| `POST /score`  | Score one window                                 |
| `GET /healthz` | Liveness                                         |
| `GET /readyz`  | Readiness, including the expected window size    |
| `GET /drift`   | Population drift against the training reference  |
| `GET /metrics` | Prometheus exposition                            |

Source and measured latency figures: https://github.com/jonesrobm/Chrono-Sentinel

Running on free-tier shared CPU. It sleeps after inactivity, so the first
request after a pause pays a cold start.

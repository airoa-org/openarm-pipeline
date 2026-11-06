# Security

## Reporting a vulnerability

Report vulnerabilities privately to [ext-oss@airoa.org](mailto:ext-oss@airoa.org), or with
**Report a vulnerability** on the repository's Security tab. Do not open a public issue.

## The policy server is unauthenticated

`policy/airoa0_inference/serve_policy.py` serves plain HTTP with no authentication and no TLS, and listens on all
interfaces by default (`--host 0.0.0.0`, port 8000). Anyone who can reach the port can send it observations, read
its metadata and reset its sessions, and the runner sends it the wrist camera images unencrypted. So run it on the
robot host or on a closed network, and firewall port 8000 (README, Container privileges).

## The container is privileged

`scripts/oap` runs the container with `--privileged -v /dev:/dev --network host`, and `scripts/gui` lets local root
clients use your X display. Code in the container has root access to the host's devices, network and display.
Run it only on a dedicated robot host, not on a shared or internet-facing machine (README, Container privileges).

openarm-pipeline is research software for a robot in a lab you control. It is not designed or hardened for
deployment outside one.

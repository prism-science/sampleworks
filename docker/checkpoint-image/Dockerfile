# syntax=docker/dockerfile:1
# Sampleworks checkpoints image: the model weights the main Dockerfile copies
# in from its CHECKPOINTS_IMAGE build arg, rebuilt from the upstream sources in
# manifest.tsv with every file's sha256 verified. Published to Harbor by
# .github/workflows/checkpoints.yml.
#
# Build locally:
#   docker build --platform linux/amd64 -t sampleworks-checkpoints:local docker/checkpoint-image

FROM alpine:3.20@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc AS fetch
RUN apk add --no-cache curl
COPY manifest.tsv fetch.sh /fetch/
RUN sh /fetch/fetch.sh /checkpoints

FROM scratch
# One layer per checkpoint, largest first, so a cold pull fetches them in
# parallel (the same reasoning as the main Dockerfile's per-file COPY lines).
COPY --from=fetch /checkpoints/boltz1_conf.ckpt                /checkpoints/
COPY --from=fetch /checkpoints/rf3_foundry_01_24_latest.ckpt   /checkpoints/
COPY --from=fetch /checkpoints/boltz2_conf.ckpt                /checkpoints/
COPY --from=fetch /checkpoints/bioemu2-preview/                /checkpoints/bioemu2-preview/
COPY --from=fetch /checkpoints/mols/                           /checkpoints/mols/
COPY --from=fetch /checkpoints/protenix_base_default_v0.5.0.pt /checkpoints/
COPY --from=fetch /checkpoints/ccd.pkl                         /checkpoints/

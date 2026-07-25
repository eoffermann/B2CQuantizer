# Deployment guide

This guide covers deploying B2CQuantizer to a [RunPod](https://runpod.io) GPU pod: choosing a
GPU, sizing disk, publishing the Docker image, starting the container, and exposing the UI.
It reflects the deployment shape defined in [`SPEC.md` §4](../SPEC.md#4-deployment-shape).

B2CQuantizer does not provision pods for you (see `SPEC.md` §3, non-goals) — you create and
manage the RunPod pod yourself; the app only runs inside it.

## 1. Choosing a GPU

The right GPU depends on the model size and which quant formats you plan to select.

| Requirement | Minimum GPU | Why |
|---|---|---|
| 24B-parameter Mistral/Mistral3 source, GGUF + non-NVFP4 safetensors quants | **H100 80 GB** or **A100 96 GB** | BF16 weights for a 24B model are ~48 GB; loading the full model on GPU for Lane A plus headroom for calibration activations needs a large-memory card. |
| Any job that includes **NVFP4** in the quant selection | **RTX PRO 6000 Blackwell** or **RTX 5090** (Blackwell tier, compute capability SM ≥ 12.0) | NVFP4 kernels require Blackwell tensor cores. The Docker image is built against CUDA 12.8 specifically for SM120 support. Non-Blackwell GPUs cannot run NVFP4 at all — the UI grays the option out when it detects `torch.cuda.get_device_properties(0).major < 12`. |
| Smaller / non-24B source models, GGUF-only jobs | Smaller GPUs may work | Not a target configuration for v1; size the GPU to comfortably hold the source model in BF16 plus working memory for whichever safetensors formats you select. |

If you plan to run **both** a 24B-class model **and** NVFP4, you need a card that is both
large-memory and Blackwell-tier (e.g., RTX PRO 6000 Blackwell 96 GB). H100/A100 (Hopper/Ampere)
cannot produce NVFP4 output regardless of memory size.

## 2. Pod disk sizing

Recommended: **≥200 GB** of pod disk for 24B-parameter models (`SPEC.md` §4).

Disk is used for:
- The one-time source model download (BF16 weights, ~48 GB for a 24B model).
- In-progress artifacts: the canonical BF16 GGUF intermediate (~45 GB for a 24B model), per-quant
  output directories/files before upload, and any uploaded calibration JSONL.
- Aggressive cleanup deletes each artifact immediately after its upload succeeds, so peak usage —
  not cumulative usage across every quant — is what matters. Still, budget for the source download,
  the BF16 GGUF intermediate, and the largest single in-flight quant output existing at once.

Smaller source models need proportionally less disk; 200 GB is sized for the 24B-parameter case
called out in the spec. Undersized disk during a job surfaces as a failure — see
[`docs/troubleshooting.md`](troubleshooting.md) for what that failure looks like and how it's handled.

## 3. Publishing the Docker image

Build and push the image from the repository root to a registry your RunPod pod can pull from —
GHCR or Docker Hub both work; use whichever your account is set up for.

**GHCR:**
```bash
docker build -t ghcr.io/<your-org>/b2cquantizer:latest .
echo "$GHCR_TOKEN" | docker login ghcr.io -u <your-username> --password-stdin
docker push ghcr.io/<your-org>/b2cquantizer:latest
```

**Docker Hub:**
```bash
docker build -t <your-dockerhub-user>/b2cquantizer:latest .
docker login
docker push <your-dockerhub-user>/b2cquantizer:latest
```

If the destination repository is private, RunPod needs registry credentials configured on the pod
template (RunPod's "Container Registry Credentials" setting) — the image itself carries no
operator credentials (`SPEC.md` §12).

Tag images meaningfully (e.g., a version or date tag in addition to `latest`) so you can pin a
known-good pod template rather than always pulling whatever `latest` currently points to.

## 4. Environment variables at container start

B2CQuantizer intentionally takes almost no runtime configuration via environment variables. Per
`SPEC.md` §9/§12, the HuggingFace token and all per-job settings (source model, calibration
source, quant selection, upload owner, public/private) are entered through the web UI at job
launch time, not passed as container environment — this keeps the token out of any subprocess
environment and out of the pod's persisted configuration.

What **is** fixed by the image itself (baked into the `Dockerfile`, not operator-configurable at
`docker run` time in the current design):

| Variable | Value | Purpose |
|---|---|---|
| `HF_HOME` | `/workspace/hf-cache` | Where `huggingface_hub` caches downloaded model/dataset files, on the pod's ephemeral disk. |
| (port) | `8000` | The app always listens on port 8000; there is no `PORT` env var to override it. Map/forward container port 8000 on the RunPod side (see §5 below). |

There is no `HF_TOKEN`, `RUNPOD_*`, or similar environment variable read by the app — do not set
one expecting it to pre-authenticate a job; enter the token in the UI for each job instead.

If you need to override registry mirrors, proxy settings, or similar host-level concerns, do so
via RunPod's pod template environment variable UI as you would for any container — those are
consumed by the OS/Docker layer, not by the B2CQuantizer application itself.

## 5. Running the container and exposing the port

Start the container with GPU access and port 8000 published:

```bash
docker run -d --gpus all -p 8000:8000 ghcr.io/<your-org>/b2cquantizer:latest
```

On RunPod specifically, when creating the pod (via a Pod Template or the on-demand GPU pod UI):

1. Set the container image to your pushed tag.
2. Under **Expose HTTP Ports**, add `8000`.
3. Leave **Expose TCP Ports** empty unless you have another reason to need raw TCP access.
4. Launch the pod.
5. Once running, open the pod's **Connect** panel. RunPod exposes the mapped port either as a
   generated HTTPS proxy URL (`https://<pod-id>-8000.proxy.runpod.net`) or via an SSH tunnel,
   depending on how the pod template is configured. Use whichever RunPod offers for HTTP port 8000.

The app itself performs **no TLS termination and no authentication** (`SPEC.md` §4, §12) — RunPod's
proxy/tunnel is the entire trust boundary. Do not expose port 8000 directly to the public internet
through any other mechanism (e.g., a raw TCP port forward without RunPod's proxy) without adding
your own access control in front of it.

## 6. Expected first-run download times (estimates)

These are **rough estimates**, not measured or spec'd figures — actual times depend on RunPod's
network egress from the HF Hub, the source model's size, and how many quant formats you selected
(more selected formats means more upload time on top of the fixed download time). Provided so you
can sanity-check that a job is progressing rather than stalled.

| Step | Approx. size (24B model) | Estimated time* |
|---|---|---|
| Source model download (BF16 safetensors) | ~48 GB | 5–15 minutes on a well-connected RunPod region |
| BF16 GGUF conversion (Lane B, one-time) | writes ~45 GB | 10–20 minutes, CPU-bound |
| Per safetensors quant (Lane A, GPU-resident) | 6–13 GB output depending on format | 10–30 minutes each, includes calibration pass |
| Per GGUF K/I-quant (Lane B) | 3–13 GB output depending on format | 2–8 minutes each |
| imatrix computation (one-time, only if an I-quant is selected) | small output file | 10–20 minutes |
| Per-quant upload to HF | matches quant output size | Minutes, depends on RunPod's upload bandwidth to the HF Hub |

*Estimates for a 24B-parameter source model on an H100/A100-class pod with typical RunPod network
performance. Smaller source models scale down roughly proportionally. A full job selecting most of
the 30-format catalog for a 24B model should be budgeted in **hours**, not minutes — plan pod
rental time accordingly.

## See also

- [`docs/troubleshooting.md`](troubleshooting.md) — common failure modes and how to diagnose them.
- [`tests/smoke/README.md`](../tests/smoke/README.md) — manual end-to-end smoke procedure to run
  against a freshly deployed pod before trusting it with a real job.
- [`SPEC.md`](../SPEC.md) — authoritative design spec (§4 deployment shape, §11 failure modes,
  §12 security).

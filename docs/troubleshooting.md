# Troubleshooting

This document covers failure modes an operator is likely to hit while running B2CQuantizer,
what they look like in the UI/logs, and how to resolve or work around them. It's grounded in the
failure-handling model in [`SPEC.md` §11](../SPEC.md#11-failure-modes-and-handling) and the
security/architecture notes in §12 and §6.

General notes before diving into specific failures:

- **Blast radius matters.** B2CQuantizer isolates failures at three levels: a single quant failing
  only affects that quant (the lane continues); a lane's one-time setup step failing (GPU load for
  Lane A, BF16 GGUF conversion for Lane B) skips every quant in that lane but the other lane keeps
  going; only source-download failure aborts the whole job. Check the job report at the bottom of
  the job view to see exactly which of these happened.
- **Per-quant and per-lane logs** are visible live via the expandable row in the job view (last
  ~50 lines, streamed over SSE) and persisted in full to `/workspace/logs/<job_id>/<quant>.log` on
  the pod (`SPEC.md` §13) — `docker cp` them out if you need the complete output rather than the
  tail.
- **Nothing here requires a new pod** unless explicitly noted (rebuilding the image, changing GPU
  tier). Most single-quant and single-lane failures can be addressed by fixing the underlying cause
  and launching a new job — B2CQuantizer does not auto-retry (`SPEC.md` §3, §11), so retry is
  always "start a new job."

---

## GPU OOM during safetensors load (Lane A)

**Symptom:** Every safetensors quant you selected shows status `SKIPPED` with an error like
`Lane A setup failed: CUDA out of memory ...`. GGUF quants (Lane B) proceed normally.

**Cause:** Lane A loads the full BF16 source model onto the GPU once, before running any quant
recipe against it (`SPEC.md` §6). If the model plus its CUDA runtime overhead doesn't fit in GPU
memory, the load itself fails before any quant work starts — this is a lane-setup failure, not a
per-quant one, so every safetensors quant in the job is marked `SKIPPED` together.

**Fix:**
- Confirm the pod's GPU has enough memory for the source model in BF16 (a 24B-parameter model is
  ~48 GB in weights alone before any activation/calibration overhead) — see
  [`docs/deployment.md` §1](deployment.md#1-choosing-a-gpu).
- Check nothing else on the pod is holding GPU memory (a stale process from a previous job, a
  separate container). Restart the pod if needed — jobs are not persisted across restarts anyway
  (`SPEC.md` §3), so this is safe.
- If you're consistently OOMing on a card that should have room, move to a larger-memory GPU
  (A100 96 GB or H100 80 GB minimum for 24B models).

---

## Disk-full during BF16 GGUF conversion (Lane B)

**Symptom:** Every GGUF quant you selected (K-quants, I-quants, misc, mmproj) shows status
`SKIPPED` with an error referencing the BF16 GGUF conversion step, e.g. `Lane B setup failed:
... No space left on device`. Safetensors quants (Lane A) proceed normally.

**Cause:** Lane B's first step converts the already-downloaded source safetensors into a canonical
BF16 GGUF (~45 GB for a 24B model) via `convert_hf_to_gguf.py` — this is the shared substrate every
downstream GGUF quant reads from (`SPEC.md` §6). If pod disk fills up during this write (source
download + BF16 GGUF output + any calibration files can add up), the conversion fails and, like the
Lane A GPU-load case, this is a lane-setup failure: every GGUF quant is marked `SKIPPED` together.

**Fix:**
- Check free disk on the pod (`df -h /workspace`, also printed in the entrypoint's startup
  diagnostics). Compare against the ≥200 GB recommendation in
  [`docs/deployment.md` §2](deployment.md#2-pod-disk-sizing).
- Clear out artifacts from a previous job if the pod was reused without a restart — cleanup
  happens per-artifact after successful upload, but a failed/aborted prior job can leave files
  behind.
- Resize the pod's disk (RunPod allows increasing volume size on some pod types) or move to a pod
  template with a larger disk allocation, then launch a new job.

---

## HF 401 on private-model access

**Symptom:** The job never starts — the UI shows an error at launch time (source model validation
fails), or if it slips past initial validation, the whole job aborts immediately with a 401/403
from the HF Hub on the source-model download.

**Cause:** The source model repo is private (or gated) and the HF token entered for the job either
doesn't have read access to it, is malformed, or expired. `SPEC.md` §11 classifies "source model
404 or 401" as whole-job blast radius — there's nothing else useful to do without the source
model, so the job aborts before starting rather than partially running.

**Fix:**
- Confirm the token was copied correctly (no truncation/whitespace) into the password field.
- Confirm the token's associated HF account actually has read access to the private/gated repo —
  gated repos (e.g., some Mistral releases) require accepting a license on huggingface.co before
  any token for that account can download them, independent of token scope.
- If the token is a fine-grained token, confirm it has the "Read access to contents of all repos
  under your personal namespace" or equivalent repo-specific read scope, not just an org-scoped or
  write-only grant.
- Remember the token is wiped from memory at the end of every job (`SPEC.md` §9, §12) — re-enter it
  for the retry job; there's no "remembered" token to go stale silently between jobs.

---

## mmproj `v.token_embd.img_break` missing (patch didn't apply — rebuild image)

**Symptom:** The mmproj export step for a multimodal Mistral3 source fails with an assertion error
referencing the `v.token_embd.img_break` tensor not being present in the exported mmproj GGUF. The
mmproj quant is marked `FAILED`; other GGUF quants in Lane B are unaffected (this is a per-quant
failure, not a lane-setup failure).

**Cause:** Mistral3's pixtral vision-tower export has two known upstream bugs in llama.cpp's
`convert_hf_to_gguf.py` (`SPEC.md` §6, step 4). B2CQuantizer works around them by applying a
runtime patch script to the in-image copy of `convert_hf_to_gguf.py` before running the mmproj
export, and then hard-asserts that `v.token_embd.img_break` made it into the output as a
correctness check — that tensor's presence is the signal the patch actually took effect. If this
assertion fails, the patch did not apply cleanly against the llama.cpp commit baked into the image
(e.g., the pinned llama.cpp version drifted from what the patch script targets, or the patch script
itself has an error).

**Fix:** This is an **image-level problem, not a per-job one** — re-running the job will hit the
same failure every time. Retrying within the UI will not help.
- Rebuild the Docker image and confirm the patch step's output during the build (or at mmproj
  export time) shows the patch applying without errors, then re-verify with the
  [smoke procedure](../tests/smoke/README.md) against a known multimodal Mistral3 model before
  trusting the image for real jobs.
- If the llama.cpp commit pinned in `scripts/build_llama_cpp.sh` was recently changed, check
  whether the patch script's target line numbers/context still match that commit — upstream
  llama.cpp changes to the pixtral conversion path can invalidate the patch, requiring an update to
  the patch script itself.
- Non-multimodal GGUF quants (K-quants, I-quants, misc) do not depend on this patch and are safe to
  use even while mmproj export is broken.

---

## NVFP4 kernel launch failure (wrong GPU tier)

**Symptom:** The NVFP4 safetensors quant fails (per-quant `FAILED`, not a whole-lane failure) with
a CUDA kernel-launch or "unsupported" error, typically referencing compute capability or missing
kernels for the current device.

**Cause:** NVFP4 requires Blackwell tensor cores (compute capability SM ≥ 12.0) — it is not
supported on Hopper (H100) or Ampere (A100) GPUs at all (`SPEC.md` §4, §7). The UI is meant to gray
out the NVFP4 checkbox when it detects `torch.cuda.get_device_properties(0).major < 12` (per `PLAN.md` Global Constraints), so hitting
this failure usually means one of:
- The job was launched on a pod whose GPU changed after the UI's detection ran (unlikely, but
  possible with certain RunPod pod-swap flows).
- The detection heuristic didn't catch an edge-case GPU/driver combination.

**Fix:**
- Confirm the pod's actual GPU tier: check `nvidia-smi` output (also printed in the entrypoint's
  startup diagnostics) for the GPU name, and cross-reference against Blackwell-tier cards (RTX 5090,
  RTX PRO 6000 Blackwell). H100 and A100 are **not** Blackwell and cannot run NVFP4 regardless of
  driver/CUDA version.
- If you need NVFP4 output, re-launch the job on a Blackwell-tier pod (see
  [`docs/deployment.md` §1](deployment.md#1-choosing-a-gpu)) and deselect NVFP4 for the rest of
  this job's run, or just don't select it at all on non-Blackwell hardware.
- All other selected quants in the same job are unaffected — this is isolated to the single NVFP4
  quant (`SPEC.md` §11, "individual quant fails" row).

---

## llm-compressor scheme errors on architecture mismatch

**Symptom:** A safetensors quant (typically AWQ) fails with an error from `llm-compressor` about a
mapping/scheme match — commonly something like a regex pattern matching too many or too few modules,
or a `Recipe`/mapping validation error referencing layer names that don't look right for the loaded
model.

**Cause:** `SPEC.md` §6 calls out specifically that default AWQ mapping patterns (e.g.
`re:.*input_layernorm$`) glob **all** decoder layers' layernorms into a single set for a
multimodal Mistral3 model and error out — AWQ on multimodal Mistral3 requires explicit
per-decoder-layer scoped mappings instead of the library's default globs. More generally, this
class of error shows up whenever the loaded model's actual module structure doesn't match what a
quant recipe's ignore-patterns/mapping assumptions expect — e.g., a differently-shaped Mistral3
variant, or a text-only Mistral model being treated as if it had vision-tower modules to ignore.

**Fix:**
- Check the error text for which pattern/mapping failed to resolve cleanly — it usually names the
  offending regex or module path.
- Confirm the source model is genuinely Mistral or Mistral3 family (`SPEC.md` §3 — non-Mistral
  architectures are out of scope and not expected to work; a scheme error is one of the ways an
  unsupported or unusually-structured checkpoint can surface).
- This is a per-quant failure — other selected formats in the same lane continue independently
  (`SPEC.md` §11). If only AWQ is affected, the other safetensors formats (GPTQ, NVFP4, FP8) and
  every GGUF quant are unaffected and their results remain usable.
- If the model is a legitimate Mistral3 variant hitting this for the first time, it likely indicates
  the per-decoder-layer scoped AWQ mapping logic needs updating for that variant's module naming —
  this is a code-level fix, not something resolvable by re-running the job.

---

## See also

- [`docs/deployment.md`](deployment.md) — GPU/disk sizing so you avoid the OOM/disk-full cases above.
- [`tests/smoke/README.md`](../tests/smoke/README.md) — run this after any image rebuild (e.g.,
  after fixing the mmproj patch) to confirm the fix actually took.
- [`SPEC.md` §11](../SPEC.md#11-failure-modes-and-handling) — the authoritative failure-mode table.

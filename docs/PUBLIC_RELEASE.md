# Public release checklist

Use this checklist before changing the repository visibility. A green CI build is necessary but **not sufficient**.

- [ ] Merge the release-readiness changes and confirm CI passes for the resulting main commit.
- [ ] **Audit all Git history and branches** for credentials, tokens, personal data, proprietary datasets, and large generated artifacts. A clean current tree does not guarantee clean history. Rotate any exposed credentials; deleting a file in a new commit does not remove it from history.
- [ ] Confirm provenance and redistribution rights for `firmware/main/font8x12.h` (generated from a local font file), the bundled `research/neurips_2026.sty`, and any other third-party materials. Replace or properly attribute if necessary.
- [ ] Confirm that any future dataset, teacher output, or trained weights have licenses permitting redistribution. None are published here.
- [ ] Verify the PDF matches the current LaTeX source; generate it locally using `make paper` if the source changes.
- [ ] Keep claims aligned with [STATUS.md](STATUS.md): host memorization/parity and a firmware *build* are not proof of working physical hardware or held-out translation quality.
- [ ] Check GitHub repository metadata (description, topics), default branch, README links, and the absence of internal-only issues or discussions.
- [ ] Test a fresh clone on a machine without private environment variables; ensure `python -m pytest -q`, the CMake build, and the smoke test work.
- [ ] If publishing a hardware demonstration, provide board photos, wiring confirmation and repeatable measured benchmarks, clearly labeled as such.

**Release policy:** This repository may be published as *prototype code and research notes* after provenance/security review; do not present it as a benchmarked commercial-quality translator before the missing experiments are completed.

# PublicHearingBR data preparation

## Purpose

Provide the fixed PublicHearingBR NLI file for supervised experiments and target
domain evaluations.

## Dataset source

The configurations use the Hugging Face dataset revision
`2f84a44bc34df483e25c987f0ff86caad0ab3433`. The repository scripts download it
to the Hugging Face cache when needed.

## Verify the local file

After the first download, locate the cached file and verify its hash:

```bash
find "$HOME/.cache/huggingface/hub" -type f -name PublicHearingBR_NLI.jsonl
```

The expected SHA-256 is
`13408024c4776eff24f05ee6a56f9b1d33614524d19e90c0cce2fa310487a34e`.

## Outputs

The fixed dataset contains 4,235 modelable examples from 206 hearings, with
501 positive labels. Each experiment configuration records the dataset revision
and validates its expected counts.

# PublicHearingBR LLM-judge prompts

These are the Portuguese prompt templates used by the offline reconstruction
of the PublicHearingBR hallucination-judge experiment.

Source: Fernandes et al., *PublicHearingBR*, arXiv:2410.07495v2, Appendix A,
Figures 8–10 (original Portuguese prompts).

The paper is the source of the prompt text. The repository did not previously
contain these templates as standalone files.

The reconstruction uses the following explicit hypothesis for `{TEXTO}`:

1. take the four stored evidence passages in their stored order;
2. join them with exactly `\n\n`;
3. substitute that string and the original Portuguese opinion into the User
   template.

This is an input reconstruction hypothesis, not a claim about the original
API serialization. The canonical outputs produced by the analysis are also
proxies, not raw API responses.

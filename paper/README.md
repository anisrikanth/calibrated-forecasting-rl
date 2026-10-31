# Paper

`main.tex` and `refs.bib`, with figures in `figures/` (copies of the ones in `eval/figures/`).

Build locally:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or upload the folder to Overleaf (New Project → Upload Project, as a zip) and compile with pdfLaTeX.

The red date line marks the draft; remove it before submitting. Every number comes from the results in
`eval/` (`eval/rl_runs5to7.md`, `eval/rl_runs9_10.md`, `eval/prompt_ablation.md`, `eval/extra_analyses.md`,
`eval/review_checks.md`, `eval/test_results.md`, and the CSVs in `eval/analysis/` and `eval/test_results/`).

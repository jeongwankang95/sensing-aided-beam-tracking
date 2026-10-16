#!/bin/bash
# Compiles the TikZ sources into one PDF per figure, cut to its bounding box, in the paper's document class (IEEEtran):
# Fig. 2 from figures/tikz/fig2_controller_flow.tikz (a diagram, no data) and Figs. 3-6 from results/figures/*.tikz (written by
# figures/fig3_frontier.py ... fig6_vsense.py), into results/figures/*.pdf. With the argument "paper" it compiles the TikZ code
# of the paper's figures in figures/tikz/ instead, into results/figures/paper/. Needs pdflatex with -shell-escape and the
# IEEEtran class (both in TeX Live). Run from anywhere: bash figures/make_figures.sh [paper]
set -e
cd "$(dirname "$0")/.."
SRC=results/figures; OUT=results/figures
if [ "$1" = paper ]; then SRC=figures/tikz; OUT=results/figures/paper; fi
W="$(mktemp -d)"
mkdir "$W/build" && mkdir -p "$OUT"
cp figures/make_figures.tex figures/tikz/fig2_controller_flow.tikz "$W"/
for f in fig3_frontier fig4_perpass fig5_example fig6_vsense; do cp "$SRC/$f.tikz" "$W"/; done
(cd "$W" && pdflatex -shell-escape -interaction=nonstopmode make_figures.tex > /dev/null)
for f in fig2_controller_flow fig3_frontier fig4_perpass fig5_example fig6_vsense; do cp "$W/build/$f.pdf" "$OUT/$f.pdf"; done
rm -rf "$W"
echo "Figs. 2-6 written to $OUT/"

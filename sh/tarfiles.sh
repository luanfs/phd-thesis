#!/bin/bash

# Srcipt to tar instalation files of imodel

date=` date +%F `
version=` date +%y.%m.%d `
echo "Today: " $date

texfiles="conteudo/*.tex"

extrafiles="extras/* "

figures="figuras/*"

scripts="sh/*.sh "

others="Makefile \
README.* \
latexmkrc \
thesis_luan.tex \
thesis_luan.pdf"

files="$texfiles $extrafiles $figures $scripts $others"

output="thesis_luan.tar.bz2"

tar cjfv $output $files

echo "File " $output " ready!"
echo

echo "-------------------------------------------"

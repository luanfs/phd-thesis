#!/bin/bash

date=` date +%F `
version=` date +%y.%m.%d `
echo "Today: " $date

output="thesis_luan.tar.bz2"
bkpdir="thesis_luan" 

#Edit place to sync relative to system used
dropdir="/home/luanfs/Dropbox/doc"/$bkpdir
echo "Sync with Dropbox:"
rsync -v -t -u $output  "$dropdir/."
echo "Synchronized with Dropbox"
echo

# remove tar file
rm -rf $output

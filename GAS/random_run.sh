#!/usr/bin/env bash
for i in {1..100}
do
  echo $i
  python -u train.py $i
done
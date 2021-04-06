#!/bin/bash
wget -O model.pth https://www.dropbox.com/s/mloqyf28i977jbq/model.pth?dl=1
python3 ly_test.py $1 $2
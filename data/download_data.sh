#!/usr/bin/env bash
# Downloads the Pavia University hyperspectral dataset (used by Figure 5)
# into this directory (~34 MB total).
set -e
cd "$(dirname "$0")"
curl -L -O http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat
curl -L -O http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat
echo "Done: PaviaU.mat and PaviaU_gt.mat downloaded to data/"

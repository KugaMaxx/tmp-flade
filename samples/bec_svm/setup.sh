
#!/bin/bash

ROOT=$(cd "$(dirname "$0")";pwd)

# Build Generator
cd ${ROOT}/generator
mkdir build
cd build
cmake ..
make

# Build Extractor
cd ${ROOT}/extractor
mkdir build
cd build
cmake ..
make

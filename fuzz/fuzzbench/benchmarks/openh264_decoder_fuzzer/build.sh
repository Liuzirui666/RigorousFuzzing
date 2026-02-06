#!/bin/bash -eu
# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# [FIX] Robustness Upgrade:
# The external repository 'mozillasecurity/fuzzdata' is unstable/unreachable, causing build failures.
# We remove the dependency on external git/svn downloads entirely.
# Instead, we rely solely on the seed files (res/*.264) provided by the openh264 project itself.

# 1. Create the corpus directory manually (since svn export won't create it for us)
mkdir -p corpus

# 2. Copy the project's own test files into the corpus directory
# The 'res' folder is part of the openh264 source code we already cloned.
cp ./res/*.264 ./corpus/

# 3. Package the seeds
zip -j0r ${OUT}/decoder_fuzzer_seed_corpus.zip ./corpus/

# build 
if [[ $CXXFLAGS = *sanitize=memory* ]]; then
  ASM_BUILD=No
else
  ASM_BUILD=Yes
fi
make -j$(nproc) ARCH=$ARCHITECTURE USE_ASM=$ASM_BUILD BUILDTYPE=Debug libraries
$CXX $CXXFLAGS -o $OUT/decoder_fuzzer -I./codec/api/svc -I./codec/console/common/inc -I./codec/common/inc -L. $LIB_FUZZING_ENGINE $SRC/decoder_fuzzer.cpp libopenh264.a
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

# FROM gcr.io/oss-fuzz-base/base-builder
FROM gcr.io/oss-fuzz-base/base-builder@sha256:87ca1e9e19235e731fac8de8d1892ebe8d55caf18e7aa131346fc582a2034fdd

LABEL maintainer="noone@gmail.com"

# [FIX] Explicitly install libstdc++-9-dev instead of the virtual package.
# The error log showed that libstdc++-dev is virtual and provided by libstdc++-9-dev (Ubuntu 20.04 default).
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y libstdc++-9-dev libstdc++-9-dev:i386 nasm subversion

RUN git clone --depth 1 https://github.com/cisco/openh264.git openh264

# [FIX] Use absolute path for WORKDIR
WORKDIR $SRC/openh264

COPY build.sh decoder_fuzzer.cpp $SRC/

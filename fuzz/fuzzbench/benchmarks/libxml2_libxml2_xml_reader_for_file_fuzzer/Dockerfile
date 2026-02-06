# Copyright 2016 Google Inc.
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


# MAINTAINER ochang@chromium.org
# RUN apt-get update && apt-get install -y make autoconf automake libtool pkg-config

# RUN git clone --depth 1 https://gitlab.gnome.org/GNOME/libxml2.git
# WORKDIR libxml2

# COPY build.sh $SRC/
# COPY *.cc *.h *.options *.dict $SRC/

# [FIX] Install zlib and lzma development libraries to ensure headers and libs are present
RUN apt-get update && apt-get install -y make autoconf automake libtool pkg-config zlib1g-dev liblzma-dev

# [FIX] Clone the repo and checkout the specific commit to ensure reproducibility
RUN git clone https://gitlab.gnome.org/GNOME/libxml2.git
WORKDIR libxml2
RUN git checkout 99a864a1f7a9cb59865f803770d7d62fb47cad69

COPY build.sh $SRC/
COPY *.cc *.h *.options *.dict $SRC/
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

MAINTAINER mmoroz@chromium.org
ADD bionic.list /etc/apt/sources.list.d/bionic.list
ADD nasm_apt.pin /etc/apt/preferences
RUN apt-get update && apt-get install -y make autoconf automake libtool build-essential \
    libass-dev libfreetype6-dev libsdl1.2-dev libnuma-dev \
    libvdpau-dev libxcb1-dev libxcb-shm0-dev \
    pkg-config texinfo libbz2-dev zlib1g-dev yasm cmake mercurial wget \
    xutils-dev libpciaccess-dev nasm


# Upgrade autoconf to satisfy projects requiring autoconf >= 2.71 (e.g., theora)
RUN set -eux; \
    cd /tmp; \
    wget -q https://ftp.gnu.org/gnu/autoconf/autoconf-2.71.tar.gz; \
    tar xf autoconf-2.71.tar.gz; \
    cd autoconf-2.71; \
    ./configure --prefix=/usr/local; \
    make -j"$(nproc)"; \
    make install; \
    autoconf --version; \
    autoreconf --version; \
    cd /; \
    rm -rf /tmp/autoconf-2.71 /tmp/autoconf-2.71.tar.gz


RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg

RUN wget https://www.alsa-project.org/files/pub/lib/alsa-lib-1.1.0.tar.bz2
# RUN git clone --depth 1 git://anongit.freedesktop.org/mesa/drm
RUN git clone --depth 1 https://gitlab.freedesktop.org/mesa/drm.git

RUN git clone --depth 1 https://github.com/mstorsjo/fdk-aac.git
ADD https://sourceforge.net/projects/lame/files/latest/download lame.tar.gz
# RUN git clone --depth 1 git://anongit.freedesktop.org/xorg/lib/libXext
# RUN git clone --depth 1 git://anongit.freedesktop.org/git/xorg/lib/libXfixes
RUN git clone --depth 1 https://gitlab.freedesktop.org/xorg/lib/libxext.git
RUN git clone --depth 1 https://gitlab.freedesktop.org/xorg/lib/libXfixes.git

# RUN git clone --depth 1 https://github.com/01org/libva
RUN git clone --depth 1 https://github.com/intel/libva.git libva

#RUN git clone --depth 1 -b libvdpau-1.2 git://people.freedesktop.org/~aplattner/libvdpau
RUN git clone --depth 1 -b libvdpau-1.2 https://gitlab.freedesktop.org/vdpau/libvdpau.git

RUN git clone --depth 1 https://chromium.googlesource.com/webm/libvpx
# RUN git clone --depth 1 git://git.xiph.org/ogg.git
# RUN git clone --depth 1 git://git.xiph.org/opus.git
# RUN git clone --depth 1 git://git.xiph.org/theora.git
# RUN git clone --depth 1 git://git.xiph.org/vorbis.git
RUN git clone --depth 1 https://gitlab.xiph.org/xiph/ogg.git
RUN git clone --depth 1 https://gitlab.xiph.org/xiph/opus.git
RUN git clone --depth 1 https://gitlab.xiph.org/xiph/theora.git
RUN git clone --depth 1 https://gitlab.xiph.org/xiph/vorbis.git

# RUN git clone --depth 1 git://git.videolan.org/git/x264.git
RUN git clone --depth 1 https://code.videolan.org/videolan/x264.git

# # RUN hg clone https://bitbucket.org/multicoreware/x265
# RUN git clone --depth 1 https://bitbucket.org/multicoreware/x265_git x265

COPY build.sh group_seed_corpus.py $SRC/

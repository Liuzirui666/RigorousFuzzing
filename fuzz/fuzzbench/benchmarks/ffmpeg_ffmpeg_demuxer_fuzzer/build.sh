#!/bin/bash -eux
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

# Disable UBSan vptr since several targets built with -fno-rtti.
export CFLAGS="$CFLAGS -fno-sanitize=vptr"
export CXXFLAGS="$CXXFLAGS -fno-sanitize=vptr"

# Build dependencies.
export FFMPEG_DEPS_PATH=$SRC/ffmpeg_deps
mkdir -p $FFMPEG_DEPS_PATH

export PATH="$FFMPEG_DEPS_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_DEPS_PATH/lib"

# newly added to fix pkg-config issues
# Make pkg-config always see our locally-installed deps.
export PKG_CONFIG_PATH="${FFMPEG_DEPS_PATH}/lib/pkgconfig:${FFMPEG_DEPS_PATH}/lib64/pkgconfig:${FFMPEG_DEPS_PATH}/share/pkgconfig"


cd $SRC

cd $SRC/fdk-aac
autoreconf -fiv
CXXFLAGS="$CXXFLAGS -fno-sanitize=shift-base" \
./configure --prefix="$FFMPEG_DEPS_PATH" --disable-shared
make clean
make -j$(nproc) all
make install

cd $SRC
tar xzf lame.tar.gz
cd lame-*
./configure --prefix="$FFMPEG_DEPS_PATH" --enable-static
make clean
make -j$(nproc)
make install


cd $SRC/libvpx
LDFLAGS="$CXXFLAGS" ./configure --prefix="$FFMPEG_DEPS_PATH" \
    --disable-examples --disable-unit-tests \
    --size-limit=12288x12288 \
    --extra-cflags="-DVPX_MAX_ALLOCABLE_MEMORY=1073741824"
make clean
make -j$(nproc) all
make install

cd $SRC/ogg
./autogen.sh
./configure --prefix="$FFMPEG_DEPS_PATH" --enable-static --disable-crc
make clean
make -j$(nproc)
make install

cd $SRC/opus
./autogen.sh
./configure --prefix="$FFMPEG_DEPS_PATH" --enable-static
make clean
make -j$(nproc) all
make install

cd $SRC/theora
# theora requires ogg, need to pass its location to the "configure" script.
CFLAGS="$CFLAGS -fPIC" LDFLAGS="-L$FFMPEG_DEPS_PATH/lib/" \
    CPPFLAGS="$CXXFLAGS -I$FFMPEG_DEPS_PATH/include/" \
    LD_LIBRARY_PATH="$FFMPEG_DEPS_PATH/lib/" \
    ./autogen.sh
./configure --with-ogg="$FFMPEG_DEPS_PATH" --prefix="$FFMPEG_DEPS_PATH" \
    --enable-static --disable-examples
make clean
make -j$(nproc)
make install

cd $SRC/vorbis
./autogen.sh
./configure --prefix="$FFMPEG_DEPS_PATH" --enable-static
make clean
make -j$(nproc)
make install

cd $SRC/x264
LDFLAGS="$CXXFLAGS" ./configure --prefix="$FFMPEG_DEPS_PATH" \
    --enable-static
make clean
make -j$(nproc)
make install

# cd $SRC/x265/build/linux
# # cmake -G "Unix Makefiles" \
# #     -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
# #     -DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
# #     -DCMAKE_INSTALL_PREFIX="$FFMPEG_DEPS_PATH" -DENABLE_SHARED:bool=off \
# #     ../../source

# rm -rf CMakeCache.txt CMakeFiles

# cmake -G "Unix Makefiles" \
#   -DCMAKE_C_COMPILER="${CC}" \
#   -DCMAKE_CXX_COMPILER="${CXX}" \
#   -DCMAKE_C_FLAGS="${CFLAGS}" \
#   -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
#   -DCMAKE_INSTALL_PREFIX="${FFMPEG_DEPS_PATH}" \
#   -DENABLE_SHARED=OFF \
#   -DENABLE_PIC=ON \
#   ../../source

# make clean
# make -j$(nproc) x265-static
# make install

# # newly added:Create a minimal x265.pc file for ffmpeg's pkg-config dependency check.
# mkdir -p "${FFMPEG_DEPS_PATH}/lib/pkgconfig" "${FFMPEG_DEPS_PATH}/share/pkgconfig"

# cat > "${FFMPEG_DEPS_PATH}/lib/pkgconfig/x265.pc" <<EOF
# prefix=${FFMPEG_DEPS_PATH}
# exec_prefix=\${prefix}
# libdir=\${exec_prefix}/lib
# includedir=\${prefix}/include

# Name: x265
# Description: H.265/HEVC encoder library
# Version: 0
# Libs: -L\${libdir} -lx265
# Libs.private: -ldl -lpthread -lm
# Cflags: -I\${includedir}
# EOF

# cp "${FFMPEG_DEPS_PATH}/lib/pkgconfig/x265.pc" "${FFMPEG_DEPS_PATH}/share/pkgconfig/x265.pc"



# Remove shared libraries to avoid accidental linking against them.

rm $FFMPEG_DEPS_PATH/lib/*.so
rm $FFMPEG_DEPS_PATH/lib/*.so.*

# Build ffmpeg.
cd $SRC/ffmpeg
# PKG_CONFIG_PATH="$FFMPEG_DEPS_PATH/lib/pkgconfig" ./configure \
./configure \
    --cc=$CC --cxx=$CXX --ld="$CXX $CXXFLAGS -std=c++11" \
    --extra-cflags="-I$FFMPEG_DEPS_PATH/include" \
    --extra-ldflags="-L$FFMPEG_DEPS_PATH/lib" \
    --prefix="$FFMPEG_DEPS_PATH" \
    --pkg-config-flags="--static" \
    --enable-ossfuzz \
    --libfuzzer=$LIB_FUZZING_ENGINE \
    --optflags=-O1 \
    --enable-gpl \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libtheora \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-nonfree \
    --disable-shared \
    --disable-vdpau \
    --disable-xlib \
    --disable-libxcb \
    --disable-vaapi \
 
make clean
make -j$(nproc) install

# Download test sampes, will be used as seed corpus.
# DISABLED.
# TODO: implement a better way to maintain a minimized seed corpora
# for all targets. As of 2017-05-04 now the combined size of corpora
# is too big for ClusterFuzz (over 10Gb compressed data).
# export TEST_SAMPLES_PATH=$SRC/ffmpeg/fate-suite/
# make fate-rsync SAMPLES=$TEST_SAMPLES_PATH

# Build ONLY the demuxer fuzzer.
cd "${SRC}/ffmpeg"
fuzzer_name="ffmpeg_DEMUXER_fuzzer"
echo -en "[libfuzzer]\nmax_len = 1000000\n" > "${OUT}/${fuzzer_name}.options"
make tools/target_dem_fuzzer
mv tools/target_dem_fuzzer "${OUT}/${fuzzer_name}"

# Find relevant corpus in test samples and archive them for every fuzzer.
#cd $SRC
#python group_seed_corpus.py $TEST_SAMPLES_PATH $OUT/

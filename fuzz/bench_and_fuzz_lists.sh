#!/usr/bin/env bash

BENCHMARKS=(
  arrow_parquet-arrow-fuzz
  ffmpeg_ffmpeg_demuxer_fuzzer
  grok_grk_decompress_fuzzer
  libhevc_hevc_dec_fuzzer
  libhtp_fuzz_htp
  matio_matio_fuzzer
  openh264_decoder_fuzzer
  php_php-fuzz-parser-2020-07-25
  poppler_pdf_fuzzer
  stb_stbi_read_fuzzer
)

FUZZERS=(
  afl
  aflfast
  aflplusplus
  aflsmart
  honggfuzz
  libfuzzer
  mopt
)

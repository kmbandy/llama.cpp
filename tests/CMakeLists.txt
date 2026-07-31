llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)
llama_build_and_test(test-paged-attn-vk.cpp)

# MAD-392: numeric oracle for the paged decode kernels. Compares the flash-decode
# and WMMA-decode paths against the scalar path elementwise. Driven by
# scratchpad/oracle.sh, which supplies the env that selects each kernel arm.
llama_build_and_test(test-paged-decode-oracle.cpp)

llama_build_and_test(test-weight-pager.cpp)
target_include_directories(test-weight-pager PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(
    test-wp-repack
    test-wp-repack.cpp
    ${CMAKE_SOURCE_DIR}/tools/wp-repack/wp-repack-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/weight-pager/wp-page-catalog.cpp
)
target_include_directories(test-wp-repack PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools/wp-repack
)
target_link_libraries(test-wp-repack PRIVATE ggml)
target_compile_features(test-wp-repack PRIVATE cxx_std_17)
llama_test(test-wp-repack)

llama_build_and_test(test-memory-tier.cpp)
target_include_directories(test-memory-tier PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-wp-resident-experts.cpp)
target_include_directories(test-wp-resident-experts PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-wp-multidevice-partition.cpp)
target_include_directories(test-wp-multidevice-partition PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-wp-size-class-slots.cpp)
target_include_directories(test-wp-size-class-slots PRIVATE ${CMAKE_SOURCE_DIR}/src)

# wp-repack blob-set reader + the catalog remap it drives (no model, no GPU)
add_executable(
    test-wp-blob-index
    test-wp-blob-index.cpp
    ${CMAKE_SOURCE_DIR}/src/weight-pager/wp-page-catalog.cpp
)
target_include_directories(test-wp-blob-index PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/weight-pager
    ${CMAKE_SOURCE_DIR}/common
)
target_link_libraries(test-wp-blob-index PRIVATE llama-common)
target_compile_features(test-wp-blob-index PRIVATE cxx_std_17)
llama_test(test-wp-blob-index)

# expert-index shard builder (synthetic blobs, no model, no GPU)
add_executable(
    test-wp-expert-shard
    test-wp-expert-shard.cpp
    ${CMAKE_SOURCE_DIR}/tools/wp-expert-shard/wp-expert-shard-lib.cpp
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
)
target_include_directories(test-wp-expert-shard PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/wp-expert-shard
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps
)
target_link_libraries(test-wp-expert-shard PRIVATE llama-common)
target_compile_features(test-wp-expert-shard PRIVATE cxx_std_17)
llama_test(test-wp-expert-shard)

# CPU-only expert worker service: synthetic O_DIRECT shard, real TCP framing,
# token batching, weighted multi-expert SwiGLU, and ownership rejection.
add_executable(test-wp-expert-worker test-wp-expert-worker.cpp)
target_include_directories(test-wp-expert-worker PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/tools/wp-expert-worker
    ${CMAKE_SOURCE_DIR}/vendor
)
target_link_libraries(test-wp-expert-worker PRIVATE
    llama-wp-expert-worker-lib
)
target_compile_features(test-wp-expert-worker PRIVATE cxx_std_17)
llama_test(test-wp-expert-worker)

# CPU-only expert dispatcher: four real worker processes over real TCP, two
# disjoint synthetic shards, issue-before-await instrumentation, reduction,
# balancing, coverage rejection, worker rejection, and worker death.
add_executable(
    test-wp-expert-dispatcher
    test-wp-expert-dispatcher.cpp
)
target_include_directories(test-wp-expert-dispatcher PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
    ${CMAKE_SOURCE_DIR}/vendor
)
target_link_libraries(test-wp-expert-dispatcher PRIVATE
    llama-wp-expert-dispatcher-lib
)
target_compile_definitions(test-wp-expert-dispatcher PRIVATE
    WP_EXPERT_WORKER_BIN="$<TARGET_FILE:llama-wp-expert-worker>"
)
target_compile_features(test-wp-expert-dispatcher PRIVATE cxx_std_17)
llama_test(test-wp-expert-dispatcher)

# cross-machine pipeline band helpers (pure, no ggml dependency)
add_executable(
    test-pipeline-band
    test-pipeline-band.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-pipeline.cpp
)
target_include_directories(test-pipeline-band PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(test-pipeline-band PRIVATE cxx_std_17)
llama_test(test-pipeline-band)

# wp-stage-split round-trip: synthetic GGUF -> stage files -> re-read
add_executable(
    test-wp-stage-split
    test-wp-stage-split.cpp
    ${CMAKE_SOURCE_DIR}/tools/wp-stage-split/wp-stage-split-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-pipeline-gguf.cpp
)
target_include_directories(test-wp-stage-split PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools/wp-stage-split
)
target_link_libraries(test-wp-stage-split PRIVATE ggml)
target_compile_features(test-wp-stage-split PRIVATE cxx_std_17)
llama_test(test-wp-stage-split)

# dense spine extraction: synthetic multi-shard GGUF -> one non-expert GGUF
add_executable(
    test-wp-dense-extract
    test-wp-dense-extract.cpp
    ${CMAKE_SOURCE_DIR}/tools/wp-dense-extract/wp-dense-extract-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/weight-pager/wp-router.cpp
)
target_include_directories(test-wp-dense-extract PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools/wp-dense-extract
)
target_link_libraries(test-wp-dense-extract PRIVATE ggml)
target_compile_features(test-wp-dense-extract PRIVATE cxx_std_17)
llama_test(test-wp-dense-extract)

# GLM-DSA dense spine loader: positive external-expert metadata signal,
# missing-dispatch refusal, complete-model regression, and empty pager catalog.
llama_build(
    test-routed-experts-external.cpp
    ${CMAKE_SOURCE_DIR}/tools/wp-dense-extract/wp-dense-extract-lib.cpp
    ${CMAKE_SOURCE_DIR}/src/weight-pager/wp-router.cpp
)
target_include_directories(test-routed-experts-external PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools/wp-dense-extract
)
target_compile_features(test-routed-experts-external PRIVATE cxx_std_17)
llama_test(test-routed-experts-external)

# Phase 2 pipeline protocol framing + HELLO mismatch matrix (no model), plus a
# loopback transport check that a closed peer is an ordinary send error.
add_executable(
    test-pipe-protocol
    test-pipe-protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/pipe-transport.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-pipeline.cpp
)
target_include_directories(test-pipe-protocol PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/pipeline
)
target_compile_features(test-pipe-protocol PRIVATE cxx_std_17)
llama_test(test-pipe-protocol)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# MAD-135: mt:: int4 quant round-trip tests
llama_build_and_test(test-mt-quant.cpp)

# MAD-223 Phase G.2: ml8-4 + f8_e4m3 CPU dequant round-trip tests.
# Needs ggml/src on the include path since ggml-quants.h + ggml-common.h are
# internal (not in the public ggml/include surface).
llama_build_and_test(test-ml8-dequant.cpp)
target_include_directories(test-ml8-dequant PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)

# MAD-223 Phase G.3: ml8 graph-level matmul test (via ggml_ml8_mul_mat helper)
llama_build_and_test(test-ml8-mul-mat.cpp)
target_include_directories(test-ml8-mul-mat PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)

# MAD-223 Phase G.12: ml8 sidecar registry + build_ml8_or_mul_mat dispatch contract.
llama_build_and_test(test-ml8-registry.cpp)
target_include_directories(test-ml8-registry PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MAD-137: mt:: tier primitive unit tests
# These compile a small slice of src/memory-tier/ directly into the test
# binary because the classes aren't exported through the public llama
# API. src/ is added to the include path so internal headers (llama-impl.h)
# resolve.
llama_build_and_test(test-mt-block-pool.cpp ../src/memory-tier/mt-block-pool.cpp)
target_include_directories(test-mt-block-pool PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-mt-block-table.cpp
    ../src/memory-tier/mt-block-table.cpp
    ../src/memory-tier/mt-block-pool.cpp)
target_include_directories(test-mt-block-table PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-mt-block-semantic-index.cpp
    ../src/memory-tier/mt-semantic.cpp)
target_include_directories(test-mt-block-semantic-index PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-mt-tiered-thin.cpp
    ../src/memory-tier/mt-semantic.cpp)
target_include_directories(test-mt-tiered-thin PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MAD-137: paged-cache integration test. Needs a real model (skips
# gracefully when LLAMACPP_TEST_MODELFILE / argv[1] are unset). Links
# the full llama lib because llama_kv_cache_paged is part of it.
llama_build_and_test(test-paged-lifecycle.cpp)
target_link_libraries(test-paged-lifecycle PRIVATE llama)
target_include_directories(test-paged-lifecycle PRIVATE ${CMAKE_SOURCE_DIR}/src)

llama_build_and_test(test-paged-semantic.cpp)
target_link_libraries(test-paged-semantic PRIVATE llama)
target_include_directories(test-paged-semantic PRIVATE ${CMAKE_SOURCE_DIR}/src)

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

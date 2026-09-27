// Standalone check for llama_context_default_params().embd_sparse_outputs == false.
// Built and run by tools/system1-rows/test.sh; not part of the normal build graph
// (LLAMA_BUILD_TESTS is off for this fork's HIP build), compiled ad hoc against the
// already-built libllama, the same way the throwaway spike tools in this repo are built.
#include "llama.h"
#include <cstdio>

int main() {
    llama_context_params p = llama_context_default_params();
    if (p.embd_sparse_outputs != false) {
        fprintf(stderr, "FAIL: llama_context_default_params().embd_sparse_outputs == true, expected false\n");
        return 1;
    }
    printf("OK: llama_context_default_params().embd_sparse_outputs == false\n");
    return 0;
}

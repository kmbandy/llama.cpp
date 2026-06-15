// loadonly.cpp - LOAD ONLY. Calls hipModuleLoad on a .hsaco, reports the
// hipError_t, unloads, and EXITS. There is no hipModuleGetFunction and no
// hipModuleLaunchKernel anywhere in this file - it physically cannot dispatch a
// wave. Used to learn whether ROCr/HIP rejects a kernel descriptor with the
// reserved COMPUTE_PGM_RSRC3 bit 17 set, *without* launching it.
#include <hip/hip_runtime.h>
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <module.hsaco>\n", argv[0]); return 1; }
  const char* path = argv[1];

  int devCount = 0;
  hipError_t e = hipGetDeviceCount(&devCount);
  if (e != hipSuccess || devCount == 0) {
    fprintf(stderr, "no HIP device (err %d)\n", (int)e); return 2;
  }
  hipSetDevice(0);

  printf("LOAD-ONLY test of: %s\n", path);
  hipModule_t mod = nullptr;
  e = hipModuleLoad(&mod, path);
  printf("hipModuleLoad -> hipError_t %d (%s)\n", (int)e, hipGetErrorString(e));
  if (e == hipSuccess) {
    printf("RESULT: LOAD ACCEPTED (module handle %p). NOT launching. Unloading.\n",
           (void*)mod);
    hipModuleUnload(mod);
  } else {
    printf("RESULT: LOAD REJECTED by ROCr/HIP.\n");
  }
  // No GetFunction, no LaunchKernel. Exit.
  return (e == hipSuccess) ? 0 : 3;
}

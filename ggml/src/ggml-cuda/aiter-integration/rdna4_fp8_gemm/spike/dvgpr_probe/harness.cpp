// harness.cpp - load a HIP code object (.hsaco) via the module API, launch the
// dvgpr_probe kernel as a single wave, read back STATUS[30] (DYN_VGPR_EN).
//
//   harness <path-to.hsaco>
//
// SAFE for the UNPATCHED probe.hsaco (control). For the patched binary the
// CONTROLLER runs this deliberately; this program does not distinguish - it just
// loads + launches whatever path it is given.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t _e = (cmd);                                                      \
    if (_e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %d (%s) at %s:%d -> %s\n", (int)_e,            \
              hipGetErrorString(_e), __FILE__, __LINE__, #cmd);                 \
      return 2;                                                                 \
    }                                                                           \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <module.hsaco>\n", argv[0]);
    return 1;
  }
  const char* path = argv[1];

  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  if (devCount == 0) { fprintf(stderr, "no HIP devices\n"); return 2; }
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device 0: %s (gcnArch %s)\n", prop.name, prop.gcnArchName);
  printf("loading module: %s\n", path);

  hipModule_t mod;
  HIP_CHECK(hipModuleLoad(&mod, path));
  printf("hipModuleLoad: OK\n");

  hipFunction_t fn;
  HIP_CHECK(hipModuleGetFunction(&fn, mod, "dvgpr_probe"));
  printf("hipModuleGetFunction(dvgpr_probe): OK\n");

  unsigned* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned)));
  unsigned sentinel = 0xdeadbeefu;
  HIP_CHECK(hipMemcpy(d_out, &sentinel, sizeof(unsigned), hipMemcpyHostToDevice));

  // kernarg buffer: single pointer argument `unsigned* out`
  struct { unsigned* out; } args{ d_out };
  size_t argSize = sizeof(args);
  void* config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,    &argSize,
      HIP_LAUNCH_PARAM_END};

  // one wave: 32-lane block (wave32 on gfx12), single workgroup
  printf("launching dvgpr_probe: grid(1,1,1) block(32,1,1)\n");
  HIP_CHECK(hipModuleLaunchKernel(fn,
                                  1, 1, 1,     // gridDim (blocks)
                                  32, 1, 1,    // blockDim (threads)
                                  0,           // sharedMemBytes
                                  nullptr,     // stream
                                  nullptr,     // kernelParams
                                  config));    // extra
  HIP_CHECK(hipDeviceSynchronize());

  unsigned h_out = 0xffffffffu;
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(unsigned), hipMemcpyDeviceToHost));
  printf("DYN_VGPR_EN = %u\n", h_out);

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipModuleUnload(mod));
  return 0;
}

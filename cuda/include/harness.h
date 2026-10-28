// harness.h — shared self-verification + timing utilities for all CUDA labs.
// Design rule: every lab binary must (1) compute on GPU, (2) compute the same
// thing on CPU in double precision, (3) compare with a stated tolerance, and
// (4) print a PASS/FAIL line plus speedup table. Exit code 0 only on PASS so
// `make test-cuda` can gate CI. Colab-safe: no external deps beyond CUDA rt.
#ifndef CP_CUDA_HARNESS_H
#define CP_CUDA_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ---- CUDA error checking: wrap every runtime call in CK() ----
#define CK(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

// ---- Timing: wall-clock seconds via monotonic clock ----
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Event-based GPU timer (preferred: excludes host launch overhead between
// start/stop events). Usage:
//   GpuTimer t; t.begin(); kernel<<<...>>>(...); float ms = t.end();
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e)); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void begin() { CK(cudaEventRecord(s, 0)); }
    float end() {  // returns elapsed milliseconds
        CK(cudaEventRecord(e, 0));
        CK(cudaEventSynchronize(e));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, s, e));
        return ms;
    }
};

// Benchmark helper: run `fn` warmup times, then iters timed, return avg ms.
template <typename F>
static float bench(F fn, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; i++) fn();
    CK(cudaDeviceSynchronize());
    GpuTimer t;
    t.begin();
    for (int i = 0; i < iters; i++) fn();
    float ms = t.end();
    return ms / iters;
}

// ---- CPU reference timer ----
static double cpu_time_begin(void) { return now_sec(); }
static double cpu_time_ms_since(double t0) { return (now_sec() - t0) * 1e3; }

// ---- Verification ----
// Max absolute difference over n floats vs CPU doubles.
static float max_abs_diff(const float *gpu, const double *ref, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf((float)ref[i] - gpu[i]);
        if (!(d == d)) d = 1e30f; // NaN guard: NaN never passes
        if (d > m) m = d;
    }
    return m;
}

// Relative-L2 error, robust for reductions where abs error grows with n.
static float rel_l2(const float *gpu, const double *ref, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)gpu[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return den == 0 ? (float)sqrt(num) : (float)sqrt(num / den);
}

static int g_pass = 1;
static void check(const char *name, int ok, const char *detail) {
    printf("[%s] %s — %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) g_pass = 0;
}
#define CHECK(name, ok, fmt, ...)                                             \
    do {                                                                      \
        char _b[512];                                                         \
        snprintf(_b, sizeof _b, fmt, __VA_ARGS__);                            \
        check(name, (ok), _b);                                                \
    } while (0)

static int harness_summary(void) {
    printf("%s\n", g_pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return g_pass ? 0 : 1;
}

// ---- Deterministic pseudo-random fill (xorshift64*, same trick as
// llm.c's sampler — see docs/audit/llm-c-01..05). Reproducible across runs.
static unsigned long long xs64_state = 0x9E3779B97F4A7C15ULL;
static inline unsigned long long xs64_next(void) {
    xs64_state ^= xs64_state >> 12;
    xs64_state ^= xs64_state << 25;
    xs64_state ^= xs64_state >> 27;
    return xs64_state * 2685821657736338717ULL;
}
static inline float rand_f(float lo, float hi) {
    double u = (double)(xs64_next() >> 11) / (double)(1ULL << 53);
    return (float)(lo + (hi - lo) * u);
}

// ---- float vs double oracle comparison helpers (used by labs 06+) ----
// check_close: PASS if max|gpu-ref| <= tol * max(1, max|ref|). NaN always fails.
static void check_close(const char *name, const double *ref, const float *gpu,
                        size_t n, float tol) {
    float m = max_abs_diff(gpu, ref, n);
    double rmax = 0;
    for (size_t i = 0; i < n; i++) { double a = fabs(ref[i]); if (a > rmax) rmax = a; }
    float scale = (float)(rmax > 1.0 ? rmax : 1.0);
    char buf[256];
    snprintf(buf, sizeof buf, "max_abs=%.3e tol=%.3e (scale %.1f)", m, tol, scale);
    check(name, m <= tol * scale, buf);
}

// fill_rand: deterministic uniform [-1,1)-ish fill with given seed (labs convention).
static void fill_rand(float *p, size_t n, unsigned long long seed) {
    unsigned long long s = seed ? seed : 1ULL;
    for (size_t i = 0; i < n; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        unsigned long long v = s * 2685821657736338717ULL;
        double u = (double)(v >> 11) / (double)(1ULL << 53);
        p[i] = (float)(2.0 * u - 1.0);
    }
}

// report(): canonical end-of-main used by labs (alias of harness_summary).
static int report(void) { return harness_summary(); }

static void print_gpu_info(void) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("# GPU: %s | SMs=%d | cc%d.%d | %.1f GB\n",
           p.name, p.multiProcessorCount, p.major, p.minor,
           p.totalGlobalMem / 1e9);
}

#endif // CP_CUDA_HARNESS_H

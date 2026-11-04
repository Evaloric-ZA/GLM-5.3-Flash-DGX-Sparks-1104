// Minimal CUPTI activity tracer: LD_PRELOAD, one CSV line per activity record, flushed at buffer completion.
// Build: gcc -shared -fPIC -O2 -o libcuptitrace.so cuptitrace.c -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcupti -lpthread
// Run:   LD_PRELOAD=.../libcuptitrace.so CUPTITRACE_OUT=/path.csv <app>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <cuda.h>
#include <cupti.h>

static FILE *out;
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static int inited;

static const char *kindname(CUpti_ActivityKind k) {
    switch (k) {
    case CUPTI_ACTIVITY_KIND_KERNEL: return "kernel";
    case CUPTI_ACTIVITY_KIND_MEMCPY: return "memcpy";
    case CUPTI_ACTIVITY_KIND_MEMSET: return "memset";
    default: return "other";
    }
}

static void emit(CUpti_Activity *r) {
    unsigned long long s = 0, e = 0;
    unsigned int stream = 0;
    char name[160] = "";
    if (r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        CUpti_ActivityKernel5 *k = (CUpti_ActivityKernel5 *)r;
        s = k->start; e = k->end; stream = k->streamId;
        snprintf(name, sizeof name, "%.155s", k->name ? k->name : "kernel?");
    } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        CUpti_ActivityMemcpy *m = (CUpti_ActivityMemcpy *)r;
        s = m->start; e = m->end; stream = m->streamId;
        snprintf(name, sizeof name, "copyKind%u bytes%llu", m->copyKind, (unsigned long long)m->bytes);
    } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
        CUpti_ActivityMemset *m = (CUpti_ActivityMemset *)r;
        s = m->start; e = m->end; stream = m->streamId;
        snprintf(name, sizeof name, "memset");
    } else {
        return;
    }
    pthread_mutex_lock(&lk);
    fprintf(out, "%llu,%llu,%u,%s,%s\n", s, e, stream, kindname(r->kind), name);
    pthread_mutex_unlock(&lk);
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, unsigned char *buf, size_t size, size_t validSize) {
    (void)ctx; (void)streamId; (void)size;
    CUpti_Activity *rec = NULL;
    CUresult st;
    do {
        st = cuptiActivityGetNextRecord((uint8_t *)buf, validSize, &rec);
        if (st == CUPTI_SUCCESS) emit(rec);
    } while (st == CUPTI_SUCCESS);
    free(buf);
}

static void CUPTIAPI
bufferRequested(unsigned char **buf, size_t *size, size_t *maxNumRecords) {
    *size = 8u << 20;
    *buf = (unsigned char *)malloc(*size);
    *maxNumRecords = 0;
}

static void wallHeader(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    unsigned long long ns = (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
    fprintf(out, "# wall_epoch_ns %llu\n", ns);
    fprintf(out, "start_ns,end_ns,stream,kind,name\n");
}

__attribute__((constructor)) static void init(void) {
    const char *path = getenv("CUPTITRACE_OUT");
    if (!path) return;
    out = fopen(path, "w");
    if (!out) return;
    setvbuf(out, NULL, _IOFBF, 1 << 20);
    if (cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted) != CUPTI_SUCCESS) {
        fprintf(stderr, "cuptitrace: register failed\n"); return;
    }
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    wallHeader();
    inited = 1;
}

__attribute__((destructor)) static void done(void) {
    if (inited) { cuptiActivityFlushAll(0); fflush(out); }
}

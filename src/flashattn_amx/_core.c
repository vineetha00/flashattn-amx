// flashattn.c — FlashAttention forward pass for Apple Silicon CPUs. No CUDA.
//
// Same algorithm as FlashAttention (Dao et al. 2022): tiled attention with an
// online softmax, O(S) extra memory instead of materializing the S x S score
// matrix. The compute-heavy inner products run through Accelerate's SGEMM,
// which executes on the AMX matrix coprocessor; exp() runs through vForce.
// Work is split across (batch * heads * query-blocks) via Grand Central
// Dispatch.
//
// Build:
//   clang -O3 -DACCELERATE_NEW_LAPACK -dynamiclib flashattn.c -o libflashattn.dylib -framework Accelerate
//
// Layout: Q, K, V, O are contiguous float32 tensors of shape (B, H, S, D).

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *Q, *K, *V;
    float *O;
    int B, H, S, D;
    int causal;
    int Br, Bc;
    int nqb;
    float scale;
} fa_args;

static void fa_block(void *ctx, size_t idx) {
    const fa_args *a = (const fa_args *)ctx;
    const int S = a->S, D = a->D, Br = a->Br, Bc = a->Bc;

    const int bh = (int)(idx / a->nqb);
    const int qb = (int)(idx % a->nqb);
    const int q0 = qb * Br;
    const int rows = (q0 + Br <= S) ? Br : (S - q0);

    const float *Qb = a->Q + ((size_t)bh * S + q0) * D;
    const float *Kh = a->K + (size_t)bh * S * D;
    const float *Vh = a->V + (size_t)bh * S * D;
    float *Ob = a->O + ((size_t)bh * S + q0) * D;

    // Scratch: score tile (Br x Bc), accumulator (Br x D), running max + sum.
    float *scratch = (float *)malloc(((size_t)Br * Bc + (size_t)Br * D + 2 * Br) * sizeof(float));
    float *P = scratch;                    // Br x Bc score/prob tile
    float *acc = P + (size_t)Br * Bc;      // Br x D output accumulator
    float *m = acc + (size_t)Br * D;       // running row max
    float *l = m + Br;                     // running row sum

    memset(acc, 0, (size_t)rows * D * sizeof(float));
    for (int i = 0; i < rows; i++) { m[i] = -INFINITY; l[i] = 0.0f; }

    // Causal: keys beyond the last query row of this block never contribute.
    const int kmax = a->causal ? (q0 + rows) : S;

    for (int k0 = 0; k0 < kmax; k0 += Bc) {
        const int cols = (k0 + Bc <= kmax) ? Bc : (kmax - k0);

        // P = scale * Qb @ Kb^T   (AMX via Accelerate)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, cols, D,
                    a->scale, Qb, D, Kh + (size_t)k0 * D, D,
                    0.0f, P, cols);

        // Causal mask inside the diagonal-crossing tile.
        if (a->causal && k0 + cols > q0) {
            for (int i = 0; i < rows; i++) {
                const int qi = q0 + i;
                for (int j = qi + 1 - k0; j < cols; j++)
                    if (j >= 0) P[(size_t)i * cols + j] = -INFINITY;
            }
        }

        // Online softmax update per row.
        for (int i = 0; i < rows; i++) {
            float *Pi = P + (size_t)i * cols;
            float rowmax;
            vDSP_maxv(Pi, 1, &rowmax, cols);
            const float mnew = m[i] > rowmax ? m[i] : rowmax;
            const float neg = -mnew;
            vDSP_vsadd(Pi, 1, &neg, Pi, 1, cols);   // Pi -= mnew
            const int n = cols;
            vvexpf(Pi, Pi, &n);                     // Pi = exp(Pi)  (exp(-inf)=0)
            float rowsum;
            vDSP_sve(Pi, 1, &rowsum, cols);
            const float c = expf(m[i] - mnew);      // rescale old contributions
            l[i] = c * l[i] + rowsum;
            if (c != 1.0f)
                vDSP_vsmul(acc + (size_t)i * D, 1, &c, acc + (size_t)i * D, 1, D);
            m[i] = mnew;
        }

        // acc += P @ Vb   (AMX via Accelerate)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, D, cols,
                    1.0f, P, cols, Vh + (size_t)k0 * D, D,
                    1.0f, acc, D);
    }

    // O = acc / l
    for (int i = 0; i < rows; i++)
        vDSP_vsdiv(acc + (size_t)i * D, 1, &l[i], Ob + (size_t)i * D, 1, D);

    free(scratch);
}

void flash_attention_f32(const float *Q, const float *K, const float *V, float *O,
                         int B, int H, int S, int D, int causal, int Br, int Bc) {
    fa_args a = {
        .Q = Q, .K = K, .V = V, .O = O,
        .B = B, .H = H, .S = S, .D = D,
        .causal = causal, .Br = Br, .Bc = Bc,
        .nqb = (S + Br - 1) / Br,
        .scale = 1.0f / sqrtf((float)D),
    };
    const size_t total = (size_t)B * H * a.nqb;
    dispatch_apply_f(total,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &a, fa_block);
}

// --- Python extension-module stub so setuptools/pip can build and import this
// file as flashattn_amx._core; the real entry point is flash_attention_f32,
// called via ctypes on this module's shared object.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static struct PyModuleDef _core_module = {
    PyModuleDef_HEAD_INIT, "_core",
    "C kernel for flashattn-amx (call via ctypes, not Python).", -1, NULL,
};

PyMODINIT_FUNC PyInit__core(void) { return PyModule_Create(&_core_module); }

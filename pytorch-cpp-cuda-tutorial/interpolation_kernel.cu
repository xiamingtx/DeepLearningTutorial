#include <torch/extension.h>

// scalar_t means using template
// the kernel function always return void
// __global means the function is called by the host (the cpu), but executed on gpu
// supplementary note: __host__ means the function is called and executed on cpu
// __host means the function is called and executed on gpu
template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp)
{
    // 1. compute the id for each thread
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. exclude unnecessary threads (prevent memory leaks)
    if (n >= feats.size(0) || f >= feats.size(2)) return;

    // point -1~1, here we perform normalization
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c;
    feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] + b * feats[n][1][f] + c * feats[n][2][f] + d * feats[n][3][f]) +
                            u * (a * feats[n][4][f] + b * feats[n][5][f] + c * feats[n][6][f] + d * feats[n][7][f]);
}

torch::Tensor trilinear_fw_cu(const torch::Tensor feats, const torch::Tensor points)
{
    // 1. define input shape and init output
    // feats shape: (N, 8, F),  points shape: (N, 3);
    const int N = feats.size(0), F = feats.size(2);
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());

    // 2. calculate threads and blocks count
    // Up to 3 parallel dimensions, and threads are up to 256
    // (here N and F are parallel, and can be allocated according to their proportion, here they are evenly distributed)
    // if only one parallel dimension, can define: const int threads = 256 / const dim3 threads(256)
    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    // 3. allocate kernel
    // AT_DISPATCH_FLOATING_TYPES(_HALF) -> perform floating-point arithmetic (half-precision);
    // supplementary note: AT_DISPATCH_INTEGRAL_TYPES.
    // "trilinear_fw_cu" -> when the kernel launch fails, it will throw an error with this name.
    // scalar_t means using dynamic types, 3 and 2 are the shapes of input (this can be calculated by feats.ndimension)
    // RestrictPtrTraits means tensors don't overlap with each other
    // size_t means what data type we use for these indices, basically just leave it as "size_t"
    // "packed_accessor" conversion is only required for tensors, and we can just throw other types in
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
        // if we know the dtype is float32, we can rewrite as follows:
        // scalar_t can be replaced, and size_t can be removed since it depends on the data type
//         trilinear_fw_kernel<<<blocks, threads>>>(
//             feats.packed_accessor<float, 3, torch::RestrictPtrTraits>(),
//             points.packed_accessor<float, 2, torch::RestrictPtrTraits>(),
//             feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits>()
//         );
    }));

    return feat_interp;
}

template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= feats.size(0) || f >= feats.size(2)) return;

    // point -1~1
    const scalar_t u = (points[n][0] + 1) / 2;
    const scalar_t v = (points[n][1] + 1) / 2;
    const scalar_t w = (points[n][2] + 1) / 2;

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c;

    dL_dfeats[n][0][f] = (1 - u) * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1 - u) * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1 - u) * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1 - u) * d * dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u * d * dL_dfeat_interp[n][f];
}


torch::Tensor trilinear_bw_cu(const torch::Tensor dL_dfeat_interp, const torch::Tensor feats, const torch::Tensor points)
{
    const int N = feats.size(0), F = feats.size(2);

    torch::Tensor dL_dfeats = torch::empty({N, 8, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu",
    ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dfeats;
}
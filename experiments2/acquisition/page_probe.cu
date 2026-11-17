#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <memory>
#include <vector>

static void driver_check(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        const char* reason = nullptr;
        cuGetErrorString(status, &reason);
        TORCH_CHECK(false, operation, ": ", reason ? reason : "CUDA driver error");
    }
}

struct PageAllocation {
    CUcontext context = nullptr;
    CUdeviceptr address = 0;
    size_t bytes = 0, page_bytes = 0, mapped = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
    ~PageAllocation() {
        // Tensor storage owns the mappings until its final reference is released.
        if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        cuCtxSynchronize();
        for (size_t i = 0; i < mapped; ++i) cuMemUnmap(address + i * page_bytes, page_bytes);
        for (auto handle : handles) cuMemRelease(handle);
        if (address) cuMemAddressFree(address, bytes);
        CUcontext previous;
        cuCtxPopCurrent(&previous);
    }
};

// Create separate physical allocation handles, not chunks of one torch allocation.
// CUDA reports allocation granularity; hardware page-table structure is not inferred.
std::tuple<torch::Tensor, int64_t> allocate_pages(int64_t count, int64_t page_bytes, int64_t device) {
    TORCH_CHECK(count > 0 && count <= 4096 && page_bytes > 0, "Invalid page allocation");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    auto allocation = std::make_shared<PageAllocation>();
    driver_check(cuCtxGetCurrent(&allocation->context), "cuCtxGetCurrent");
    CUmemAllocationProp properties = {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = static_cast<int>(device);
    size_t granularity = 0;
    driver_check(cuMemGetAllocationGranularity(&granularity, &properties,
                 CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
    TORCH_CHECK(page_bytes % granularity == 0, "Requested page size is incompatible with device allocation granularity");
    allocation->page_bytes = page_bytes;
    allocation->bytes = count * page_bytes;
    driver_check(cuMemAddressReserve(&allocation->address, allocation->bytes, page_bytes, 0, 0),
                 "cuMemAddressReserve");
    for (int64_t i = 0; i < count; ++i) {
        CUmemGenericAllocationHandle handle;
        driver_check(cuMemCreate(&handle, page_bytes, &properties, 0), "cuMemCreate");
        allocation->handles.push_back(handle);
        driver_check(cuMemMap(allocation->address + i * page_bytes, page_bytes, 0, handle, 0), "cuMemMap");
        allocation->mapped++;
    }
    CUmemAccessDesc access = {};
    access.location = properties.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    driver_check(cuMemSetAccess(allocation->address, allocation->bytes, &access, 1), "cuMemSetAccess");
    auto tensor = torch::from_blob(reinterpret_cast<void*>(allocation->address), {count * page_bytes},
        [allocation](void*) {}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
    tensor.zero_();
    return std::make_tuple(tensor, static_cast<int64_t>(granularity));
}

// One invocation scans all preallocated pages. Output is a checksum, never
// per-page latency. The Python caller records total invocation CUDA-event time.
__global__ void scan_pages(const unsigned char* data, long long size, int stride,
                           int64_t* result) {
    unsigned long long total = 0;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i * stride < size; i += blockDim.x * gridDim.x) {
        total += data[i * stride];
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(result), total);
}

void scan(torch::Tensor input, torch::Tensor output, int64_t stride) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(input.scalar_type() == at::kByte && input.is_contiguous(), "byte buffer required");
    TORCH_CHECK(output.scalar_type() == at::kLong && output.numel() == 1, "int64 scalar required");
    TORCH_CHECK(stride > 0 && input.device() == output.device(), "invalid stride/device");
    c10::cuda::CUDAGuard guard(input.device());
    scan_pages<<<64, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<unsigned char>(), input.numel(), stride, output.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &scan);
    m.def("allocate_pages", &allocate_pages);
}

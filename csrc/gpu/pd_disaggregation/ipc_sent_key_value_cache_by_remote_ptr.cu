#include "helper.h"
#include "fstream"
#include "iostream"
#include "iomanip"
// #define DEBUG_IPC_SENT
// #define DEBUG_IPC_SENT_SYNC_AND_PRINT

template<typename T>
void sent_key_value_by_remote_ptr(
    const T* local_key_tensor_base_ptr,     // gpu ptr
    const T* local_value_tensor_base_ptr,   // gpu ptr
    const int32_t* local_block_ids_ptr, //cpu ptr,
    const int32_t* remote_block_ids_ptr,
    const int32_t block_num,
    const int64_t block_idx_stride,
    const int64_t block_size_byte,
    const int32_t local_device_id,
    const int32_t remote_device_id,
    T* remote_key_tensor_base_ptr,    // gpu ptr
    T* remote_value_tensor_base_ptr,  // gpu ptr
    cudaStream_t stream){
    for(int block_idx=0;block_idx < block_num; ++block_idx){
        const T* local_key_tensor_sent_ptr = local_key_tensor_base_ptr + local_block_ids_ptr[block_idx] * block_idx_stride;
        T* remote_key_tensor_sent_ptr = remote_key_tensor_base_ptr + remote_block_ids_ptr[block_idx] * block_idx_stride;
        #ifdef DEBUG_IPC_SENT
            std::cout<<"remote_key_tensor_sent_ptr:"<<(int64_t)remote_key_tensor_sent_ptr
                     <<" local_key_tensor_sent_ptr:"<<(int64_t)local_key_tensor_sent_ptr
                     <<" local_device_id:" << local_device_id 
                     <<" remote_device_id:" << remote_device_id
                     <<" block_idx_stride:" << block_idx_stride
                     <<" block_size_byte:" << block_size_byte
                     <<" stream: " << stream
                     <<" local_block_ids: " << local_block_ids_ptr[block_idx]
                     <<" remote_block_ids: " << remote_block_ids_ptr[block_idx]
                     <<std::endl;
        #endif
#ifdef DEBUG_IPC_SENT_SYNC_AND_PRINT
            cudaDeviceSynchronize();
            PrintMatrix<T>(reinterpret_cast<const T*>(local_key_tensor_sent_ptr),
                           128 * 1,
                           "ipc_send_src_key.datatxt." + std::to_string(local_device_id),
                           128 * 1);
            cudaDeviceSynchronize();
#endif
#ifndef DEBUG_IPC_SENT_SYNC_AND_PRINT
        cudaMemcpyPeerAsync(
            reinterpret_cast<void*>(remote_key_tensor_sent_ptr), 
            remote_device_id, 
            reinterpret_cast<const void*>(local_key_tensor_sent_ptr), 
            local_device_id, 
            block_size_byte, 
            stream);
#endif
#ifdef DEBUG_IPC_SENT_SYNC_AND_PRINT
        cudaMemcpyPeer(
            reinterpret_cast<void*>(remote_key_tensor_sent_ptr), 
            remote_device_id, 
            reinterpret_cast<const void*>(local_key_tensor_sent_ptr), 
            local_device_id, 
            block_size_byte);
#endif
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }
#ifdef DEBUG_IPC_SENT_SYNC_AND_PRINT
            cudaDeviceSynchronize();
            PrintMatrix<T>(reinterpret_cast<T*>(remote_key_tensor_sent_ptr),
                    128 *  1,
                    "ipc_send_tgt_key.datatxt." + std::to_string(local_device_id),
                    128 *  1);
            cudaDeviceSynchronize();
#endif
        const T* local_value_tensor_sent_ptr = local_value_tensor_base_ptr + local_block_ids_ptr[block_idx] * block_idx_stride;
        T* remote_value_tensor_sent_ptr = remote_value_tensor_base_ptr + remote_block_ids_ptr[block_idx] * block_idx_stride;
#ifdef DEBUG_IPC_SENT
            std::cout<<"remote_value_tensor_sent_ptr:"<<(int64_t)remote_value_tensor_sent_ptr
                     <<" local_value_tensor_sent_ptr:"<<(int64_t)local_value_tensor_sent_ptr
                     <<" local_device_id:" << local_device_id 
                     <<" remote_device_id:" << remote_device_id
                     <<" block_idx_stride:" << block_idx_stride
                     <<" block_size_byte:" << block_size_byte
                     <<" stream: " << stream
                     <<" local_block_ids: " << local_block_ids_ptr[block_idx]
                     <<" remote_block_ids: " << remote_block_ids_ptr[block_idx]
                     <<std::endl;
#endif
#ifdef DEBUG_IPC_SENT_SYNC_AND_PRINT
            cudaDeviceSynchronize();
            PrintMatrix<T>(reinterpret_cast<const T*>(local_value_tensor_sent_ptr),
                           128 * 1,
                           "ipc_send_src_value.datatxt." + std::to_string(local_device_id),
                           128 * 1);
            cudaDeviceSynchronize();
#endif
#ifndef DEBUG_IPC_SENT_SYNC_AND_PRINT
        cudaMemcpyPeerAsync(
            reinterpret_cast<void*>(remote_value_tensor_sent_ptr), 
            remote_device_id, 
            reinterpret_cast<const void*>(local_value_tensor_sent_ptr), 
            local_device_id, 
            block_size_byte, 
            stream);
#endif
#ifdef DEBUG_IPC_SENT_SYNC_AND_PRINT
        cudaMemcpyPeer(
            reinterpret_cast<void*>(remote_value_tensor_sent_ptr), 
            remote_device_id, 
            reinterpret_cast<const void*>(local_value_tensor_sent_ptr), 
            local_device_id, 
            block_size_byte);
        cudaDeviceSynchronize();
#endif
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }
#ifdef DEBUG_IPC_SENT_SYNC_AND_PRINT
        PrintMatrix<T>(reinterpret_cast<T*>(remote_value_tensor_sent_ptr),
                128 *  1,
                "ipc_send_tgt_value.datatxt." + std::to_string(local_device_id),
                128 *  1);
        cudaDeviceSynchronize();
#endif
    }
}
void SentKeyValueByRemotePtr(const paddle::Tensor& local_key_tensor,
                             const paddle::Tensor& local_value_tensor,
                             const paddle::Tensor& local_block_ids,  // cpu
                             const paddle::Tensor& remote_block_ids, // cpu
                             const paddle::Tensor& remote_key_tensor,
                             const paddle::Tensor& remote_value_tensor,
                             const int& block_num,
                             const int& local_device_id,
                             const int& remote_device_id) {
    std::vector<int64_t> cache_key_tensor_shape = local_key_tensor.shape();
    auto cuda_stream = local_key_tensor.stream();
    // const cudaStream_t cuda_stream = *(reinterpret_cast<const cudaStream_t*>(&stream));
    #ifdef DEBUG_IPC_SENT
    std::cout<<"#### 000"<<std::endl;
    #endif

    int32_t total_block_num_local = cache_key_tensor_shape[0];
    int32_t kv_num_head_local = cache_key_tensor_shape[1];
    int32_t block_size_local = cache_key_tensor_shape[2];
    int32_t hidden_size_local = cache_key_tensor_shape[3];

    auto local_block_ids_ptr = local_block_ids.data<int32_t>();   // cpu
    auto remote_block_ids_ptr = remote_block_ids.data<int32_t>(); // cpu
    auto remote_key_ptr =  remote_key_tensor.data<int64_t>()[0];
    auto remote_value_ptr =  remote_value_tensor.data<int64_t>()[0];

    #ifdef DEBUG_IPC_SENT
    std::cout<<"#### 1111"
             << " remote_key_ptr: "<<remote_key_ptr
             << " remote_value_ptr: "<<remote_value_ptr<<std::endl;
    #endif
    int64_t block_idx_stride = kv_num_head_local*block_size_local*hidden_size_local;
    auto local_key_tensor_ptr = local_key_tensor.data();
    auto local_value_tensor_ptr = local_value_tensor.data();
    #ifdef DEBUG_IPC_SENT
    std::cout<<"#### 2222"<<std::endl;
    #endif

    switch (local_key_tensor.type()) {
        case paddle::DataType::BFLOAT16: {
            using dataT=__nv_bfloat16;
            // std::cout<<"#### cache type __nv_bfloat16" << std::endl;
            return sent_key_value_by_remote_ptr<dataT>(
                reinterpret_cast<const dataT*>(local_key_tensor_ptr),
                reinterpret_cast<const dataT*>(local_value_tensor_ptr),
                local_block_ids_ptr,
                remote_block_ids_ptr,
                block_num,
                block_idx_stride,
                block_idx_stride * 2,
                local_device_id,
                remote_device_id,
                reinterpret_cast<dataT*>((void*)remote_key_ptr),
                reinterpret_cast<dataT*>((void*)remote_value_ptr),
                cuda_stream
                );
        }
        case paddle::DataType::FLOAT16: {
            using dataT=half;
            return sent_key_value_by_remote_ptr<dataT>(
                reinterpret_cast<const dataT*>(local_key_tensor_ptr),
                reinterpret_cast<const dataT*>(local_value_tensor_ptr),
                local_block_ids_ptr,
                remote_block_ids_ptr,
                block_num,
                block_idx_stride,
                block_idx_stride * 2,
                local_device_id,
                remote_device_id,
                reinterpret_cast<dataT*>((void*)remote_key_ptr),
                reinterpret_cast<dataT*>((void*)remote_value_ptr),
                cuda_stream
                );
        }
        case paddle::DataType::INT8: {
            using dataT=int8_t;
            return sent_key_value_by_remote_ptr<dataT>(
                reinterpret_cast<const dataT*>(local_key_tensor_ptr),
                reinterpret_cast<const dataT*>(local_value_tensor_ptr),
                local_block_ids_ptr,
                remote_block_ids_ptr,
                block_num,
                block_idx_stride,
                block_idx_stride * 1,
                local_device_id,
                remote_device_id,
                reinterpret_cast<dataT*>((void*)remote_key_ptr),
                reinterpret_cast<dataT*>((void*)remote_value_ptr),
                cuda_stream
                );
        }
        case paddle::DataType::UINT8: {
            using dataT=uint8_t;
            // std::cout<<"#### cache type uint8" << std::endl;
            return sent_key_value_by_remote_ptr<dataT>(
                reinterpret_cast<const dataT*>(local_key_tensor_ptr),
                reinterpret_cast<const dataT*>(local_value_tensor_ptr),
                local_block_ids_ptr,
                remote_block_ids_ptr,
                block_num,
                block_idx_stride,
                block_idx_stride * 1,
                local_device_id,
                remote_device_id,
                reinterpret_cast<dataT*>((void*)remote_key_ptr),
                reinterpret_cast<dataT*>((void*)remote_value_ptr),
                cuda_stream
                );
        }
    }
    // using dataT=std::remove_pointer<decltype(local_block_ids_ptr)>;
}


PD_BUILD_OP(ipc_sent_key_value_cache_by_remote_ptr)
    .Inputs({"local_key_tensor", "local_value_tensor", "local_block_ids", "remote_block_ids", "remote_key_tensor", "remote_value_tensor"})
    .Attrs({ "block_num: int", 
             "local_device_id: int", 
             "remote_device_id: int"})
    .Outputs({"remote_block_ids_out"})
    .SetInplaceMap({{"remote_block_ids", "remote_block_ids_out"}})
    .SetKernelFn(PD_KERNEL(SentKeyValueByRemotePtr));
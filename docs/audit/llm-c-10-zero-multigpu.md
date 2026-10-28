# llm.c Audit 10 — Multi-GPU & ZeRO Sharding (`llmc/zero.cuh`, 597 lines)

Compile-time gates: `MULTI_GPU` (adds NCCL), `USE_MPI` (adds MPI). Include guard LLMC_ZERO_CUH.
`ncclFloatX` = ncclFloat / ncclHalf / **ncclBfloat16** per precision flag (mirrors floatX).
Error macros: `ncclCheck` (prints string + exit), `mpiCheck` (MPI_Error_string buffer 4096).

## MultiGpuConfig struct
```
process_rank, num_processes, local_device_idx
zero_stage                    // 0 disabled, 1 optimizer-state sharding (OSS); 2/3 NOT supported
shard_num_parameters          // total_parameters or total/nproc when stage>=1
#ifdef MULTI_GPU
ncclComm_t nccl_comm; cudaStream_t nccl_stream; cudaEvent_t compute_nccl_sync;
float* unified_buffer;        // cudaMallocManaged 1-float scratch for CPU-side collectives
#endif
```
Global instance `inline MultiGpuConfig multi_gpu_config;` (header-safe single definition).
Macro `printf0(...)` → printf only if process_rank==0 (used throughout trainer prints).

## NCCL unique-ID bootstrap: three init methods (`multi_gpu_config_init`)
1. **"mpi"** (default): requires USE_MPI else exit. MPI_Init; rank/size from COMM_WORLD; `multi_gpu_get_local_device_idx`: hostname (truncated at first '.') djb2-xor hash → Allgather hashes → device idx = #earlier ranks sharing my hostname (NCCL example pattern). Rank0 ncclGetUniqueId → MPI_Bcast.
2. **"tcp"**: rank0 = server on hardcoded port **12345**: socket/bind(SO_REUSEADDR|SO_REUSEPORT)/listen(num_procs-1)/accept all clients/send nccl_id(72B struct)/close. Clients: connect with 5 retries × 2s sleep, recv id. Separate Windows Winsock clone (`get_nccl_id_via_tcp_windows`, WSAStartup/closesocketCheck).
3. **"fs"**: shared filesystem file `<fs_path>/ncclUniqueId.sync`; rank0 writes after getting id; others poll every 1s until fopen succeeds then read. "naive and not 100% robust" comment; clients pre-sleep 2s to avoid reading their own stale file races.
Non-mpi methods take rank/procs from CLI args (-pr/-pn) and `local_device_idx = process_rank % gpus_per_node`.
After id obtained: cudaSetDevice(local_device_idx); ncclCommInitRank; create nccl_stream + disable-timing event; NVTX-name both; mallocManaged unified_buffer(float).
Single-GPU build: prints notice, setDevice(0), identity config.
`multi_gpu_config_free`: destroy comm/stream/event, free buffer, MPI_Finalize.

## Collective helpers
- `multi_gpu_barrier(config)`: allreduce of the 1-float unified buffer + cudaDeviceSynchronize (cheap global sync point; used around checkpoint writes).
- `multi_gpu_cpu_float_sum(value, config)`: put CPU float in managed buffer, ncclAllReduce SUM (ncclFloat), device sync, return sum. Used for val-loss averaging, resume status votes, OOM fallback flags.
- `set_zero_configs(config, zero_stage, total_parameters)`: enables stage 1 only if `total_parameters % num_processes == 0` else silently downgrades to 0 with table print; stages ≥2 always rejected ("not yet supported"). Sets shard_num_parameters accordingly.
- `ShardInfo {ptrdiff_t offset; size_t size;} multi_gpu_get_shard_offset(elements, config, shard_at_stage)`: if stage ≥ threshold → contiguous block assignment `{rank*(n/N), n/N}` (hard exit if n%N≠0); else whole tensor {0,n}. Called with 1 for optimizer/grad-norm sharding.

## `multi_gpu_async_reduce_gradient<N>(pointers[N], sizes[N], config, compute_stream)`
Template over array size (type-safe pairing of pointer/size arrays). No-op if single process.
Pattern: record event on compute stream → nccl_stream waits (no host sync; CPU can keep enqueueing).
Inside `ncclGroupStart/End` (fuses N collectives into one kernel chain):
- zero_stage==0 (DDP): `ncclAllReduce(ptr, ptr, size, ncclFloatX, ncclAvg)` in-place per grad tensor.
- zero_stage==1: assert size%N==0; `ncclReduceScatter(ptr, ptr + rank*shard, shard_size, ncclFloatX, ncclAvg)` — each rank ends owning averaged slice (sendbuff==recvbuff base allowed since overlapping? note recv is a sub-range of send; NCCL handles it as long as input fully consumed before overwrite — standard trick).
Called from gpt2_backward_and_reduce on last micro-step (doc 07 §7) for the 12 per-layer grads; wte/wpe/lnf grads reduced separately via plain ncclAllReduce in backward tail; ZeRO-1 allgather after adamw update in gpt2_update.

## Interaction summary with trainer (doc 07)
- DataLoader shards batches by process_rank (slab offset), EvalLoader partitions examples round-robin-block style.
- Grad flow DDP: full grads allreduce avg → identical updates each rank (no comms in optimizer).
- Grad flow ZeRO-1: reduce-scatter → rank updates its 1/N param+master+m+v slice → allgather updated params → forward consistent. Checkpoints save per-rank state files `state_%08d_%05d.bin` (only own shard!). Resume asserts same num_processes/rank.
- grad_norm under ZeRO-1: local shard norms then ncclAllReduce SUM of squared norm before sqrt.

# LLAMA.CPP 解读
 LLAMA.CPP或者说GGML这个项目对LLM的处理大致可以分成以下三个阶段：
 - Convert huggingface pytorch model into `.gguf` format.
 - Quantize `.gguf` model
 - Do inference with `.gguf` model.

## Convert
 第一步其实没有什么好说的，convert程序分别读取配置文件和tokenizer和model的binary文件，先把hyparam的参数写入文件头，再依次讲tokenizer和model的weights写入文件，值得注意的是可能是为了加快读效率，作者在写入tensor的时候按照32的整数倍进行了padding操作。

## Quantize
 第二步开始时注意到作者有一个init的操作`llama_backend_init(false)`, 截取其中的`ggml_init`来单独看一下：
 ```cpp
struct ggml_context * ggml_init(struct ggml_init_params params) {
    // make this function thread safe
    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            ggml_fp16_t ii;
            for (int i = 0; i < (1 << 16); ++i) {
                uint16_t ui = i;
                memcpy(&ii, &ui, sizeof(ii));
                const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
                table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
                table_exp_f16[i]  = GGML_FP32_TO_FP16(expf(f));
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }
    ...
    }
 ```
 首先`ggml_critical_section_start`实现了一个简单的自旋锁，使得每次只有一个thread去操作后面的内容。(没有发现这个实例起作用的地方)
 ```cpp
 // barrier via spin lock
inline static void ggml_critical_section_start(void) {
    int processing = atomic_fetch_add(&g_state_barrier, 1);// return the previous value.

    while (processing > 0) {
        // wait for other threads to finish
        atomic_fetch_sub(&g_state_barrier, 1);
        sched_yield(); // let cpu to consider schedule other threads to run.
        processing = atomic_fetch_add(&g_state_barrier, 1);
    }
}
 ```
我们发现这里对每个uint16的整形值都预计算了他们可能会用到的数学运算的值(比如：gelu和silu)，猜测是推理过程中就直接从这里取现有的值。事实上这些array的大小也并没有很大，以其中最大的`table_f32_f16`为例, 全为float32(4bytes)，那么这样一个数组总占用在2^16 * 4 = 256kB.
这个过程剩下的功能就是初始化了一个`g_state`的context的管理结构：
```cpp
struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];//64
    struct ggml_numa_nodes numa;
};
...
            g_state = (struct ggml_state) {
                /*.contexts =*/ { { 0 } },
                /*.numa =*/ {
                    .n_nodes = 0,
                    .total_cpus = 0,
                },
            };
```
`ggml_init`这个函数会非常有用，他会在thread safe的情况下去用现有的context pool中的没有被使用的ggml_context_container来承载新申请的context的内存。ggml_context的内存不会提前申请。
回到quantize的主函数，init之后就是常见的对args的parse的操作。llama.cpp对model进行quant的入口在`llama_model_quantize -> llama_model_quantize_internal`里面:
```cpp
static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {}
```
默认用上全部的核来跑：
```cpp
    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }
```
### llama_model_loader
用`llama_model_loader`读.bin文件头, 没有使用mmap，用kv对管理一些hyparam, 主要功能是将数据从file读进一个`gguf_context ctx`。
#### gguf_init_from_file
```cpp
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    FILE * file = fopen(fname, "rb");
    ...
        struct gguf_context * ctx = GGML_ALIGNED_MALLOC(sizeof(struct gguf_context));

    // read the header
    {
        ctx->header.magic = magic;

        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ...
    }

    // read the kv pairs
    {
        ctx->kv = malloc(ctx->header.n_kv * sizeof(struct gguf_kv));

        for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);
        }
    }

    // read the tensor infos
    {
        ctx->infos = malloc(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));

        for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);
        }
    }
    ...
    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }
```
OK, 到目前为止，对hyparams的预操作已经结束，同时这也是真正开始处理tensor数据的开始。首先通过加载的tensor_info，计算总共需要的size，接着再初始化tensor。不过这里作者也并不想在这一步初始化所有的tensor的data，也是套一层壳，控制这一行为的操作的no_alloc是true。
```cpp
    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        // the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (ctx->header.n_tensors    )*ggml_tensor_overhead() :
            (ctx->header.n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = NULL;

        // 不太确定，这里如果成功的话，一次性把所有的tensor都读进来？
        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);
            ...

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

   
}
```
此处只申请了tensor的baseinfo信息大小内存。不过这里的ctx->size不是某一个tensor的size，而是之前算出来的tensor(info)的总size。接着再看创建tensor的过程（之前就算全读进来也只是一堆raw data）:
```cpp
        // create the tensors
        for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
            const int64_t ne[GGML_MAX_DIMS] = {
                ctx->infos[i].ne[0],
                ctx->infos[i].ne[1],
                ctx->infos[i].ne[2],
                ctx->infos[i].ne[3],
            };

            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, ctx->infos[i].type, ctx->infos[i].n_dims, ne);

            ok = ok && cur != NULL;

            ggml_set_name(cur, ctx->infos[i].name.data);

            if (!ok) {
                break;
            }

            // point the data member to the appropriate location in the binary blob using the tensor infos
            if (!params.no_alloc) {
              //cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset; // offset from start of file
                cur->data = (char *) data->data + ctx->infos[i].offset;               // offset from data
            }
        }
```
这里的`ggml_new_tensor`的过程值得说一下。
##### ggml_new_tensor
```cpp
static struct ggml_tensor * ggml_new_tensor_impl(
        struct ggml_context * ctx,
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t       * ne,
        struct ggml_tensor  * view_src,
        size_t                view_offs) {

    assert(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = ggml_type_size(type)*(ne[0]/ggml_blck_size(type));
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == NULL || data_size + view_offs <= ggml_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    if (view_src == NULL && !ctx->no_alloc) {
        if (ctx->scratch.data != NULL) {
            // allocate tensor data in the scratch buffer
            if (ctx->scratch.offs + data_size > ctx->scratch.size) {
                GGML_PRINT("%s: not enough space in the scratch memory pool (needed %zu, available %zu)\n",
                        __func__, ctx->scratch.offs + data_size, ctx->scratch.size);
                assert(false);
                return NULL;
            }

            data = (char * const) ctx->scratch.data + ctx->scratch.offs;

            ctx->scratch.offs += data_size;
        } else {
            // allocate tensor data in the context's memory pool
            obj_alloc_size = data_size;
        }
    }
    ```
  因为标识了no_alloc，这里还是不会为tensor的array申请内存，不过可以推测，如果这里要申请的话，ctx->scratch.data 起了内存池的作用（一次申请一大块，可以顺序读写，速度快），限度内的申请会直接复用，超过的就额外申请。
    ```cpp
    struct ggml_context {
        size_t mem_size;
        void * mem_buffer;
        bool   mem_buffer_owned;
        bool   no_alloc;
        bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

        int    n_objects;

        struct ggml_object * objects_begin;
        struct ggml_object * objects_end;

        struct ggml_scratch scratch;
        struct ggml_scratch scratch_save;
    };

    struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);

    // TODO: for recoverable errors, we would need to free the data allocated from the scratch buffer here

    struct ggml_tensor * const result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);
    ```
  先向ctx注册这个tensor目前的信息，注册成功再向ctx的memory实际实例化这一块内存。看一下`ggml_new_object`注册的过程：
  ```cpp
  static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed, ctx->mem_size);
        assert(false);
        return NULL;
    }

    *obj_new = (struct ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    ggml_assert_aligned(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    return obj_new;
}

  ```
  其实借用了链表的方式来保存所有已注册的对象，整个context只记录这个list的头和尾，要加新的tensor就直接插入更新就好了。看起来是在申请内存，其实也顺便把tensor信息给注册到list里面了。
    
  ```cpp
    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.backend      =*/ GGML_BACKEND_CPU,
        /*.n_dims       =*/ n_dims,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.is_param     =*/ false,
        /*.grad         =*/ NULL,
        /*.src          =*/ { NULL },
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data, // still NULL Here
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //ggml_assert_aligned(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    // nb[0] = sizeof(type)
    // nb[1] = nb[0]   * ne[0] + padding
    // nb[i] = nb[i-1] * ne[i-1]
    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}
```
stride的作用类似于torch [tensor.as_strided()](https://pytorch.org/docs/stable/generated/torch.as_strided.html#torch-as-strided), 对于同一片tensor的内存区域，可以通过该方法来将其中一块的内存视为一个尺寸可以随时增长的tensor来读写。
后面返回gguf_init_from_file, 就没有什么好说的，目前ctx保存了除开tensor data以外的所有model信息。
再回到`llama_model_loader`的定义，后面根据ctx统计各种type的param的数量，查找数量最多的type, 以此来guess这个model本来的type（`ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);
`来标记这个ftype是猜出来的）。

至此`llama_model_loader`的初始化过程结束，我们得到了整个model的meta data。

回到`llama_model_quantize_internal`：
```cpp
    llm_load_arch(*ml, model);  //     LLM_ARCH_LLAMA,
                                // LLM_ARCH_FALCON,
                                // LLM_ARCH_GPT2,
                                // LLM_ARCH_GPTJ,
                                // LLM_ARCH_GPTNEOX,
                                // LLM_ARCH_MPT,
    llm_load_hparams(*ml, model, 0, 0, 0);
```

`llm_load_hparams`加载一些model的参数从loader到model：
```cpp
    // get general kv
    GGUF_GET_KEY(ctx, model.name, gguf_get_val_str, GGUF_TYPE_STRING, false, kv(LLM_KV_GENERAL_NAME));

    // get hparams kv
    GGUF_GET_KEY(ctx, hparams.n_vocab,        gguf_get_arr_n,   GGUF_TYPE_ARRAY,   true, kv(LLM_KV_TOKENIZER_LIST));
    GGUF_GET_KEY(ctx, hparams.n_ctx_train,    gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_CONTEXT_LENGTH));
    GGUF_GET_KEY(ctx, hparams.n_embd,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_EMBEDDING_LENGTH));
    GGUF_GET_KEY(ctx, hparams.n_ff,           gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_FEED_FORWARD_LENGTH));
    GGUF_GET_KEY(ctx, hparams.n_head,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
    GGUF_GET_KEY(ctx, hparams.n_layer,        gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_BLOCK_COUNT));

    // n_head_kv is optional, default to n_head
    ...
    // rope_freq_base

    // rope_freq_scale (inverse of the kv) is optional

    // sanity check for n_rot (optional)

    // arch-specific KVs
```
Llama.cpp里面用上了位置内部插值参数，这个在拓展ctx长度中使用很关键，参考[苏神的文章](https://kexue.fm/archives/9675/comment-page-1#%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)
```cpp
float ropebase = 10000.0f;//位置编码的base
```
继续往下看，从这里开始正式从文件/mmap中读取tensor到data数据，并且每读一份就quant这一份再传回去，不会一次性全部读进来：
```cpp
    std::vector<uint8_t> read_data;
    std::vector<uint8_t> work;
    // placeholder for the meta data
    ::zeros(fout, meta_size);

    for (int i = 0; i < ml->n_tensors; ++i) {
        struct ggml_tensor * tensor = ml->get_tensor_meta(i);

        const std::string name = ggml_get_name(tensor);

        read_data.resize(ggml_nbytes(tensor));
        tensor->data = read_data.data();
        ml->load_data_for(tensor);
```

这里使用ggml_nbytes算出这个tensor->data需要多大的内存，将read_data resize到这个大小，最后`load_data_for`从文件/mmap中读出对应的size的数据。

```cpp
    void load_data_for(struct ggml_tensor * cur) const {
        const size_t offs = file_offset(ggml_get_name(cur));

        if (use_mmap) {
            cur->data = (uint8_t *) mapping->addr + offs;
        } else {
            file.seek(offs, SEEK_SET);
            file.read_raw(cur->data, ggml_nbytes(cur));
        }
    }
```
看到这里也许可以多多少少体会到一点，整个项目的内存管理，读写管理做的非常理想化，几乎你想读的任何一个obj都会有对应的meta data来指明他的大小、起始位置，按照这些记录读就好了。

## GGML_QUANT
对于一个给定的从文件读出的tensor，只要是被认为不是被quantized过的数据类型，都会强制转回fp32:
```cpp
            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_convert_tensor_internal(tensor, f32_conv_buf, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }
```
来看一下`llama_convert_tensor_internal`这个函数：
```cpp
    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }
```
`ggml_internal_get_type_traits` 从`type_traits`中取出对应type的一些信息，以Q4_0为例：
```cpp
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float               = quantize_row_q4_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_0_reference,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
```
甚至还记录从q4_0转回float32的对应方法。dequantize_row_q4_0。我们先看一下quantize_row_q4_0_reference的实现：
```cpp
// reference implementation for deterministic creation of model files
static void quantize_row_q4_0_reference(const float * restrict x, block_q4_0 * restrict y, int k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;// 32为一个block

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) { // 记录每一个block中的amax和max
                amax = fabsf(v);
                max  = v;
            }
        }
        // 算出scale factor，将每个block归一化到int4的范围内
        const float d  = max / -8; // INT4 [-2^3, 2^3-1] 为什么要是负的?
        const float id = d ? 1.0f/d : 0.0f; // 如果d是0，不可被除。id= -8.0f / max

        y[i].d = GGML_FP32_TO_FP16(d);// #define GGML_FP32_TO_FP16(x) (x)
                                    // 损失后13位实部和后三位指数部
                                    // 3.75:
                                    // float32: 0 10000000 11100000000000000000000
                                    // float16: 0 10000    1110000000

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;// belongs to (float)[-2^3, 2^3-1]
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));// (float)[0.5f, 15.5f] -> int8_t [0, 15], 截断任何小数部分
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));
            // 8.5: -8~ -7.5 -> 0, -7.5~ -6.5 -> 1 ... 6.5~7 -> 15
            // 8: -8~ -7 -> 0, -7~ -6 -> 1 ... 6~7 -> 14 动态范围小了

            // [00000000, 00001111]
            // 用一个int8的前后4位来同时储存两个int4
            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
```
关于量化的介绍->https://zhuanlan.zhihu.com/p/645362500
TODO:求一下截断造成的最大的误差
delta(x_qs) <= 0.5, delta(x_01)= delta(x_qs)/ | id | = 0.5 * max / 8.0 = 0.0625max
最大误差大概在6.25%

顺便看一下dequantize的实现：
```cpp
static void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int k) {
    static const int qk = QK4_0; // 32

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d); // ((float) (x))

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8; // 0x0F: 00001111，取后4位再减8，相当于上面的逆向操作
            const int x1 = (x[i].qs[j] >>   4) - 8;// 取前四位

            y[i*qk + j + 0   ] = x0*d;// restore
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

```
再回到`llama_convert_tensor_internal`。单线程的话在这里就直接转回fp32。
```cpp
    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype.to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ASSERT(false); // unreachable
        }
        return;
    }
```
看一下多线程的实现，这里做的还是dequantize
```cpp
    GGML_ASSERT(nelements % block_size == 0);
    auto nblocks = nelements / block_size; // 一个Q4_0的block可以含有32个elements，这里算总共有多少blocks
    auto blocks_per_thread = nblocks / nthread; // 每个线程分配的blocks
    auto spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count，余数

    std::vector<std::thread> workers;
    for (auto tnum = 0, in_buff_offs = 0, out_buff_offs = 0; tnum < nthread; tnum++) {
        auto thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        auto thr_elems = thr_blocks * block_size; // number of elements for this thread
        auto thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };// 定义线程的任务原型，分块读入写出
        workers.push_back(std::thread(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems));
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & worker : workers) {
        worker.join();// 等待任务完成
    }
```
回到主函数`llama_model_quantize_internal`
```cpp
            work.resize(nelements * 4); // upper bound on size, 上限是全是4bytes的float
            new_data = work.data();
            std::vector<int64_t> hist_cur(1 << 4, 0);

            static const int chunk_size = 32 * 512; //怎么得出来的？还是人为规定的？
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
```
看一下多线程quant的逻辑。
```cpp
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements]() {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) {
                            local_hist.resize(hist_cur.size(), 0);
                        }
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if ((int) workers.size() < nthread_use - 1) {
                    workers.resize(nthread_use - 1);
                }
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it] = std::thread(compute);
                }
                compute();//剩下的在主线程处理，有点炫技。
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it].join();
                }
```
这里没有显式地分配每一个thread具体take的范围，而是去让他们自己去抢占。这是因为`const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;`这一步导致我们没法确定nchunk是否大于nthread，
如果大于的话，那么我们去schedule这些thread就会变成一个非常头疼的问题，与其头疼，不如直接将问题丢给worker自己去调度，每个线程每次只处理chunk_size大小的elements，每次处理都会有记录，处理完了再从共享的变量counter中读取下次处理的起始点。

至此quant的主要逻辑就结束了，剩下的就是一些更新tensor的meta data的记录信息，写入.bin文件的工作。当所有的tensor都这样写过一遍后，再回到文件头去更新model的meta data，最后打印sumary。



# Some hints
 - (void) x, 要有这个空操作的原因是，GGML_PRINT_DEBUG这个宏在release的时候会关掉，届时编译器会放出warning。
 ```cpp
 void myFunction(int x) {
    (void)x; // 告诉编译器参数 x 未使用
    // 这里可以没有任何与 x 相关的代码
}
 ```
 - linux平台用unsigned short来储存fp16
 - 用数组和enum类型来搭配做flag的设计极其优雅, GGML_OP_COUNT刚好是enum类型的最后一项：
 ```cpp
 static bool GGML_OP_HAS_INIT    [GGML_OP_COUNT] = { 0 };
 static bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = { 0 };

 static void ggml_setup_op_has_task_pass(void) {
    {   // INIT
        bool * p = GGML_OP_HAS_INIT;

        p[GGML_OP_ACC                    ] = true;
        p[GGML_OP_MUL_MAT                ] = true;
        p[GGML_OP_OUT_PROD               ] = true;
        p[GGML_OP_SET                    ] = true;
        p[GGML_OP_GET_ROWS_BACK          ] = true;
        p[GGML_OP_DIAG_MASK_INF          ] = true;
        p[GGML_OP_DIAG_MASK_ZERO         ] = true;
        p[GGML_OP_CONV_1D                ] = true;
        p[GGML_OP_CONV_2D                ] = true;
        p[GGML_OP_CONV_TRANSPOSE_2D      ] = true;
        p[GGML_OP_FLASH_ATTN_BACK        ] = true;
        p[GGML_OP_CROSS_ENTROPY_LOSS     ] = true;
        p[GGML_OP_ADD_REL_POS            ] = true;
    }

    {   // FINALIZE
        bool * p = GGML_OP_HAS_FINALIZE;

        p[GGML_OP_CROSS_ENTROPY_LOSS     ] = true;
    }
}
 ```
 - 复合字面量这没看懂
 ```cpp
    g_state = (struct ggml_state) {
    /*.contexts =*/ { { 0 } },
    /*.numa =*/ {
        .n_nodes = 0,
        .total_cpus = 0,
    },
};
 ```
 - 算一下ggml_tensor的size: 289bytes，对最高的成员8bytes对齐，实际占用是304bytes？不重要
 ```cpp
     struct ggml_tensor {
        enum ggml_type    type; // 4
        enum ggml_backend backend; // 4

        int     n_dims; // 4 + pad 4
        int64_t ne[4]; // number of elements 4 * 8
        size_t  nb[4]; // stride in bytes: 4*8
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op; // 4 + pad 4

        // op params - allocated as int32_t for alignment
        int32_t op_params[8]; // 8 * 4

        bool is_param; // 1+ pad 7

        struct ggml_tensor * grad; // 8
        struct ggml_tensor * src[6]; // 6*8

        // performance
        int     perf_runs; // 4 + pad 4
        int64_t perf_cycles; // 8
        int64_t perf_time_us; // 8

        struct ggml_tensor * view_src; // 8
        size_t               view_offs; // 8

        void * data; // 8

        char name[64]; // 1 * 64

        void * extra; // extra things e.g. for ggml-cuda.cu 8

        char padding[4]; // 4 + pad 4
    };
 ```
 - ggml_nbytes 的逻辑没有看懂：
 ```cpp
 size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes = tensor->ne[0]*tensor->nb[0]/ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
    }
    return nbytes;
}
 ```
做一个实验看一下, 以这个tensor为例，ggml的tensor NE记录是倒着的，第一维记录的是tensor的最后一维的size：
```log
NB elements: [2, 8192, 90177536, 90177536, ]
NE elements: [4096 11008 1 1 ]
```
正常算法应该是直接4096*11008，按照ggml的算法呢？4096*2+(11008-1)*8192,因为一般成立nb[0]*ne[0]=nb[1],所以两式实际上相等。不是很清楚这样算的原理
考虑一下今天tensor.as_strided的情况，一个tensor block [4, 4], stride[4, 1], 现在取size [2, 2], 那么有：
```log
a:
NB elements: [1, 4]
NE elements: [4, 4]

b:
NB elements: [1, 4]
NE elements: [2, 2]
```
用ggml的算法来算a和b的Nbytes(假设都是unsigned char):
a: 1*4+3*4=16
b: 1*2+1*4=6
b似乎是b横跨的所有size（包括无意义的0）。不确定。

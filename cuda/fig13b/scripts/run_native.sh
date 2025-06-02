install_path=/bigdisk/install/
models_dir=/bigdisk/models/cuda

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig13b/raw

rm ${RESULT_DIR}/cuda_ms_lns2.txt

num_streams=15

export LD_LIBRARY_PATH=${install_path}/lib:$LD_LIBRARY_PATH

for i in {1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000,2222,2500,2857,3333,4000,5000,6667,10000,20000,40000,80000,160000}; do
    "${install_path}"/bin/tvm_direct_multistream \
        --iat $i \
        --ln_sigma 2.0 \
        --start_record_num 0 \
        --seed 1 \
        --prefix "${RESULT_DIR}/cuda_ms" \
        --iat_n \
        --iat_g \
        --ln_sigma_n \
        --num_jobs 3000 \
        --concurrency $num_streams \
        "${models_dir}/mobilenetv2-7-cuda-pack.so" 0.257 36 \
        "${models_dir}/densenet-9-cuda-pack.so" 0.0706 10 \
        "${models_dir}/googlenet-9-cuda-pack.so" 0.0546 8 \
        "${models_dir}/inception_v3-cuda-pack.so" 0.0138 2 \
        "${models_dir}/resnet18-v2-7-cuda-pack.so" 0.272 38 \
        "${models_dir}/resnet34-v2-7-cuda-pack.so" 0.168 24 \
        "${models_dir}/resnet50-v2-7-cuda-pack.so" 0.0745 10 \
        "${models_dir}/squeezenet1.1-7-cuda-pack.so" 0.0894999999999999 13
done
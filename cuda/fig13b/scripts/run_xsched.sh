models_dir=/bigdisk/models/cuda

TESTCASE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../ && pwd -P)
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../../../ && pwd -P)
RESULT_DIR=${ROOT}/results/fig13b/raw

rm ${RESULT_DIR}/xsched_lns2.txt

export LD_LIBRARY_PATH=/bigdisk/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/xsched-artifacts/sys/xsched/output/lib:$LD_LIBRARY_PATH
export XSCHED_POLICY=KEDF

num_threads=8
deadline=30
for i in {1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000,2222,2500,2857,3333,4000,5000,6667,10000,20000,40000,80000,160000}; do
    $TESTCASE_ROOT/build/xsched_tvm \
        --iat $i \
        --ln_sigma 2.0 \
        --seed 1 \
        --output_path "${RESULT_DIR}/xsched_lns2.txt" \
        --num_jobs 3000 \
        --concurrency $num_threads \
        --deadline $deadline \
        "${models_dir}/mobilenetv2-7-cuda-pack.so" 0.257 \
        "${models_dir}/densenet-9-cuda-pack.so" 0.0706 \
        "${models_dir}/googlenet-9-cuda-pack.so" 0.0546 \
        "${models_dir}/inception_v3-cuda-pack.so" 0.0138 \
        "${models_dir}/resnet18-v2-7-cuda-pack.so" 0.272 \
        "${models_dir}/resnet34-v2-7-cuda-pack.so" 0.168 \
        "${models_dir}/resnet50-v2-7-cuda-pack.so" 0.0745 \
        "${models_dir}/squeezenet1.1-7-cuda-pack.so" 0.0894999999999999 
done


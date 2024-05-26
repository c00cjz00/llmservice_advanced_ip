# 目前位置以及輸入值argv $1
# Example: ./01-init.sh 203.145.221.127 cpu
[[ -z "$1" ]] && { echo "Parameter 1 is empty" ; exit 1; }
[[ -z "$2" ]] && { echo "Parameter 2 is empty" ; exit 2; }
INTERNET_IP=$1
IS_GPU=$2
mypwd=$(pwd)

# 更換IP
cd ${mypwd}
cp website/index_demo.html website/index.html
sed -i "s/nchcip/${INTERNET_IP}/g" website/index.html

# CPU/GPU
if [[ "$IS_GPU" == "gpu" ]] 
then
	cp compose/docker-compose_gpu.yml docker-compose.yml
else
	cp compose/docker-compose_cpu.yml docker-compose.yml
fi

# 建立目錄
cd ${mypwd}
mkdir -p storage/anythingllm_data
mkdir -p storage/anythingllm_hotdir 
mkdir -p storage/anythingllm_outputs 
mkdir -p storage/ollama_data 
mkdir -p storage/openwebui_data 
mkdir -p storage/hf_cache 
mkdir -p storage/factory_data storage/factory_saves storage/factory_cache storage/jupyter_data
mkdir -p storage/output
mkdir -p storage/postgres
mkdir -p storage/pgadmin
cp factory/data/dataset_info.json storage/factory_data/
cp factory/data/identity.json storage/factory_data/
rsync -avHS notebook/ storage/jupyter_data/notebook
 
if ! [ -f ./storage/anythingllm_env.txt ]; then
	touch ./storage/anythingllm_env.txt
fi 

if ! [ -f ./.env ]; then
	cp ./env.sample .env
fi 
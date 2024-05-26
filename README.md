# nchc service
```
# AUTO RUN
./03-init_argv.sh $clouflarekey nctu 140.110.139.64
```
## DOCKER 安裝
- https://hackmd.io/@whYPD8MBSHWRZV6y-ymFwQ/Hk8pJ95eA
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


## Step1: GIT 下載套件
```
git clone https://github.com/c00cjz00/llmservice_basic_demo
```

## Step 2: 編輯env.sample
```
vi  env.sample 
```

## Step3: 確認您的IP以及是否有GPU, 並執行以下指令
- ./01-init.sh $IP [cpu/gpu]
```
./01-init.sh 203.145.221.127 cpu
``` 

## Step 4: 啟動服務
```
docker compose up -d 
```

## Step 5: 關閉服務
```
docker compose down 
```

## Step 6: 服務網址 
- openwebui: http://$IP:8080
- ollama api: http://$IP:11434
- factory: http://$IP:7860
- notebook: http://$IP:9999
- anythingllm: http://$IP:3001
- litell api: http://$IP:4000
- litell ui: http://$IP:4000/ui/
- postgres api: http://$IP:5432
- pgadmin4 api: http://$IP:5080
- portal: http://$IP:80

## Step 6.3: 內部服務器
- ollama api: http://host.docker.internal:11434
- postgres api: http://host.docker.internal:5432


## 注意事項: Ollama Template, 記憶體太小請修正 num_ctx=2048
```
FROM ./taide-8b-a.3-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER num_keep 24
PARAMETER num_ctx 4096
```

## 注意事項:  GPU
```
sudo nvidia-smi -pm 1
sudo nvidia-smi -e 0
```

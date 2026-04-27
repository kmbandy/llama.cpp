Here is what I use to start the server:

llama-server-feature \
          --model ~/models/MiniMax-M2.7-UD-IQ3_S-00001-of-00003.gguf \
          -ngl 999 \
          --device ROCm0 \
          --ctx-size 196608 \
          --cache-type-k turbo4 \
          --cache-type-v turbo4 \
          --flash-attn on \
          --metrics \
          --port 8080 \
          --host 0.0.0.0 \
          --no-mmap \
          --kv-tiered 12.5,50,37.5 \
          --tier-ssd-path ~/kv-cold \
          --tier-eviction-policy 3 \
          --parallel 1 \
          --kv-semantic-index ~/models/bge-small-en-v1.5-q8_0.gguf \
          --kv-semantic-threshold 0.65 \
          --kv-semantic-topk 5 \
          --alias default \
          --direct-io \
          --kv-warm-device 1 \
          --weight-paging \
          --fit off \
          --weight-paging-slots 8 \
          --jinja
flowchart TD
  A[启动训练脚本 train.py] --> B[参数解析\n--model_config, --device, --fp16, ...]
  B --> C[加载 GPT2LMHeadModel]
  C --> D[准备数据分片]
  D --> D1[读取 raw_data\nlines → [SEP]]
  D1 --> D2[分片 num_pieces]
  D2 --> D3[过滤 min_length]
  D3 --> D4[添加 [MASK]/[CLS]]
  D4 --> D5[Tokenizer 编码 → tokenized_train_i.txt]
  D5 --> E[数据迭代]
  E --> E1[随机打乱分片顺序]
  E1 --> E2[切滑动窗口 (n_ctx, stride)]
  E2 --> E3[随机打乱窗口顺序]
  E3 --> F[Batch 组织]
  F --> F1[聚合 window → batch_size]
  F1 --> G[前向/反向传播]
  G --> G1[计算交叉熵 Loss]
  G1 --> G2[可选 FP16 + Apex]
  G2 --> G3[梯度裁剪 & 梯度累积]
  G3 --> H[优化器 & 调度器 step\n(Warmup-Linear)]
  H --> I[TensorBoard 记录 Loss\n(--log_step)]
  I --> J{是否 Epoch 结束?}
  J -- 否 --> E
  J -- 是 --> K[保存模型\nmodel/model_epoch{epoch}]
  K --> L{是否完成所有 Epoch?}
  L -- 否 --> E
  L -- 是 --> M[保存最终模型\nmodel/final_model]
  M --> N[训练结束]

  %% 文本生成流程
  N --> O[启动生成脚本 generate.py]
  O --> P[解析参数\n--length,--nsamples,...]
  P --> Q[加载模型 & Tokenizer]
  Q --> R[准备前缀 context_tokens]
  R --> S{is_fast_pattern?}
  S -- 否 --> T[sample_sequence\n(标准方式)]
  S -- 是 --> U[fast_sample_sequence\n(缓存 past_key_values)]
  T --> V[拼接生成 ID 列表]
  U --> V
  V --> W[后处理：ID→token→文本\n移除 [MASK]/[CLS]/[SEP]]
  W --> X{save_samples?}
  X -- 否 --> Y[打印到控制台]
  X -- 是 --> Z[写入 samples.txt]
  Y --> END[生成结束]
  Z --> END


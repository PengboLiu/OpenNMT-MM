CUDA_VISIBLE_DEVICES=6 \
python train.py -data data/demo -save_model save_model/demo-model-text \
-enc_layers 4 -dec_layers 4 \
-rnn_size 128 -word_vec_size 128 -transformer_ff 256 -heads 4  \
-encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 100000  -max_generator_batches 2 -dropout 0.3 \
-batch_size 64 -batch_type sents -normalization tokens  -accum_count 2 \
-optim adam -adam_beta2 0.98 -decay_method noam -warmup_steps 2000 -learning_rate 0.5 \
-max_grad_norm 0 -param_init 0  -param_init_glorot \
-label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 \
-world_size 1 -gpu_ranks 0 -log_file log_mutli-modal_text.txt


stage=0
export CUDA_VISIBLE_DEVICES="0,1,5,7"


if [ $stage -eq 0 ]; then
  # Preparation stage.

  # Install fairseq according to:
  # https://github.com/pytorch/fairseq
  # when testing this code:
  # commit 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 is used.
  #
  # Install quantization toolkit:
  # pip install git+https://github.com/danpovey/quantization.git@master
  # when testing this code:
  # commit c17ffe67aa2e6ca6b6855c50fde812f2eed7870b is used.

  echo "Download hubert model."
  # Parameters about model.
  exp_dir=./pruned_transducer_stateless4/exp/
  model_id=hubert_xtralarge_ll60k_finetune_ls960
  hubert_model_dir=${exp_dir}/hubert_models
  hubert_model=${hubert_model_dir}/${model_id}.pt
  mkdir -p ${hubert_model_dir}
  # For more models refer to: https://github.com/pytorch/fairseq/tree/main/examples/hubert
  wget -c https://dl.fbaipublicfiles.com/hubert/${model_id} -P ${hubert_model_dir}
  wget -c wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -P ${hubert_model_dir}
fi

if [ ! -d ./data/fbank ]; then
  echo "This script assumes ./data/fbank is already generated by prepare.sh"
  exit 0
fi

if [ $stage -eq 1 ]; then
  # This stage is not directly used by codebook indexes extraction.
  # It is a method to "prove" that the downloaed hubert model
  # is inferenced in an correct way if WERs look like normal.
  # Expect WERs:
  # [test-clean-ctc_greedy_search] %WER 2.04% [1075 / 52576, 92 ins, 104 del, 879 sub ]
  # [test-other-ctc_greedy_search] %WER 3.71% [1942 / 52343, 152 ins, 126 del, 1664 sub ]
  ./vq_pruned_transducer_stateless4/hubert_decode.py
fi

if [ $stage -eq 2 ]; then
  # Analysis of disk usage:
  # With num_codebooks==8, each teacher embedding is quantized into
  # a sequence of eight 8-bit integers, i.e. only eight bytes are needed.
  # Training dataset including clean-100h with speed perturb 0.9 and 1.1 has 300 hours.
  # The output frame rates of Hubert is 50 per second.
  # Theoretically, 412M = 300 * 3600 * 50 * 8 / 1024 / 1024 is needed.
  # The actual size of all "*.h5" files storaging codebook index is 450M.
  # I think the extra "48M" usage is some meta information.

  # Time consumption analysis:
  # For quantizer training data(teacher embedding) extraction, only 1000 utts from clean-100 are used.
  # Together with quantizer training, no more than 20 minutes will be used.
  #
  # For codebook indexes extraction,
  # with two pieces of NVIDIA A100 gpus, around three hours needed to process 300 hours training data,
  # i.e. clean-100 with speed purteb 0.9 and 1.1.

  # GPU usage:
  # During quantizer's training data(teacher embedding) and it's training,
  # only the first ONE GPU is used.
  # During codebook indexes extraction, ALL GPUs set by CUDA_VISIBLE_DEVICES are used.
  ./pruned_transducer_stateless4/extract_codebook_index.py \
    --full-libri 0
fi

if [ $stage -eq 3 ]; then
  # Example training script.
  # Note: it's better to set spec-aug-time-warpi-factor=-1
  WORLD_SIZE=$(echo ${CUDA_VISIBLE_DEVICES} | awk '{n=split($1, _, ","); print n}')
  ./pruned_transducer_stateless4/train.py \
    --codebook-loss-scale 0.1 \
    --num-codebooks=${bytes_per_frame} \
    --start-epoch 0 \
    --master-port 12358 \
    --manifest-dir ${cdidx_manifests_dir} \
    --full-libri 0 \
    --spec-aug-time-warp-factor -1 \
    --max-duration 300 \
    --world-size ${WORLD_SIZE} \
    --num-epochs 20 \
fi

if [ $stage -eq 4 ]; then
  # Expected results:
  # errs-test-clean-beam_size_4-epoch-19-avg-10-beam-4.txt:%WER = 5.67
  # errs-test-other-beam_size_4-epoch-19-avg-10-beam-4.txt:%WER = 15.60
  ./pruned_transducer_stateless4/decode.py \
    --decoding-method "modified_beam_search" \
    --epoch $epoch \
    --avg 10 \
    --max-duration 200 \
    --exp-dir ./vq_pruned_transducer_stateless2/exp
fi

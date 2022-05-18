stage=1

text=data/local/lm/librispeech-lm-norm.txt.gz
 text_dir=data/lm/text
 all_train_text=$text_dir/librispeech.txt
 # there are 40,398,052 pieces in all_train_text, which will take 50 MINUTES to be tokenized, with a single process.
 # use $train_pieces data to validate pipeline
 # train_pieces=300000 # 15 times of dev.txt
 # uncomment follwoing line to use all_train_text
 train_pieces=
 dev_text=$text_dir/dev.txt
if [ $stage -le 0 ]; then
   # reference:
   # https://github.com/kaldi-asr/kaldi/blob/pybind11/egs/librispeech/s5/local/rnnlm/tuning/run_tdnn_lstm_1a.sh#L75
   # use the same data seperation method to kaldi whose result can be used as a baseline
   if [ ! -f $text ]; then
     wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm
   fi
   echo -n >$text_dir/dev.txt
   # hold out one in every 2000 lines as dev data.
   gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%2000 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$all_train_text
 fi

if [ $stage -eq 1 ]; then
  # for text_file in dev.txt librispeech.txt; do
  #  python ./vq_pruned_transducer_stateless2/tokenize_text.py \
  #   --tokenizer-path ./data/lang_bpe_500/bpe.model \
  #   --text-file ./data/lm/text/$text_file
  # done
  lmplz -o 4 --text data/lm/text/librispeech.txt --arpa train.arpa -S 10%
  # lmplz -o 4 --text data/lm/text/librispeech.txt --arpa discount_train.arpa -S 10% \
  #   --discount_fallback
  # lmplz -o 4 --text data/lm/text/librispeech.txt.tokens --arpa token_train.arpa -S 10%  \
  #   --discount_fallback 0.5 
fi

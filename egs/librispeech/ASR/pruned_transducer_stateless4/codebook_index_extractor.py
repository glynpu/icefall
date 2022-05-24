import argparse
import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from asr_datamodule import LibriSpeechAsrDataModule
from hubert_xlarge import HubertXlargeFineTuned
from icefall.utils import AttributeDict
from lhotse.features.io import NumpyHdf5Writer

class CodebookIndexExtractor:
    """
    A wrapper of quantiation.Quantizer.

    It's responsible for:
        1. extract and save activations from a teacher model.
        2. train quantizer from previous activations.
        3. extract codebook indexes for whole training set.
           Normally this step needs multi GPUs.
    """

    def __init__(self, params: AttributeDict):
        self.params = params

        self.memory_dir = self.params.exp_dir / \
                f"mem/{self.params.teacher_model_id}/"

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.build_dl()
        self.teacher_model = HubertXlargeFineTuned(self.params)

    @staticmethod
    def worldsize():
        assert torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" in os.environ, \
            "It's better to GPU to extrac codebook indices"
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        return world_size

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="CodebookIndexExtracto related options",
            description="These options are used to build the CodebookIndexExtractor.",
        )

        # Options about teacher embeddings eatraction.
        parser.add_argument(
            "--memory-layer",
            type=int,
            help="layer to extract teacher embeddings, 1-based.",
            default=36,
        )

        parser.add_argument(
            "--num-utts",
            type=int,
            default=1000,
            help="num utts to train quantizer",
        )


    @property
    def memory_file_path(self):
        memory_file_id = f"num_utts_{self.params.num_utts}" + \
                         f"-layer_{self.params.memory_layer}" + \
                         "-memory_embeddings.h5"

        memory_file_path= self.memory_dir / memory_file_id
        return memory_file_path

    def extract_and_save_memory(self):
        if self.memory_file_path.exists():
            warn_message = f"{self.memory_file_path} already exists." + \
                    " Skip extracting activations from teacher model"
            logging.warn(warn_message)
            return

        total_cuts = 0
        with NumpyHdf5Writer(self.memory_file_path) as writer:
            for batch_idx, batch in enumerate(self.quantizer_train_dl):
                cut_list = batch["supervisions"]["cut"]
                encoder_memory, num_frames = self.teacher_model.extract_memory(batch)
                for idx, cut in enumerate(cut_list):
                    cut.encoder_memory = writer.store_array(
                        key=cut.id,
                        value=encoder_memory[idx][:num_frames[idx]],
                    )
                total_cuts += len(cut_list)
                logging.info(f"Processed {total_cuts} cuts.")

        logging.info(f"In total, processed {total_cuts}.")

    def build_dl(self):
        # dl to train quantizer.
        librispeech = LibriSpeechAsrDataModule(self.params)
        quantizer_trian_cuts = librispeech.train_clean_100_cuts().subset(first=self.params.num_utts)
        self.quantizer_train_dl = librispeech.train_dataloaders(quantizer_trian_cuts)

        # dl to extract codebook indexes.
        train_cuts = librispeech.train_clean_100_cuts()
        if self.params.full_libri:
            train_cuts += librispeech.train_clean_360_cuts()
            train_cuts += librispeech.train_other_500_cuts()

        self.train_dl = librispeech.train_dataloaders(train_cuts)
    def train_quantizer():
        import pdb; pdb.set_trace()
        pass

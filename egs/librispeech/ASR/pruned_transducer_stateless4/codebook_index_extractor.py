import argparse
import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from hubert_xlarge import HubertXlargeFineTuned

class CodebookIndexExtractor:
    """
    A wrapper of quantiation.Quantizer.

    It's responsible for:
        1. extract and save activations from a teacher model.
        2. train quantizer from previous activations.
        3. extract codebook indexes for whole training set.
           Normally this step needs multi GPUs.
    """

    def __init__(self, params: AttributeDict, teacher_model):
        self.teacher_model = teacher_model
        self.params = params
        self.memory_dir = self.params.exp_dir / \
                f"mem/{self.teacher_model_id}/"

    @staticmethod
    def get_worldsize()
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
            help="num utts to train quantizer",
        )


    def extract_and_save_memory(self, dl):

        memory_file_id = f"{self.params.num_utts}" + \
                         "-{self.params.teacher_model_id}" +  \
                         "-{self.params.memory_layer}layer" + \
                         "-memory_embeddings"
        memory_writer_path = self.memory_dir / memory_file_id
        with NumpyHdf5Writer(memory_writer_path) as writer:
            for batch_idx, batch in enumerate(dl):
                encoder_memory, num_frames = self.teacher_model.extract_memory(batch)
                for idx, cut in enumerate(cut_list):
                    cut.encoder_memory = writer.store_array(
                        key=cut.id,
                        value=encoder_memory[idx][:num_frames[idx]],
                    )
                    total_frames += num_frames
                total_cuts += len(cut_list)
                logging.info(f"Processed {total_cuts} cuts with {total_frames} frames.")

        logging.info(f"In total, processed {total_cuts} cuts with {total_frames} frames.")


if __name__ == "__main__":
    codebook_indexes_extractor = CodebookIndexExtractor()
    CodebookIndexExtractor.extract_codebook_indexes()

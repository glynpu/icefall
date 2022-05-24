import argparse
from pathlib import Path

import torch
from codebook_index_extractor import CodebookIndexExtractor
from asr_datamodule import LibriSpeechAsrDataModule
from hubert_xlarge import HubertXlargeFineTuned
from icefall.utils import AttributeDict

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="pruned_transducer_stateless4/exp/",
        help="The experiment dir",
    )

    return parser

@torch.no_grad()
def main():
    parser = get_parser()
    assert CodebookIndexExtractor.worldsize() > 0
    LibriSpeechAsrDataModule.add_arguments(parser)
    HubertXlargeFineTuned.add_arguments(parser)
    CodebookIndexExtractor.add_arguments(parser)

    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    # reset some parameters needed by hubert.
    params.update(HubertXlargeFineTuned.get_params())
    params.device = torch.device("cuda", 0)

    extractor = CodebookIndexExtractor(params=params)
    extractor.extract_and_save_memory()


if __name__ == "__main__":
    main()



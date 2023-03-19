from io import TextIOWrapper
from typing import Iterator, Optional, Union

from text_correction_utils.api.cli import TextCorrectionCli
from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils import data

from spelling_correction import version
from spelling_correction.api.corrector import SpellingCorrector
from spelling_correction.api.server import SpellingCorrectionServer


class SpellingCorrectionCli(TextCorrectionCli):
    text_corrector_cls = SpellingCorrector
    text_correction_server_cls = SpellingCorrectionServer

    def version(self) -> str:
        return version.__version__

    def setup_corrector(self) -> TextCorrector:
        cor = super().setup_corrector()
        # perform some additional setup
        assert isinstance(cor, SpellingCorrector)
        cor.set_inference_options(
            strategy=self.args.search_strategy,
            beam_width=self.args.beam_width,
            sample_top_k=self.args.sample_top_k
        )
        return cor

    def correct_iter(
        self,
        corrector: SpellingCorrector,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        yield from corrector.correct_iter(
            ((data.text, data.language) for data in iter),
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            return_raw=True,
            show_progress=self.args.progress
        )

    def correct_file(
        self,
        corrector: SpellingCorrector,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        corrector.correct_file(
            path,
            self.args.input_format,
            out_file,
            self.args.output_format,
            lang,
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            show_progress=self.args.progress
        )


def main():
    parser = SpellingCorrectionCli.parser(
        "Spelling correction",
        "Detect and correct spelling errors in text"
    )
    parser.add_argument(
        "--search-strategy",
        choices=["greedy", "beam", "sample"],
        type=str,
        default="greedy",
        help="Search strategy to use during decoding"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width to use for beam search decoding"
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=5,
        help="Sample from top k tokens during sampling decoding"
    )
    SpellingCorrectionCli(parser.parse_args()).run()

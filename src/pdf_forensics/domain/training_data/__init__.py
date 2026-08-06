"""Domain model for building an ML training table from an external benchmark's output.

Free of any dependency on the benchmark project's own code — these types only
describe the shape of a labeled training sample in the validator's own
feature space, once a benchmark-generated PDF has been re-parsed and
re-featurized by this platform's own Modules 1-2.
"""

from pdf_forensics.domain.training_data.benchmark_sample_ref import BenchmarkSampleRef
from pdf_forensics.domain.training_data.training_sample import (
    SkippedSample,
    TrainingDataset,
    TrainingSample,
)

__all__ = [
    "BenchmarkSampleRef",
    "SkippedSample",
    "TrainingDataset",
    "TrainingSample",
]

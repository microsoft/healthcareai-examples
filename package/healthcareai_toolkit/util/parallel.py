# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from joblib import Parallel, delayed
from tqdm import tqdm
import itertools


def batchify(iterable, batch_size=10):
    """Yield successive chunks of a specified size from an iterable."""
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, batch_size))
        if not chunk:
            break
        yield chunk


class ParallelSubmitter:
    def __init__(self, submit_batch_func, n_jobs=5, batch_size=1, return_as="list"):
        """
        Initialize the ParallelSubmitter.

        Args:
            submit_batch_func (callable): The function to submit a batch of inputs.
                                          It should accept a variable number of keyword arguments, each being a list of inputs.
                                          For example, if there are two lists of inputs, the function signature should be:
                                          def submit_batch_func(list1, list2, ...).
            n_jobs (int): The number of jobs to run in parallel.
            batch_size (int): The size of each batch.
            return_as (str): The format to return results ('list', 'generator', or 'generator_unordered').
        """
        self.submit_batch_func = submit_batch_func
        self.batch_size = batch_size
        self.return_as = return_as
        self._generate_as = (
            "generator_unordered" if return_as == "generator_unordered" else "generator"
        )
        self.tqdm_params = {}
        self.parallel_params = {}
        self.set_parallel_params(n_jobs=n_jobs, prefer="threads")

    def set_tqdm_params(self, **kwargs):
        """Set custom parameters for the tqdm progress bar."""
        if "total" in kwargs:
            raise ValueError(
                "Do not set 'total' in tqdm parameters. Pass it to the 'submit' method instead."
            )
        self.tqdm_params.update(kwargs)

    def set_parallel_params(self, **kwargs):
        """Set custom parameters for the Parallel class."""
        if "return_as" in kwargs:
            raise ValueError("Do not set 'return_as' in Parallel parameters.")
        self.parallel_params.update(kwargs)

    def _process_batch(self, index, batch):
        return (index, self.submit_batch_func(**batch))

    def _submit(self, total=None, **kwargs):
        kwarg_keys = kwargs.keys()
        batch_generators = [
            batchify(value, batch_size=self.batch_size) for value in kwargs.values()
        ]
        indexed_batches = enumerate(
            itertools.zip_longest(*batch_generators, fillvalue=[])
        )

        with tqdm(total=total, **self.tqdm_params) as pbar:
            results_gen = Parallel(return_as=self._generate_as, **self.parallel_params)(
                delayed(self._process_batch)(index, dict(zip(kwarg_keys, batch)))
                for index, batch in indexed_batches
            )

            for batch_results in results_gen:
                index, batch_result = batch_results
                for i, result in enumerate(batch_result):
                    yield (index * self.batch_size + i, result)
                pbar.update(len(batch_result))

    def submit(self, total=None, **kwargs):
        """
        Submit inputs and collect results.

        Args:
            total (int, optional): The total number of inputs.
            **kwargs: Keyword arguments where each key is a name and each value is a list of inputs.

        Returns:
            list or generator: The results of the inputs.

        The method processes the inputs in parallel batches. It uses the `submit_batch_func` to process each batch of inputs.
        The results can be returned as a list or a generator, depending on the `return_as` parameter.

        Example:
            def process_batch(list1, list2):
                return [x + y for x, y in zip(list1, list2)]

            submitter = ParallelSubmitter(process_batch, n_jobs=4, batch_size=2, return_as='list')
            submitter.set_tqdm_params(desc='Processing')
            indexes, results = submitter.submit(total=4, list1=[1, 2, 3, 4], list2=[5, 6, 7, 8])
            print(indexes)  # Output: [0, 1, 2, 3]
            print(results)  # Output: [6, 8, 10, 12]
        """
        results_gen = self._submit(total=total, **kwargs)

        if self.return_as == "list":
            results = list(results_gen)
            indexes, results = zip(*results)
            return list(indexes), list(results)
        elif self.return_as in {"generator", "generator_unordered"}:
            return ((index, result) for index, result in results_gen)

    def __str__(self):
        return (
            f"ParallelSubmitter(submit_batch_func={self.submit_batch_func.__name__}, "
            f"n_jobs={self.parallel_params.get('n_jobs', 'not set')}, "
            f"batch_size={self.batch_size}, return_as={self.return_as}, "
            f"tqdm_params={self.tqdm_params}, parallel_params={self.parallel_params})"
        )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import magic
import os

cur_dir = os.path.dirname(__file__)
custom_magic_file = os.path.join(cur_dir, "extra_types.magic")
default_magic = magic.Magic(uncompress=True, mime=True)
custom_magic = magic.Magic(uncompress=True, mime=True, magic_file=custom_magic_file)


def get_filetype(input_data):
    def detect_filetype(input_bytes, m):
        return m.from_buffer(input_bytes)

    if isinstance(input_data, str):
        with open(input_data, "rb") as f:
            input_data = f.read()
    elif not isinstance(input_data, bytes):
        raise ValueError("Input data must be either bytes or a file path (string).")

    result = detect_filetype(input_data, default_magic)
    if result == "application/x-dbt":
        result = detect_filetype(input_data, custom_magic)
    return result

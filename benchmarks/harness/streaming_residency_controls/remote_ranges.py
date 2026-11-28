"""Read validated spans of this lease's immutable canonical archive."""
import base64
import json
import os
import re
import sys
from pathlib import Path


def copy_spans(root, name, encoded, skip, output):
    root = os.path.realpath(root)
    assert re.fullmatch(r'[a-z0-9_-]+', name) and skip >= 0
    path = os.path.realpath(Path(root) / (name + '.tar'))
    assert os.path.commonpath([root, path]) == root
    requests = json.loads(base64.b64decode(encoded))
    size = os.path.getsize(path)
    with open(path, 'rb') as source:
        for request in requests:
            offset, length = request['offset'], request['bytes']
            assert 0 <= offset <= size and 0 < length <= 4 * 2**20
            assert offset + length <= size
            if skip >= length:
                skip -= length
                continue
            source.seek(offset + skip)
            remaining = length - skip
            skip = 0
            while remaining:
                data = source.read(min(1024 * 1024, remaining))
                if not data:
                    raise EOFError('immutable archive truncated')
                output.write(data)
                remaining -= len(data)
        output.flush()
    assert skip == 0


if __name__ == '__main__':
    copy_spans('/root/liger', sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.stdout.buffer)

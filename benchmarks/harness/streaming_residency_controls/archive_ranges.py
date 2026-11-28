"""Reassemble the SAME canonical tar from hash-verified cached encoded objects.

No root codec, file selection, tensor arithmetic or dtypes are modified. Only
missing immutable archive object spans cross SSH; the reconstructed canonical
archive must match the remote whole-file SHA before the root decoder is used.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path

from benchmarks.harness import lossless_results as codec


def catalogue(paths):
    result = {}
    for path in paths:
        path = Path(path)
        size = path.stat().st_size
        try:
            with tarfile.open(path, 'r:') as archive:
                first = archive.next()
                if first.name != 'original-files.json' or first.offset_data + first.size > size:
                    continue
                raw = archive.extractfile(first).read()
                manifest = json.loads(raw)
                index = path.parent / 'index.json'
                if index.exists():
                    if codec.sha(raw) != json.loads(index.read_text())['manifest_sha256']:
                        raise ValueError('prior manifest SHA mismatch')
                if manifest['format'] != 'soup-lossless-results-v1':
                    continue
                while True:
                    try:
                        member = archive.next()
                    except (tarfile.ReadError, EOFError):
                        break
                    if member is None or member.offset_data + member.size > size:
                        break
                    if not member.name.startswith('objects/'):
                        continue
                    identity = member.name.split('/')[1]
                    row = manifest['objects'][identity]
                    if member.size != row['encoded_bytes']:
                        raise ValueError('prior encoded object size mismatch')
                    key = (row['encoded_sha256'], member.size)
                    result.setdefault(key, []).append((path, member.offset_data))
        except (tarfile.ReadError, EOFError):
            continue
    return result


def cached_bytes(row, existing, cache):
    size, expected = row['encoded_bytes'], row['encoded_sha256']
    local = cache / expected
    if local.is_file():
        data = local.read_bytes()
        if len(data) == size and codec.sha(data) == expected:
            return data
        raise ValueError('received encoded object SHA mismatch')
    for path, offset in existing.get((expected, size), []):
        with path.open('rb') as source:
            source.seek(offset)
            data = source.read(size)
        if len(data) == size and codec.sha(data) == expected:
            return data
    return None


def layout_from_archive(path):
    with tarfile.open(path, 'r:') as archive:
        manifest = archive.extractfile('original-files.json').read()
        objects = {member.name.split('/')[1]: {'offset': member.offset_data,
                   'bytes': member.size} for member in archive.getmembers()
                   if member.name.startswith('objects/')}
    return {'manifest_base64': base64.b64encode(manifest).decode(), 'objects': objects}


def rebuild(layout, index, folder, prior_archives, stream, remaining, rate,
            refresh=lambda: None, record=lambda value: None,
            clock=time.monotonic, sleep=time.sleep):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / 'received-objects'
    cache.mkdir(exist_ok=True)
    raw = base64.b64decode(layout['manifest_base64'], validate=True)
    if codec.sha(raw) != index['manifest_sha256']:
        raise ValueError('canonical manifest SHA mismatch')
    manifest = json.loads(raw)
    if manifest['format'] != 'soup-lossless-results-v1' or manifest['chunk_bytes'] != 4 * 2**20:
        raise ValueError('unsupported canonical format')
    if set(layout['objects']) != set(manifest['objects']):
        raise ValueError('layout object set mismatch')
    existing = catalogue(prior_archives)
    missing, reused = [], 0
    for identity, row in sorted(manifest['objects'].items()):
        span = layout['objects'][identity]
        if (not re.fullmatch(r'[0-9a-f]{64}', identity)
                or span['bytes'] != row['encoded_bytes']
                or not 0 < span['bytes'] <= 4 * 2**20
                or not 0 <= span['offset'] <= index['archive_bytes'] - span['bytes']):
            raise ValueError('invalid immutable object span')
        if cached_bytes(row, existing, cache) is None:
            missing.append({'identity': identity, 'offset': span['offset'],
                            'bytes': span['bytes'], 'sha256': row['encoded_sha256']})
        else:
            reused += row['encoded_bytes']
    need = sum(row['bytes'] for row in missing)
    if need / max(rate, 1) * 1.5 + 180 > remaining():
        raise RuntimeError('actual missing archive objects cannot fit collection reserve')
    record({'phase': 'range-plan', 'encoded_bytes_reused': reused,
            'encoded_bytes_missing': need, 'missing_objects': len(missing)})
    network = 0
    batches, batch, total = [], [], 0
    for item in missing:
        if batch and total + item['bytes'] > 16 * 2**20:
            batches.append(batch)
            batch, total = [], 0
        batch.append(item)
        total += item['bytes']
    if batch:
        batches.append(batch)
    for batch_index, batch in enumerate(batches):
        request = json.dumps(batch, sort_keys=True, separators=(',', ':')).encode()
        key = codec.sha(request)
        part = folder / (key + '.range-part')
        (folder / (key + '.request.json')).write_bytes(request)
        expected = sum(row['bytes'] for row in batch)
        last_progress = clock()
        while (part.stat().st_size if part.exists() else 0) < expected:
            offset = part.stat().st_size if part.exists() else 0
            if remaining() < 180:
                raise RuntimeError('collection financial/time deadline')
            failure = None
            try:
                with part.open('ab') as sink:
                    stream(batch, offset, sink, remaining() - 120)
            except (OSError, EOFError) as error:
                failure = error
            after = part.stat().st_size
            network += after - offset
            if after > expected:
                raise ValueError('range response exceeds requested bytes')
            if after > offset:
                last_progress = clock()
            if after < expected and failure is None:
                failure = EOFError('SSH response ended before the requested range completed')
            record({'phase': 'range-receive', 'batch': batch_index, 'batches': len(batches),
                    'received_this_batch': after, 'expected_this_batch': expected,
                    'network_bytes_this_call': network,
                    'no_progress_seconds': clock() - last_progress,
                    'error_type': type(failure).__name__ if failure else None})
            # Promote every COMPLETE encoded object even after a connection failure.
            position = 0
            with part.open('rb') as source:
                for row in batch:
                    if position + row['bytes'] > after:
                        break
                    data = source.read(row['bytes'])
                    if codec.sha(data) != row['sha256']:
                        raise ValueError('received range object SHA mismatch')
                    target = cache / row['sha256']
                    temporary = target.with_suffix('.incoming')
                    temporary.write_bytes(data)
                    os.replace(temporary, target)
                    position += row['bytes']
            if after == expected:
                break
            if clock() - last_progress > 600:
                raise RuntimeError('no new archive bytes for ten minutes') from failure
            sleep(15)
            try:
                refresh()
            except (OSError, EOFError) as error:
                record({'phase': 'port-refresh-transient', 'error_type': type(error).__name__})
        part.unlink()
    target = folder / 'results.tar'
    incoming = folder / 'results.tar.incoming'
    with tarfile.open(incoming, 'w') as archive:
        member = tarfile.TarInfo('original-files.json')
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
        for identity, row in sorted(manifest['objects'].items()):
            data = cached_bytes(row, existing, cache)
            if data is None:
                raise ValueError('canonical object still missing')
            member = tarfile.TarInfo('objects/' + identity)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    if (incoming.stat().st_size != index['archive_bytes']
            or codec.file_sha(incoming) != index['archive_sha256']):
        raise ValueError('reconstructed canonical archive SHA mismatch')
    os.replace(incoming, target)
    return {'archive_sha256': index['archive_sha256'], 'canonical_archive_verified': True,
            'encoded_object_bytes_reused': reused, 'new_network_object_bytes': network,
            'new_object_count': len(missing), 'range_batches': len(batches),
            'canonical_archive_bytes': index['archive_bytes'],
            'transport': 'authenticated SSH object spans; full identical tar reassembled'}

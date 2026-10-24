#!/usr/bin/env python3
"""Interactive official wScene download; credentials stay in process memory."""
import getpass
import os
from pathlib import Path
import sys
import requests

NAME = 'interactvlm-3d-hcontact-wScene-damon-lemon-rich.zip'
URL = 'https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=' + NAME


def download(destination, username, password):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / NAME
    partial = destination / (NAME + '.part')
    if final.exists():
        raise FileExistsError(f'Already present: {final}')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        with partial.open('rb') as handle:
            if handle.read(4) != b'PK\x03\x04':
                raise ValueError('Existing partial file is not a ZIP; inspect it before retrying')
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    # Same authenticated POST endpoint used by upstream fetch_data.sh.
    # TLS verification remains enabled. Never print submitted credentials.
    with requests.post(URL, data={'username': username, 'password': password},
                       headers=headers, stream=True, timeout=(20,120)) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            raise RuntimeError('Server returned a web page instead of the ZIP. Check your InteractVLM account/login.')
        resume = offset > 0 and response.status_code == 206
        if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Unexpected resume range; existing partial file was preserved')
        if not resume:
            offset = 0
        blocks = response.iter_content(chunk_size=1024*1024)
        first = next(blocks, b'')
        if not first or (not resume and not first.startswith(b'PK\x03\x04')):
            raise RuntimeError('Response is not a ZIP; existing file was preserved')
        expected = response.headers.get('Content-Length')
        total = offset + int(expected) if expected else None
        written = offset
        with partial.open('ab' if resume else 'wb') as handle:
            handle.write(first)
            written += len(first)
            for block in blocks:
                if block:
                    handle.write(block)
                    written += len(block)
                    end = f' / {total/1e9:.2f} GB' if total else ' GB'
                    print(f'\r{written/1e9:.2f}{end}', end='', flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        if total is not None and written != total:
            raise RuntimeError('Incomplete download; rerun to request resume')
    import zipfile
    with zipfile.ZipFile(partial) as archive:
        if not archive.namelist():
            raise RuntimeError('Empty ZIP archive')
    partial.rename(final)
    print(f'\nSaved: {final}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print('Official InteractVLM wScene model download. Use your InteractVLM account.')
    username = input('Email / username: ').strip()
    password = getpass.getpass('Password (hidden): ')
    try:
        download(root/'data/downloads/manual', username, password)
    except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
        print(f'\nDownload stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)

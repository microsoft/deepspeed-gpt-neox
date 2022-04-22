import os
import time
import requests


base_url = "https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/"


def download(filename, dst, max_retry=3, verbose=True):
    start = time.time()
    url = base_url + filename
    success = False
    attempt = 0
    while not success or attempt < max_retry:
        response = requests.get(url)
        success = response.ok
        attempt += 1
    if not success:
        raise RuntimeError(f"Unable to download file from {url} after {max_retry} attempts")

    with open(os.path.join(dst, filename), 'wb') as fd:
        fd.write(response.content)
    duration = time.time() - start
    if verbose:
        print(f"Downloaded {filename} in {duration:.2f} sec")


def download_neox20b_checkpoint(dst_path, verbose=True):
    if verbose:
        print(f"Downloading checkpoint to {dst_path}")
    os.makedirs(dst_path, exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'configs'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'global_step150000'), exist_ok=True)

    files = [
        'latest',
        '20B_tokenizer.json',
        'configs/20B.yml',
    ]
    files += [f'global_step150000/mp_rank_0{i}_model_states.pt' for i in range(8)]
    files = []
    for layer in range(49):
      if layer == 1: continue
      for tp in range(2):
        files.append(f"global_step150000/layer_{layer:02}-model_{tp:02}-model_states.pt")

    for filename in files:
        download(filename, dst_path, verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dst', type=str, help="destination folder to download checkpoint to")
    args = parser.parse_args()

    download_neox20b_checkpoint(args.dst)

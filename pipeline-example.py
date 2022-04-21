import json
import torch
import time
import argparse
import deepspeed

from megatron import mpu, NeoXPipeline
from megatron.utils import print_rank_0, setup_for_inference_or_eval
from megatron.text_generation_utils import generate_samples_from_prompt


def print_latency(latency_set, title=""):
    # drop first few samples for warmup
    warmup = 5
    latency_set = latency_set[warmup:]
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print("====== latency stats {0} ======", title)
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))
    else:
        print(f"Use more than {warmup} trials to see latency stats")


def main():
    config = {
        "load": "/data/users/reyazda/gpt-neox20B",
        "vocab_file": "/data/users/reyazda/gpt-neox20B/20B_tokenizer.json",
        "model_parallel_size": 4
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--deepspeed', action="store_true", help='enable ds inference')
    parser.add_argument('-t', '--trials', default=10, type=int, help='number of queries to try')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()
    print(f'deepspeed enabled={args.deepspeed}, running {args.trials} queries')

    pipeline = NeoXPipeline(config)

    if args.deepspeed:
        deepspeed.init_inference(
                model=pipeline.model,
                mp_size=config["model_parallel_size"],
                mpu=mpu,
                dtype=torch.half,
                replace_with_kernel_inject=True
        )

    maximum_tokens = 50
    trials = args.trials
    query = "deepspeed is"

    responses = []
    torch.cuda.synchronize()
    start = time.time()
    for i in range(trials):
        response = pipeline(query, maximum_tokens=maximum_tokens)
        if torch.distributed.get_rank() == 0:
            responses.append(response[0]['text'])
    torch.cuda.synchronize()
    end = time.time()

    if torch.distributed.get_rank() == 0:
        per_query = (end - start) / float(trials)
        print_rank_0(f"query={query}, response={response}")
        print_rank_0(f"per_query={per_query} sec")
        per_token = (per_query / maximum_tokens) * 1000.0
        print_rank_0(f"per_token={per_token} ms")

        print_rank_0(set(responses))
        print_rank_0(f"num of unique responses: {len(set(responses))}")

        print_latency(pipeline.latencies)


if __name__ == "__main__":
    main()

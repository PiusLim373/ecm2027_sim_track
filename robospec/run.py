"""gRPC policy client loop for the ecm27.policy.v1 evaluator.

usage: python -m robospec.run --policy scripted_smoke [--max_steps N] [--endpoint host:port]
The endpoint defaults to $ECM27_EVAL_ENDPOINT, then 127.0.0.1:50000.
"""
import argparse
import os
import time

import grpc
import numpy as np

from robofinals.distributed import policy_v1_pb2 as pb
from robofinals.distributed.grpc_policy import PolicyClient
from robospec import policies
from robospec.obs import decode

RPC_TIMEOUT_S = 300 * 60  # same as the upstream example client
MAX_MSG_BYTES = 64 * 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="scripted_smoke", help="registry name or module:Class")
    p.add_argument("--endpoint", default=os.environ.get("ECM27_EVAL_ENDPOINT", "127.0.0.1:50000"))
    p.add_argument("--max_steps", type=int, default=None, help="default: policy.max_steps, else run forever")
    p.add_argument("--connect_timeout", type=float, default=30 * 60, help="seconds to wait for the server")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    policy = policies.load(args.policy)()
    max_steps = args.max_steps if args.max_steps is not None else policy.max_steps

    channel = grpc.insecure_channel(args.endpoint, options=[("grpc.max_receive_message_length", MAX_MSG_BYTES)])
    print(f"[robospec] waiting for evaluator at {args.endpoint} ...", flush=True)
    grpc.channel_ready_future(channel).result(timeout=args.connect_timeout)
    client = PolicyClient(channel)

    transition = client.get_observation(pb.Empty(), timeout=RPC_TIMEOUT_S)
    n_envs = len(transition.observations)
    for i in range(n_envs):
        policy.reset(i)
    print(f"[robospec] connected: {n_envs} env(s), policy={args.policy}, max_steps={max_steps}", flush=True)

    step, t0 = 0, time.time()
    while max_steps is None or step < max_steps:
        for i, (terminated, truncated) in enumerate(zip(transition.terminated, transition.truncated)):
            if terminated or truncated:
                print(f"[robospec] env {i} episode ended (terminated={terminated}, truncated={truncated})",
                      flush=True)
                policy.reset(i)
        actions = np.asarray(policy.act([decode(o) for o in transition.observations]), np.float32)
        if actions.shape not in ((n_envs, 19), (n_envs, 20)):
            raise ValueError(f"policy returned {actions.shape}, expected ({n_envs}, 19)")
        request = pb.ActionBatch(actions=[pb.Action(values=row) for row in actions])
        try:
            transition = client.step(request, timeout=RPC_TIMEOUT_S)
        except grpc.RpcError as error:
            # Wall-clock timeout reset: read the reset result, then continue (flags trigger reset()).
            if error.code() != grpc.StatusCode.FAILED_PRECONDITION:
                raise
            transition = client.get_observation(pb.Empty(), timeout=RPC_TIMEOUT_S)
            continue
        step += 1
        if step % args.log_every == 0:
            print(f"[robospec] step {step}  {step / (time.time() - t0):.1f} steps/s", flush=True)
    print(f"[robospec] done after {step} steps", flush=True)


if __name__ == "__main__":
    main()

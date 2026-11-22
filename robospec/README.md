# robospec

Our policy code for the Chamber Service task. It talks to the evaluator only through the gRPC
interface (`robofinals/distributed/policy_v1.proto`, spec in the top-level README) and imports
nothing from Isaac, so it can ship in a slim policy container.

```
robospec/
├── run.py                 # gRPC client loop: connect, decode, per-env reset, step
├── obs.py                 # observation decoding (schema from robofinals/distributed/chamber_schema.py)
├── policies/
│   ├── base.py            # Policy interface: reset(env_index), act(obs_list) -> (N, 19)
│   ├── __init__.py        # registry: name -> "module:Class"
│   └── scripted_smoke.py  # open-loop base/arm/gripper motion test
├── configs/               # simulator (evaluator) configs for local runs
│   ├── chamber_eval_default.json  # like the upstream example: robot spawns from the box anchor
│   └── chamber_demo_pose.json     # robot starts at the demo pose (0.6, -0.85), facing +y
└── tools/make_montage.py  # recorded episode -> labelled head+wrist video
```

## Run locally (two terminals, repo root, `conda activate ecm2027-sim-track`)

Terminal 1, simulator. Wait for `[grpc_policy] listening on 127.0.0.1:50000`:

```bash
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
  env -u PYTHONPATH -u LD_LIBRARY_PATH \
  bash scripts/run_eval_env.sh --eval_env_config robospec/configs/chamber_demo_pose.json \
  --eval_record_dir ./eval_result/episodes
```

Terminal 2, policy:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=$PWD python -m robospec.run --policy scripted_smoke
```

Make a video of what was recorded:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=$PWD python -m robospec.tools.make_montage \
  eval_result/smoke.mp4 eval_result/episodes/<run_id> --policy scripted_smoke
```

The memory cap and `env -u ...` are for this laptop: the cap kills the simulator instead of
freezing the machine, and the `env -u` drops ROS paths that `~/.bashrc` adds.

## Adding a policy

Subclass `policies.base.Policy`, implement `reset` and `act`, and add it to `REGISTRY` in
`policies/__init__.py` (or pass `--policy module:Class`).

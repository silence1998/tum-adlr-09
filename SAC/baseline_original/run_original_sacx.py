"""
Headless runner for the ORIGINAL training_SAC-X.py algorithm (kept unchanged in training_SAC-X_original.py,
model_original.py, environment_original.py) so it can be used as a baseline.

The optimisation step, networks, hyper-parameters (batch 512, gamma 0.999, lr 3e-4, tau 0.005 once per episode,
replay 10000, entropy coefficient 1, std clamp [1e-6, 0.5]) and env are exactly the original ones. Only changes:
  * no wandb / matplotlib (logging to CSV instead), env-step budget instead of episode count,
  * grid size configurable (original hard-coded 20, parameters.py uses 10; default here = 10 to match the other runs),
  * episodes are capped at --max-episode-steps (otherwise a policy that stands still never ends an episode);
    a capped transition is stored as a normal non-terminal transition, exactly as the original code would store it,
  * periodic deterministic evaluation (tanh(mu)) on the GPU twin env (torch_env.py) with the same protocol as the
    fixed SAC-X / PPO runs.

Run from this folder:  python run_original_sacx.py --total-steps 100000 --seed 0 --log ../../results/runs/orig_s0.csv
"""
import argparse
import csv
import os
import sys
import time
import types
import random
from collections import namedtuple

import numpy as np

# --- minimal stand-ins for gym / pygame (not installed; only used for spaces + rendering by the original env)
if "gym" not in sys.modules:
    try:
        import gym  # noqa: F401
    except ImportError:
        gym = types.ModuleType("gym")

        class _Env:
            def reset(self, seed=None, options=None):
                if seed is not None or not hasattr(self, "np_random"):
                    self.np_random = np.random.default_rng(seed)

        gym.Env = _Env
        spaces = types.ModuleType("gym.spaces")
        spaces.Box = lambda *a, **k: None
        spaces.Dict = lambda *a, **k: None
        spaces.Discrete = lambda *a, **k: None
        gym.spaces = spaces
        sys.modules["gym"] = gym
        sys.modules["gym.spaces"] = spaces
if "pygame" not in sys.modules:
    try:
        import pygame  # noqa: F401
    except ImportError:
        sys.modules["pygame"] = types.ModuleType("pygame")

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # for torch_env

import torch
import torch.nn.functional as F
from model_original import ActorNetwork, CriticNetwork, ValueNetwork, ReplayMemory
from environment_original import GridWorldEnv
from torch_env import evaluate_policy

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--num-obstacles", type=int, default=5)
    ap.add_argument("--max-episode-steps", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-episodes", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="../../results/runs/orig.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = {'input_dims': 4 + args.num_obstacles * 2, 'batch_size': 512, 'gamma': 0.999,
          'alpha': 0.0003, 'beta': 0.0003, 'tau': 0.005}
    env = GridWorldEnv(render_mode=None, size=args.size, num_obstacles=args.num_obstacles)
    n_actions = 2
    actorNet = ActorNetwork(hp["alpha"], hp["input_dims"], n_actions=n_actions, name='actor', max_action=[1, 1])
    criticNet_1 = CriticNetwork(hp["beta"], hp["input_dims"], n_actions=n_actions, name='critic_1')
    criticNet_2 = CriticNetwork(hp["beta"], hp["input_dims"], n_actions=n_actions, name='critic_2')
    valueNet = ValueNetwork(hp["beta"], hp["input_dims"], name='value')
    target_valueNet = ValueNetwork(hp["beta"], hp["input_dims"], name='target_value')
    memory = ReplayMemory(10000)

    # ---------------- verbatim copy of the original optimize_model (wandb/print removed)
    def optimize_model():
        if len(memory) < hp["batch_size"]:
            return
        transitions = memory.sample(hp["batch_size"])
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        value = valueNet(state_batch).view(-1)
        value_ = torch.zeros(hp["batch_size"], device=device)
        value_[non_final_mask] = target_valueNet(non_final_next_states).view(-1)

        actions, log_probs = actorNet.sample_normal(state_batch, reparametrize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = criticNet_1.forward(state_batch, actions)
        q2_new_policy = criticNet_2.forward(state_batch, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)

        valueNet.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        valueNet.optimizer.step()

        actions, log_probs = actorNet.sample_normal(state_batch, reparametrize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = criticNet_1.forward(state_batch, actions)
        q2_new_policy = criticNet_2.forward(state_batch, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy).view(-1)
        actor_loss = torch.mean(log_probs - critic_value)
        actorNet.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actorNet.optimizer.step()

        criticNet_1.optimizer.zero_grad()
        criticNet_2.optimizer.zero_grad()
        q_hat = reward_batch + hp["gamma"] * value_
        q1_old_policy = criticNet_1.forward(state_batch, action_batch).view(-1)
        q2_old_policy = criticNet_2.forward(state_batch, action_batch).view(-1)
        critic_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat) + 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss.backward()
        criticNet_1.optimizer.step()
        criticNet_2.optimizer.step()

    def target_update():  # original: once per episode
        target_value_state_dict = dict(target_valueNet.named_parameters())
        value_state_dict = dict(valueNet.named_parameters())
        for name in value_state_dict:
            value_state_dict[name] = hp["tau"] * value_state_dict[name].clone() + \
                                     (1 - hp["tau"]) * target_value_state_dict[name].clone()
        target_valueNet.load_state_dict(value_state_dict)

    def obs_to_state(obs):
        vals = [obs["agent"], obs["target"]]
        for i in range(args.num_obstacles):
            vals.append(obs["obstacle_{0}".format(i)])
        return torch.tensor(np.array(vals), dtype=torch.float, device=device).view(1, -1)

    def policy_det(o):
        mu, _ = actorNet.forward(o)
        return torch.tanh(mu)

    def policy_stoch(o):
        return actorNet.sample_normal(o, reparametrize=False)[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    f = open(args.log, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["env_steps", "eval_return", "eval_success", "eval_collision", "eval_length",
                     "train_success", "train_return", "train_episodes", "wall_time",
                     "eval_return_stoch", "eval_success_stoch"])
    t0 = time.time()
    env_steps, next_eval, seed = 0, 0, 3407 + 1000 * args.seed
    succ_hist, ret_hist = [], []

    def maybe_eval():
        nonlocal next_eval
        if env_steps >= next_eval:
            ev = evaluate_policy(policy_det, args.size, args.num_obstacles, args.max_episode_steps,
                                 args.eval_episodes, device=device, seed=10_000 + env_steps // args.eval_every)
            evs = evaluate_policy(policy_stoch, args.size, args.num_obstacles, args.max_episode_steps,
                                  args.eval_episodes, device=device, seed=10_000 + env_steps // args.eval_every)
            ts = float(np.mean(succ_hist[-100:])) if succ_hist else float("nan")
            tr = float(np.mean(ret_hist[-100:])) if ret_hist else float("nan")
            writer.writerow([env_steps, ev["eval_return"], ev["eval_success"], ev["eval_collision"],
                             ev["eval_length"], ts, tr, len(succ_hist), time.time() - t0,
                             evs["eval_return"], evs["eval_success"]]); f.flush()
            print(f"[orig s{args.seed}] steps {env_steps:>7d} | eval ret {ev['eval_return']:7.2f} succ "
                  f"{ev['eval_success']:.3f} coll {ev['eval_collision']:.3f} len {ev['eval_length']:.1f} | "
                  f"train succ {ts:.3f} ret {tr:.2f} eps {len(succ_hist)} | {time.time() - t0:.0f}s", flush=True)
            next_eval += args.eval_every

    maybe_eval()
    while env_steps < args.total_steps:
        seed += 1
        env.reset(seed=seed)
        state = obs_to_state(env._get_obs())
        ep_ret, reached = 0.0, False
        for t in range(args.max_episode_steps):
            action, _ = actorNet.sample_normal(state, reparametrize=False)
            action = action.cpu().detach().numpy()[0]
            _, reward, done, _, _ = env.step(action)
            ep_ret += reward
            reached = done and reward == env.reward_parameters['target_value']
            reward_t = torch.tensor([reward], dtype=torch.float, device=device)
            next_state = None if done else obs_to_state(env._get_obs())
            memory.push(state, torch.tensor(np.array([action]), dtype=torch.float, device=device), next_state,
                        reward_t)
            state = next_state
            optimize_model()
            env_steps += 1
            maybe_eval()
            if done or env_steps >= args.total_steps:
                break
        succ_hist.append(float(reached)); ret_hist.append(ep_ret)
        target_update()
    maybe_eval()
    f.close()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

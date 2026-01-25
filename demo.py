"""Simple demo script for the OrderedNetHackDataloader."""

import time
from collections import Counter

from nle_prediction import OrderedNetHackDataloader

# Create dataloader
dl = OrderedNetHackDataloader(
    data_dir = "./data/nld-nao",
    batch_size = 10,
    format = "raw",
    prefetch = 0,
)

# Collect stats for first few games
game_steps = Counter()
total_steps = 0
num_games_to_load = 10

start_time = time.time()
for batch in dl:
    for gid in batch["gameids"].flatten():
        gid = int(gid)
        game_steps[gid] += 1
        total_steps += 1

    if len(game_steps) >= num_games_to_load:
        break
elapsed = time.time() - start_time

# Print stats
print(f"\n{'='*50}")
print(f"Loaded {len(game_steps)} games, {total_steps} total steps")
print(f"Time: {elapsed:.2f}s | {elapsed/total_steps*1000:.2f}ms per step")
print(f"{'='*50}\n")

for i, (gid, steps) in enumerate(game_steps.items()):
    print(f"Game {i+1}: gameid={gid}, steps={steps}")

print(f"\n{'='*50}")
print("Sample batch structure (batch size = 3):")
print(f"{'='*50}")
dl2 = OrderedNetHackDataloader(data_dir="./data/nld-nao", batch_size=3, format="raw")
batch = next(iter(dl2))
for key, val in batch.items():
    shape = val.shape if hasattr(val, "shape") else len(val)
    print(f"  {key}: {shape}")

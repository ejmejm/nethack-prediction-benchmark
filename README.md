### Overview

This repository provides a script for creating a dataset based on the NetHack Learning NAO Dataset for testing supervised streaming learning algorithms.
It also provides a dataloader that serves the data created by the script above. It also gives the option of downloading the dataset created in the step above.
At each step, the dataloader returns an observation and the change in score since the last step in the game.

### Dataset Creation

The NetHack Learning NAO dataset contains 1.5 million games played by humans on nethack.alt.org.
Though datasets like this are typically split up and randomly shuffled into an i.i.d. dataset, this repository instead creates an ordered dataset that maintains a specific ordering.
Games are first grouped by the player and sorted chronologically first by the start time of the game, and then by each step in the game so that temporal coherence is preserved.
Players with less than 3 games are excluded from the dataset, and players are then ordered by their median score.
The overall hierarchy of sorting from top to bottom is player name -> player median score -> game start time -> game step.

The goal of the dataset is to have a challenging non-stationary problems that mirrors many of the attributes of real-world problems.
NetHack ordered as described mimics how many real-world problems have different types of non-stationarities that change at different frequencies.
At the most granular level, the state of the game changes from step to step.
As the player progresses deeper into the game, new types of items, enemies, and scenarios are introduced.
Within the games of a single player, there may be consistent strategies used, but even those may change as a player progresses in skill.
As the games progress to more skilled players over time, the distribution of time spent at each floor level will change.
The many levels of non-stationarities in this dataset make it an excellent testbed for streaming learning algorithms.

Other options I may consider in the future, but don't need to be implemented right now:
- Breaking players up if they have more than a threshold number of games to target a maximum number of games played between player switches
- Sorting by major game version before sorting by player


### Dataloader

The dataloader simply serves the dataset in the order described above.
It provides options for a batch size and an amount of prefetching (including any required preprocessing) to speed up the process.
There is also an option to specify the number of games to serve from the dataset, and an option to determine whether the subset should consist of the first part of the dataset, 
or a subset that skips certain players or games to get better coverage over the entire dataset.
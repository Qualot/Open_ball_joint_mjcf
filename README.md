# mjcf file of bio-inspired open ball joint
[WIP] featuring muscle actuation and a floating joint without any constraint

![thumbnail](https://raw.githubusercontent.com/Qualot/Open_ball_joint_mjcf/main/media/thumbnail.png)

## usage
```
python3 -m venv .venv
. .venv/bin/activate
```
Installing obj2mjcf https://github.com/kevinzakka/obj2mjcf

making socket mjcf from the external .obj file
```
obj2mjcf --obj-dir ./external/obj --save-mjcf --decompose --coacd-args.threshold 0.02 --coacd-args.mcts-iterations 100 
```

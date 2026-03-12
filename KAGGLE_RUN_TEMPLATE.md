# Kaggle Run Template

## Cell 1
```python
!bash kaggle_setup.sh
```

## Cell 2
```python
!python setup_network.py
!python train.py --scenario standard --seed 0 --episodes 100 --checkpoint_every 20 --steps_per_episode 300
```

## Cell 3 (L7)
```python
!python train.py --scenario standard --ablation l7 --seed 0 --episodes 100 --checkpoint_every 20 --steps_per_episode 300
```

## Cell 4 (Batch seeds)
```python
!python run_seed_batch.py --scenario standard --ablation full --seed_start 0 --seed_end 5 --episodes 300 --resume --skip_existing
!python run_seed_batch.py --scenario standard --ablation l7 --seed_start 0 --seed_end 5 --episodes 300 --resume --skip_existing
```

## Cell 5 (Collect)
```python
!python collect_results.py --scenario standard
!cat results/ablation_table.txt
```

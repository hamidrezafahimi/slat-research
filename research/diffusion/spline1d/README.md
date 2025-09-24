The followings are ran in sequence. Remember that the inputs args must match.

```bash
python3 generate.py --noisy n.csv --clean c.csv --plot

python3 visualize_scores.py --file n.csv --max-dist 1 --with-heatmap

python3 search_splines.py --file n.csv --plot --max-dist 1 --n-splines 1000 --save-best-ctrl best.csv --K-ctrl 4

# (optional)
python3 visualize_scores.py --file n.csv --max-dist 1 --spline-ctrl best.csv --gt-spline c.csv --with-heatmap

python3 optim.py --file n.csv --max-dist 1 --spline-ctrl best.csv --with-heatmap --iters 400 --fast

# (optional - optimization with regularization)
python3 optimReg.py --file n.csv --max-dist 1 --spline-ctrl best.csv --with-heatmap --iters 400 --fast
```
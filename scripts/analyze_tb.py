"""Read tensorboard event files and dump all scalar data."""

from tbparse import SummaryReader

# Read the lstm128 run (non-pivot to avoid merge issues)
reader = SummaryReader("battle_royale/runs/lstm128")
df = reader.scalars

print(f"Total data points: {len(df)}")
print(f"Step range: {df['step'].min()} .. {df['step'].max()}")

tags = sorted(df['tag'].unique())
print(f"Metrics: {tags}\n")

for tag in tags:
    series = df[df['tag'] == tag].sort_values('step')
    vals = series['value']
    steps = series['step']
    print(f"--- {tag} ---")
    print(f"  steps: {int(steps.iloc[0])} .. {int(steps.iloc[-1])}  ({len(series)} points)")
    print(f"  range: {vals.min():.4f} .. {vals.max():.4f}")
    print(f"  mean:  {vals.mean():.4f}  std: {vals.std():.4f}")
    # Last 10 values
    tail = series.tail(10)
    for _, row in tail.iterrows():
        print(f"    step {int(row['step']):>5d}: {float(row['value']):.4f}")
    print()

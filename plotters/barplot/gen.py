import altair as alt
from vega_datasets import data
from altair_saver import save
import pandas as pd

FONT_SIZE = 12
SORT = ["Terminal", "Retr. Term.", "Embedding", "Distance",
"Combined", "PPO Dist.", "PPO. Embed."]
SORT_M = ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'MRR']
range_ = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600']

header = ["value", "reward", "metric"]
data = [
[0.355, "Terminal", "hits@1"],
[0.732, "Terminal", "hits@3"],
[0.892, "Terminal", "hits@5"],
[0.988, "Terminal", "hits@10"],
[0.5601, "Terminal", "MRR"],
[0.568, "Retr. Term.", "hits@1"],
[0.912, "Retr. Term.", "hits@3"],
[0.984, "Retr. Term.", "hits@5"],
[1.0, "Retr. Term.", "hits@10"],
[0.7445, "Retr. Term.", "MRR"],
[0.758, "Embedding", "hits@1"],
[0.982, "Embedding", "hits@3"],
[0.998, "Embedding", "hits@5"],
[1.0, "Embedding", "hits@10"],
[0.8626, "Embedding", "MRR"],
[0.682, "Distance", "hits@1"],
[0.974, "Distance", "hits@3"],
[1.0, "Distance", "hits@5"],
[1.0, "Distance", "hits@10"],
[0.8235, "Distance", "MRR"],
[0.592, "Combined", "hits@1"],
[0.86, "Combined", "hits@3"],
[0.956, "Combined", "hits@5"],
[1.0, "Combined", "hits@10"],
[0.6860, "Combined", "MRR"],
[0.792, "PPO Dist.", "hits@1"],
[0.965, "PPO Dist.", "hits@3"],
[1.0, "PPO Dist.", "hits@5"],
[1.0, "PPO Dist.", "hits@10"],
[0.8626, "PPO Dist.", "MRR"],
[0.830, "PPO. Embed.", "hits@1"],
[0.989, "PPO. Embed.", "hits@3"],
[1.0, "PPO. Embed.", "hits@5"],
[1.0, "PPO. Embed.", "hits@10"],
[0.9029, "PPO. Embed.", "MRR"]
]

df = pd.DataFrame(data)
df.columns = header

print(df)

chart = alt.Chart(df).mark_bar().encode(
    alt.X('reward:N', title='', sort=SORT),
    alt.Y('value:Q', title='', sort=SORT_M),
    color=alt.Color('reward:N', scale=alt.Scale(domain=SORT, range=range_), legend=None),
    column='metric:N'
).configure(
    font='arial narrow'
).configure_axis(
    labelFontSize=FONT_SIZE,
    titleFontSize=FONT_SIZE
)

save(chart, f"barchart.png")
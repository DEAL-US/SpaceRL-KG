import pandas as pd
import altair as alt
from vega_datasets import data
from altair_saver import save

FONT_SIZE = 12

class plotter():
    def __init__(self, filename, header, data, groups, bar_names, colors):
        self.header = header
        self.data = data
        self.groups = groups
        self.bar_names = bar_names
        self.colors = colors
        self.filename = filename

        self.df = pd.DataFrame(data)
        self.df.columns = header

        print(f"Initialized plotter with data:\n{self.df}")

    def generate(self):
        chart = alt.Chart(self.df).mark_bar().encode(
        alt.X(f'{self.header[1]}:N', title='', sort=self.bar_names),
        alt.Y(f'{self.header[0]}:Q', title='', sort=self.groups),
        color=alt.Color(f'{self.header[1]}:N', scale=alt.Scale(domain=self.bar_names, range=self.colors), legend=None),
        column=f'{self.header[0]}:N'
        ).configure(
            font='arial narrow'
        ).configure_axis(
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE
        )

        save(chart, f"{self.filename}.png")

countries_trans_e = plotter(filename="countries_trans_e",
header = ["value", "reward", "metric"],
data = [[0.355, "Terminal", "hits@1"],
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
[0.9029, "PPO. Embed.", "MRR"]],
groups = ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'MRR'],
bar_names = ["Terminal", "Retr. Term.", "Embedding", "Distance", "Combined", "PPO Dist.", "PPO. Embed."],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600'])
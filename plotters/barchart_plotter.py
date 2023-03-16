import pandas as pd
import altair as alt
from vega_datasets import data
from altair_saver import save
import pathlib

FONT_SIZE = 12

local_dir = pathlib.Path(__file__).parent.resolve()

class barchart_plotter():
    def __init__(self, filename, header, data, groups, bar_names, colors):
        self.header = header
        self.data = data
        self.groups = groups
        self.bar_names = bar_names
        self.colors = colors
        self.filename = filename

        self.df = pd.DataFrame(data)
        self.df.columns = header

    def generate(self):
        print(f"generating plot with data:\n{self.df}")

        chart = alt.Chart(self.df).mark_bar().encode(
        alt.X(f'{self.header[1]}:N', title='', sort=self.bar_names),
        alt.Y(f'{self.header[0]}:Q', title='', sort=self.groups),
        color=alt.Color(f'{self.header[1]}:N',
        scale=alt.Scale(domain=self.bar_names, range=self.colors), legend=None),
        column=f'{self.header[2]}:N'
        ).configure(
            font='arial narrow'
        ).configure_axis(
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE
        )

        save(chart, f"{local_dir}/results/{self.filename}.html")


class logarithmic_bar_plot():
    def __init__(self, filename, header, data, colors):
        self.header = header
        self.filename = filename
        self.data = data
        self.colors = colors

        self.df = pd.DataFrame(data, columns = header)

        self.names = list(self.df[header[0]])

    def generate(self):
        chart = alt.Chart(self.df).mark_bar().encode(
        x=alt.X(
            f'{self.header[0]}:O',
            sort = self.names
        ),
        y=alt.Y(
            f'{self.header[1]}',
            scale=alt.Scale(type="log")
        ),
        color=alt.Color(
            f'{self.header[0]}:N',
            legend=None,
            scale=alt.Scale(domain=self.names, range=self.colors),
        )).configure(
            font='arial narrow'
        ).configure_axis(
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE
        )

        save(chart, f"{local_dir}/results/{self.filename}.html")


relation_appearances = logarithmic_bar_plot(
filename = "relation_appearances", 
header = ["rel name", "rel #"], 
data = [
["similar_to", 80],
["verb_group", 1138],
["also_see", 1299],
["derivationally_related_from", 29715],
["GENERIC", 80798]],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600']
)

MRR_relations = logarithmic_bar_plot(
filename = "MRR_relations", 
header = ["rel name", "MRR"], 
data = [
['similar_to', 0.909846667],
['verb_group', 0.771853889],
['also_see', 0.745857619],
['derivationally_related_from', 0.637527063],
['GENERIC', 0.443105317]
],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600']
)

countries_trans_e = barchart_plotter(filename="countries_trans_e",
header = ["value", "reward", "metric"],
data = [[0.355, "Terminal", "hits@01"],
[0.732, "Terminal", "hits@03"],
[0.892, "Terminal", "hits@05"],
[0.988, "Terminal", "hits@10"],
[0.5601, "Terminal", "MRR"],
[0.568, "Retr. Term.", "hits@01"],
[0.912, "Retr. Term.", "hits@03"],
[0.984, "Retr. Term.", "hits@05"],
[1.0, "Retr. Term.", "hits@10"],
[0.7445, "Retr. Term.", "MRR"],
[0.758, "Embedding", "hits@01"],
[0.982, "Embedding", "hits@03"],
[0.998, "Embedding", "hits@05"],
[1.0, "Embedding", "hits@10"],
[0.8626, "Embedding", "MRR"],
[0.682, "Distance", "hits@01"],
[0.974, "Distance", "hits@03"],
[1.0, "Distance", "hits@05"],
[1.0, "Distance", "hits@10"],
[0.8235, "Distance", "MRR"],
[0.592, "Combined", "hits@01"],
[0.86, "Combined", "hits@03"],
[0.956, "Combined", "hits@05"],
[1.0, "Combined", "hits@10"],
[0.6860, "Combined", "MRR"],
[0.792, "PPO Dist.", "hits@01"],
[0.965, "PPO Dist.", "hits@03"],
[1.0, "PPO Dist.", "hits@05"],
[1.0, "PPO Dist.", "hits@10"],
[0.8626, "PPO Dist.", "MRR"],
[0.830, "PPO. Embed.", "hits@01"],
[0.989, "PPO. Embed.", "hits@03"],
[1.0, "PPO. Embed.", "hits@05"],
[1.0, "PPO. Embed.", "hits@10"],
[0.9029, "PPO. Embed.", "MRR"]],
groups = ['hits@01', 'hits@03', 'hits@05', 'hits@10', 'MRR'],
bar_names = ["Terminal", "Retr. Term.", "Embedding", "Distance", "Combined", "PPO Dist.", "PPO. Embed."],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600'])


nell_thing_has_color = barchart_plotter(filename="nell_thing_has_color",
header = ["value", "config", "metric"],
data = [
[0.445, "BASE-DISTANCE", "hits@01",],
[0.804, "BASE-DISTANCE", "hits@03",],
[0.908, "BASE-DISTANCE", "hits@05",],
[0.949, "BASE-DISTANCE", "hits@10",],
[0.6923, "BASE-DISTANCE", "MRR"],

[0.588, "BASE-EMBEDDING", "hits@01"],
[0.897, "BASE-EMBEDDING", "hits@03"],
[0.964, "BASE-EMBEDDING", "hits@05"],
[0.987,  "BASE-EMBEDDING", "hits@10"],
[0.7711, "BASE-EMBEDDING", "MRR"],

[0.551, "PPO-DISTANCE", "hits@01"],
[0.869, "PPO-DISTANCE", "hits@03"],
[0.947, "PPO-DISTANCE", "hits@05"],
[0.983, "PPO-DISTANCE", "hits@10"],
[0.7335, "PPO-DISTANCE", "MRR"],

[0.634, "PPO-EMBEDDING", "hits@01"],
[0.953, "PPO-EMBEDDING", "hits@03"],
[0.996, "PPO-EMBEDDING", "hits@05"],
[0.999, "PPO-EMBEDDING", "hits@10"],
[0.7889, "PPO-EMBEDDING", "MRR"],

],
groups = ['hits@01', 'hits@03', 'hits@05', 'hits@10', 'MRR'],
bar_names = ["BASE-DISTANCE","BASE-EMBEDDING","PPO-DISTANCE", "PPO-EMBEDDING"],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600'])

nell_is_taller = barchart_plotter(filename="nell_is_taller",
header = ["value", "config", "metric"],
data = [
[0.410, "BASE-DISTANCE", "hits@01",],
[0.770, "BASE-DISTANCE", "hits@03",],
[0.873, "BASE-DISTANCE", "hits@05",],
[0.915, "BASE-DISTANCE", "hits@10",],
[0.6909, "BASE-DISTANCE", "MRR"],

[0.585, "BASE-EMBEDDING", "hits@01"],
[0.893, "BASE-EMBEDDING", "hits@03"],
[0.960, "BASE-EMBEDDING", "hits@05"],
[0.983,  "BASE-EMBEDDING", "hits@10"],
[0.7690, "BASE-EMBEDDING", "MRR"],

[0.548, "PPO-DISTANCE", "hits@01"],
[0.866, "PPO-DISTANCE", "hits@03"],
[0.944, "PPO-DISTANCE", "hits@05"],
[0.980, "PPO-DISTANCE", "hits@10"],
[0.7313, "PPO-DISTANCE", "MRR"],

[0.630, "PPO-EMBEDDING", "hits@01"],
[0.950, "PPO-EMBEDDING", "hits@03"],
[0.993, "PPO-EMBEDDING", "hits@05"],
[1.0, "PPO-EMBEDDING", "hits@10"],
[0.7868, "PPO-EMBEDDING", "MRR"],

],
groups = ['hits@01', 'hits@03', 'hits@05', 'hits@10', 'MRR'],
bar_names = ["BASE-DISTANCE","BASE-EMBEDDING","PPO-DISTANCE", "PPO-EMBEDDING"],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600'])

nell_music_artist_genre = barchart_plotter(filename="nell_music_artist_genre",
header = ["value", "config", "metric"],
data = [

[0.409, "BASE-DISTANCE", "hits@01",],
[0.769, "BASE-DISTANCE", "hits@03",],
[0.857, "BASE-DISTANCE", "hits@05",],
[0.898, "BASE-DISTANCE", "hits@10",],
[0.6496, "BASE-DISTANCE", "MRR"],

[0.583, "BASE-EMBEDDING", "hits@01"],
[0.892, "BASE-EMBEDDING", "hits@03"],
[0.944, "BASE-EMBEDDING", "hits@05"],
[0.967,  "BASE-EMBEDDING", "hits@10"],
[0.7599, "BASE-EMBEDDING", "MRR"],

[0.547, "PPO-DISTANCE", "hits@01"],
[0.865, "PPO-DISTANCE", "hits@03"],
[0.928, "PPO-DISTANCE", "hits@05"],
[0.964, "PPO-DISTANCE", "hits@10"],
[0.7460, "PPO-DISTANCE", "MRR"],

[0.629, "PPO-EMBEDDING", "hits@01"],
[0.949, "PPO-EMBEDDING", "hits@03"],
[0.976, "PPO-EMBEDDING", "hits@05"],
[0.984, "PPO-EMBEDDING", "hits@10"],
[0.7755, "PPO-EMBEDDING", "MRR"],

],
groups = ['hits@01', 'hits@03', 'hits@05', 'hits@10', 'MRR'],
bar_names = ["BASE-DISTANCE","BASE-EMBEDDING","PPO-DISTANCE", "PPO-EMBEDDING"],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600'])


fb_film_genre = barchart_plotter(filename="fb_film_genre",
header = ["value", "config", "metric"],
data = [
[0.133, "BASE-DISTANCE", "hits@01",],
[0.252, "BASE-DISTANCE", "hits@03",],
[0.315, "BASE-DISTANCE", "hits@05",],
[0.38, "BASE-DISTANCE", "hits@10",],
[0.2567, "BASE-DISTANCE", "MRR"],

[0.167, "BASE-EMBEDDING", "hits@01"],
[0.282, "BASE-EMBEDDING", "hits@03"],
[0.348, "BASE-EMBEDDING", "hits@05"],
[0.405,  "BASE-EMBEDDING", "hits@10"],
[0.2672, "BASE-EMBEDDING", "MRR"],

[0.232, "PPO-DISTANCE", "hits@01"],
[0.376, "PPO-DISTANCE", "hits@03"],
[0.423, "PPO-DISTANCE", "hits@05"],
[0.516, "PPO-DISTANCE", "hits@10"],
[0.3829, "PPO-DISTANCE", "MRR"],

[0.250, "PPO-EMBEDDING", "hits@01"],
[0.395, "PPO-EMBEDDING", "hits@03"],
[0.447, "PPO-EMBEDDING", "hits@05"],
[0.552, "PPO-EMBEDDING", "hits@10"],
[0.4082, "PPO-EMBEDDING", "MRR"],

],
groups = ['hits@01', 'hits@03', 'hits@05', 'hits@10', 'MRR'],
bar_names = ["BASE-DISTANCE","BASE-EMBEDDING","PPO-DISTANCE", "PPO-EMBEDDING"],
colors = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600'])


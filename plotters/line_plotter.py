import pandas as pd
import altair as alt
from vega_datasets import data
from altair_saver import save
import pathlib

FONT_SIZE = 12
local_dir = pathlib.Path(__file__).parent.resolve()

class line_plotter():
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

    def generate_logarithmic(self):
        chart = alt.Chart(source).mark_line().encode(
            x='year:O',
            y=alt.Y(
                'sum(people)',
                scale=alt.Scale(type="log")
            )
        )

        save(chart, f"{local_dir}/results/{self.filename}.html")


header = [["MRR", "rel name"]],
data = [
[0.909846667, "similar_to"],
[0.771853889, "verb_group"],
[0.745857619, "also_see"],
[0.637527063, "derivationally_related_from"],
[0.443105317, "GENERIC"],
]

df = pd.DataFrame(data=data, columns=header)

header = ["rel # ", "rel name"],
data = [
[80, "similar_to"],
[1138, "verb_group"],
[1299, "also_see"],
[29715, "derivationally_related_from"],
[80798, "GENERIC"],
]

df = pd.DataFrame(data)
df.columns = header

print(df)

filename = "test"

chart = alt.Chart(df).mark_line().encode(
    x=f'{header[1]}:O',
    y=alt.Y(
        f'{header[0]}',
        scale=alt.Scale(type="log")
    )
)

save(chart, f"{local_dir}/results/{filename}.html")
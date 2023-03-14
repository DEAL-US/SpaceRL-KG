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


data = [['similar_to', 0.909846667],
['verb_group', 0.771853889],
['also_see', 0.745857619],
['derivationally_related_from', 0.637527063],
['GENERIC', 0.443105317]]
header = ['rel name', 'MRR']

df1 = pd.DataFrame(data, columns = header )

data = [
["similar_to", 80],
["verb_group", 1138],
["also_see", 1299],
["derivationally_related_from", 29715],
["GENERIC", 80798]]
header = ["rel name", "rel #"] 

df2 = pd.DataFrame(data, columns = header)

print(df1)
print(df2)

filename = "test"

chart = alt.Chart(df2).mark_bar().encode(
    x=alt.X(
        f'{header[0]}:O',
        sort = ["similar_to", "verb_group", "also_see", "derivationally_related_from", "GENERIC"]
    ),
    y=alt.Y(
        f'{header[1]}',
        scale=alt.Scale(type="log")
    ),
    color=alt.Color(
        f'{header[0]}:N',
        scale=alt.Scale(domain=["similar_to", "verb_group", "also_see", "derivationally_related_from", "GENERIC"],
        range=['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600']), legend=None),
).configure(
    font='arial narrow'
).configure_axis(
    labelFontSize=FONT_SIZE,
    titleFontSize=FONT_SIZE
)

save(chart, f"{local_dir}/results/{filename}.html")
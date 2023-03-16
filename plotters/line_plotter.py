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

filename = "MRRvsRelNum"

header = ['rel name', 'rel#', 'MRR']

data = [["similar_to (80)", 80, 0.909846667],
["verb_group (1138)", 1138, 0.771853889],
["also_see (1299)", 1299, 0.745857619],
["derivationally_related_from (29715)", 29715, 0.637527063],
["GENERIC (80798)", 80798, 0.443105317]]

df = pd.DataFrame(data, columns = header)

axis_labels = (
    """datum.label == 80 ? 'similar_to (80)'
    : datum.label == 1138 ? 'verb_group (1138)'
    : datum.label == 1299 ? 'also_see (1299)'
    : datum.label == 29715 ? 'derivationally_related_from (29715)'
    : datum.label == 80798 ? 'GENERIC (80798)'"""
)

chart = alt.Chart(df).mark_line(point = True).encode(
    x=alt.X(
        f'{header[1]}:O',
        axis = alt.Axis(title = ["similar_to (80)", "verb_group (1138)", "also_see (1299)",
        "deriv_rel_from (29715)", "GENERIC (80798)"], labelAngle=0),

    ),
    y=alt.Y(
        f'{header[2]}',
        scale=alt.Scale(domain=[0.4, 1])
    )
).configure(
    font='arial narrow'
).configure_axis(
    labelFontSize=FONT_SIZE,
    titleFontSize=FONT_SIZE
).properties(
    width=200,
    height=200
)

save(chart, f"{local_dir}/results/{filename}.html")
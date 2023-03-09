import altair as alt
from vega_datasets import data
from altair_saver import save
import pandas as pd
import plotdata

FONT_SIZE = 12
SORT = ["Terminal", "Retr. Term.", "Embedding", "Distance",
"Combined", "PPO Dist.", "PPO. Embed."]
SORT_M = ['hits@1', 'hits@3', 'hits@5', 'hits@10', 'MRR']
range_ = ['#003f5c', '#374c80', '#7a5195', '#bc5090', '#ef5675','#ff764a','#ffa600']

header = ["value", "reward", "metric"]
data = plotdata.countries_trans_e

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
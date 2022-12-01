import altair as alt
from altair_saver import save
import pandas as pd

SORT = ["Terminal", "Retr. Term.", "Embedding", "Distance",
"Combined", "PPO Dist.", "PPO. Embed."]
FONT_SIZE = 12
range_ = ['#450a5c', '#249f87', '#f5e626',
'#450a5c', '#249f87', '#f5e626','#450a5c']

data=dict()
i=0
with open("./data.txt", 'r') as f:
    for line in f:
        a = line.split(',')
        res = [float(i) for i in a]
        data[SORT[i]]=res
        i+=1

df = pd.DataFrame(data)
df2 = df.mean()
print(df2)
df3= df.std()
print(df3)


chart = alt.Chart(
        df.reset_index().melt(value_vars=["Terminal", "Retr. Term.", "Embedding", "Distance", "Combined", "PPO Dist.", "PPO. Embed."])
    ).mark_boxplot(
        extent='min-max',
        size=20,
    ).encode(
        alt.X('variable:N', title='', sort=SORT),
        alt.Y('value:Q', title='', scale=alt.Scale(domain=[1, 10])),
        #column=alt.Column('tech:N', title='', header=alt.Header(labelFontSize=FONT_SIZE)),
        color=alt.Color('variable:N', sort=SORT, scale=alt.Scale(domain=SORT, range=range_), legend=None)
    ).configure(
        font='arial narrow'
    ).configure_axis(
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE
    ).properties(
        height=500,
        width=300,
    )

save(chart, f"boxplot.png")
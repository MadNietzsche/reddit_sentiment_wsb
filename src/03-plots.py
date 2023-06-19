import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df_posts = pd.read_csv('../data/senti_posts.zip').drop(6161)  # bad row
df_posts['timestamp'] = pd.to_datetime(df_posts['timestamp'])
df_posts['score'] = pd.to_numeric(df_posts['score'])
df_posts['hour_day'] = pd.to_datetime(df_posts['timestamp'].dt.strftime('%Y-%m-%dT%H'))

df_comments = pd.read_csv('../data/senti_comments.zip', lineterminator='\n') \
    .merge(df_posts[['id', 'hour_day']], left_on='id_col', right_on='id') \
    .drop('id', axis=1)


col_name = ['hour_day', 'count']

num_posts = df_posts.groupby('hour_day')['id'].nunique().reset_index()
num_posts.columns = col_name

num_comments = df_comments.groupby('hour_day').size().to_frame().reset_index()
num_comments.columns = col_name

day_of_squeeze = num_posts[num_posts['count']==np.max(num_posts['count'])]['hour_day'].to_list()[0]
peak_comments = num_comments[num_comments['count']>=np.max(num_comments['count'])]['hour_day'].to_list()[0]

# Number of posts and comments
plt.rcParams["figure.figsize"] = [10, 5]
fig, ax = plt.subplots(2, 1, sharex = True)

ax[0].plot(num_posts['hour_day'], num_posts['count']/1000)
ax[0].set_ylabel('Number of Posts\n(by Thousands)', fontsize=8)
ax[0].annotate('29th of January', xy=(day_of_squeeze, np.max(num_posts['count']/1000)),
               xytext=(4.5, -5.5), textcoords='offset points', color='crimson')

ax[1].plot(num_comments['hour_day'], num_comments['count']/1000)
ax[1].set_ylabel('Number of Comments\n(by Thousands)', fontsize=8)
ax[1].set_xlabel('Submission Date')

ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
plt.xticks(fontsize=6)
for label in ax[1].get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

plt.tight_layout()
fig.savefig('../img/n_posts_comments.png')
plt.close(fig)

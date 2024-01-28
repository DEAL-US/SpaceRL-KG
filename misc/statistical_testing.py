from scipy.stats import ks_2samp, wilcoxon
import os
from tqdm import tqdm

testing_results = "/home/mike/SpaceRL-KG/model/data/results"

ALPHA = 0.05
METRICS_OUTPUT_FILE = f"{testing_results}/_STATISTIC_TEST"

# THIS IS A HARDCODED TEST SPECIFIC FOR THE RESULTS PRESENT IN THE REPOSITORY.

f=[]
for _, dirnames, _  in os.walk(testing_results):
	f.extend(dirnames)

group_table_8 = ["film_genre_FB_PPO_distance_22", "film-genre-FB-base-simple-distance-22-test", "film-genre-FB-base-simple-embedding-22-test", "film-genre-FB-PPO-simple-embedding-22-test"]

group_table_9_1 = ["has-color-nell-base-emb-250", "has-color-nell-base-dist-250", "has-color-nell-ppo-dist-250", "has-color-nell-ppo-emb-250"]
group_table_9_2 = ["is-taller-nell-base-dist-250", "is-taller-nell-base-emb-250", "is-taller-nell-ppo-dist-250", "is-taller-nell-ppo-emb-250"]
group_table_9_3 = ["music_artist_genre-nell-base-dist-150", "music_artist_genre-nell-base-emb-150", "music_artist_genre-nell-ppo-dist-150", "music_artist_genre-nell-ppo-emb-150",]

group_table_10 = ["similar-to-wn18-PPO-simple-emb-100", "verb-group-wn18-PPO-simple-emb-100", "also-see-wn18-PPO-simple-emb-100", "derivationally-related-from-wn18-PPO-simple-emb-100", "WN18_generic_PPO_simple_embedding_50"]

winner_idx = [3, 3, 3, 3, 4]
res = ""
for idx, group in enumerate([group_table_8, group_table_9_1, group_table_9_2, group_table_9_3, group_table_10]):
	winner_path = f"{testing_results}/{group[winner_idx[idx]]}/res.txt"
	with open(winner_path, 'r') as file:
		best_array = eval(file.readline())

	for exp_name in tqdm(group):
		data_path = f"{testing_results}/{exp_name}/res.txt"
		if(data_path != winner_path):

			with open(data_path, 'r') as file:
				curr_array = eval(file.readline())
		
			p_wilcoxon = 1.0
			try:
				p_wilcoxon = wilcoxon(best_array, curr_array).pvalue
			except ValueError:
				print("Value error when computing Wilcoxon test. Probably caused by exactly similar distributions.")
			except:
				print("yeah something went wrong asf, u better debug.")
				quit()

			if(p_wilcoxon < ALPHA):

				# print("There are significant differences (paired)")
				res += f"{exp_name}: +\n"
			else:
				res += f"{exp_name}: =\n"
			
		else:
			res += f"{exp_name}: blank\n"
	
	res += '\n'


print(res)

with open(METRICS_OUTPUT_FILE, "w") as file:
	file.writelines(res)
import ast

lmao = "{'dataset': 'KINSHIP', 'models': ['TransE_l2'], 'use_gpu': True, 'regenerate_existing': False, 'normalize': True, 'add_inverse_path': True, 'fast_mode': False}"
# lmao = lmao.replace("\'", "\"")
# ey = json.loads(lmao)
res = ast.literal_eval(lmao)
print(res)
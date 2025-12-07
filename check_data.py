import pickle  
  

print("===== ref_name.pkl =====")
with open("ref_name.pkl", "rb") as f:
    print(pickle.load(f))

print("\n===== ref_embed.pkl =====")
with open("ref_embed.pkl", "rb") as f:
    data = pickle.load(f)
    print({k: len(v) for k, v in data.items()})



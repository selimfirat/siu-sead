import os

res = "type,dataset,model,roc,ap\n"

for experiment_dir in os.listdir("experiments"):

    if not os.path.isdir(os.path.join("experiments", experiment_dir)):
        continue
    t1, t2, dataset, model = experiment_dir.split("_")

    type = t1 + "_" + t2

    if os.path.exists(f"experiments/{experiment_dir}/scores.csv"):
        with open(f"experiments/{experiment_dir}/scores.csv", "r") as f:

            print(f.readlines()[0])
            line = f.readlines()[0].split("\\n")[1]

            roc, ap = map(float, line.split(","))
    else:
        roc, ap = -1, -1 # to detect bugs if there is

    res += f"{type},{dataset},{model},{str(roc)},{ap}\n"


with open("figures/all_results.csv", "w+") as f:
    f.write(res)

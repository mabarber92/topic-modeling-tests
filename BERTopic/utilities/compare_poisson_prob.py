import seaborn as sns
import pandas as pd
import os

def compare_poisson_prob(csv_dir, image_path):
    sns.set_style("darkgrid")
    # Create df to be use for drawing line chart
    df = pd.DataFrame()
    for root, dirs, files in os.walk(csv_dir, topdown=False):
        for name in files:
            csv_path = os.path.join(root, name)
            field_name = name.split(".")[0]
            case_name = name.split("-")[0]
            pois_df = pd.read_csv(csv_path)
            pois_df["case"] = case_name
            pois_df["field"] = field_name
            df = pd.concat([df, pois_df])
    
    g = sns.FacetGrid(df, col="case")
    g.map_dataframe(sns.lineplot, "year", "poisson_prob", hue="field", linewidth=0.5)
    g.set(xticks= list(range(0, 1000, 25)))
    labels = [str(x) if str(x)[-1] == '0' else '' for x in range(0, 1000, 25)]
    print(labels)
    g.set_xticklabels(labels = labels, rotation=90)
    g.tick_params(axis='x', labelsize='small', direction='out')
    figure = g.figure
    figure.savefig(image_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    csv_dir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/poisson-calcs/tok_count_csvs/"
    compare_poisson_prob(csv_dir, "comp_poisson_probs.png")
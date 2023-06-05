import pandas as pd

def merge_tag_df_on_index(original_df, left_csv):
    left_df = pd.read_csv(left_csv, index_col=0)
    merged = left_df.merge(original_df, how= "left", left_index=True, right_index=True, suffixes=("_old", ""))
    merged = merged.drop(columns=["unique_tags_old", "counts_old", "uris_old"])
    return merged

if __name__ == "__main__":
    file_list = ["C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/Yusuf-hadith-all-genre-tags.csv",
                 "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/Yusuf-hadith-history-tags.csv",
                 "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/Yusuf-hadith-top-10-tags.csv",
                 "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/Yusuf-hadith-top-hadith-tags.csv"
                 ]
    original_df = pd.read_csv("C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/Yusuf-hadith-top-tags.csv", index_col=0)
    for file in file_list:
        new_df = merge_tag_df_on_index(original_df, file)
        new_path = file.split("/")[-1]
        print(new_path)
        new_df.to_csv(new_path, encoding='utf-8-sig')

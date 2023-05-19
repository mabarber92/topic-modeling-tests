import pandas as pd
import sys
from tqdm import tqdm


def check_and_merge(ms_dict, new_ms_df):
        candidates = new_ms_df[new_ms_df["idCol"] == ms_dict["idCol"]].to_dict("records")
        if len(candidates) == 1:
            ms_dict["corrected-ms"] = candidates[0]["ms"]
        elif len(candidates) > 1:
            gaps = []
            for candidate in candidates:
                gaps.append(0 + int(candidate["ms"]) - int(ms_dict["ms"]))
            closest_idx = gaps.index(min(gaps))
            ms_dict["corrected-ms"] = candidates[closest_idx]["ms"]
        else:
            print("No match found!")
            ms_dict["corrected-ms"] = 0
        return ms_dict

def merge_corrected_ms(old_ms_csv, new_ms_csv, out_path, id_fields = ["phrase", "uri"]):
    """CPU intensive approach, but guarantees that we can sort out cases where there are two matches for a text string"""
    # Load dataframes

    old_ms_df = pd.read_csv(old_ms_csv)
    new_ms_df = pd.read_csv(new_ms_csv)

    # Create unique fields to merge on

    old_ms_df["idCol"] = old_ms_df[id_fields[0]] + "." + old_ms_df[id_fields[1]].astype(str)
    new_ms_df["idCol"] = new_ms_df[id_fields[0]] + "." + new_ms_df[id_fields[1]].astype(str)

    # We have multiple possible matches - so loop through manually and for cases where we have multiple matches, use the closest ms

    old_ms_dicts = old_ms_df.to_dict("records")
    
    new_ms_dict = [check_and_merge(ms_dict, new_ms_df) for ms_dict in tqdm(old_ms_dicts)]
    print(new_ms_dict)
    # for ms_dict in tqdm(old_ms_dicts):
    #     candidates = new_ms_df[new_ms_df["idCol"] == ms_dict["idCol"]].to_dict("records")
    #     if len(candidates) == 1:
    #         ms_dict["corrected-ms"] = candidates[0]["ms"]
    #     elif len(candidates) > 1:
    #         gaps = []
    #         for candidate in candidates:
    #             gaps.append(0 + int(candidate["ms"]) - int(ms_dict["ms"]))
    #         closest_idx = gaps.index(min(gaps))
    #         ms_dict["corrected-ms"] = candidates[closest_idx]["ms"]
    #     else:
    #         print("No match found!")
    #         ms_dict["corrected-ms"] = 0
    
    # Transform the final dict into a dataframe

    df_out = pd.DataFrame(new_ms_dict)
    df_out = df_out.drop(columns=["idCol"])

    # Check for duplications
    print(len(old_ms_df))
    print(len(df_out))

    # Output

    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    if len(sys.argv) == 4:
        old_ms_csv = sys.argv[1]
        new_ms_csv = sys.argv[2]
        out_path = sys.argv[3]
    else:
        print("Give path to main csv")
        old_ms_csv = input().replace("\\", "/")

        print("Give path to second csv")
        new_ms_csv = input().replace("\\", "/")

        print("Give path for output")
        out_path = input().replace("\\", "/")
    
    merge_corrected_ms(old_ms_csv, new_ms_csv, out_path)


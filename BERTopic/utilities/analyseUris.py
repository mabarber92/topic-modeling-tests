import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import poisson
import re


class getTags():
    def __init__ (self, input, col = "tags", input_df = False, use_pri = False, use_all_versions=False):
        if not input_df:
            input = pd.read_csv(meta_csv, sep="\t")
        if use_pri:
            input = input[input["status"] == "pri"]
        elif use_all_versions:
            book_list = input["book"].drop_duplicates().tolist()
            filtered_meta_dicts = []
            for book in book_list:                
                versions_df = input[input["book"] == book]
                all_tags = versions_df["tags"].drop_duplicates().dropna().tolist()
                # If all versions return nan in their tags field, then we give them 'None' in new df
                if len(all_tags) > 0:
                    all_tags = " :: ".join(all_tags)
                else:
                    all_tags = 'None'    
                filtered_meta_dicts.append({"book": book, "tags" : all_tags})
            filtered_df = pd.DataFrame(filtered_meta_dicts)
            # Merge in new tag categories
            input = input.drop(columns=["tags"])
            input = input.merge(filtered_df, on="book", how="left")
            input = input[input["status"] == "pri"]
        self.meta_df = input
    
            
        self.fetch_col(col)
    def fetch_col(self, col="tags"):
        self.col_series = self.meta_df[col]
    def list_all_unique(self):
        return self.get_unique_sort().tolist()
    def df_all_unique(self, col_name = "unique_tags"):
        return self.get_unique_sort(col_name)
    def get_unique_sort(self, col_name = "unique_tags"):
        df = pd.DataFrame()
        df[col_name] = list(dict.fromkeys(self.list_of_all_tags()))
        return df.sort_values(by = col_name)
    def count_tags(self, col_name="unique_tags", texts_list = False):
        df = pd.DataFrame()
        df[col_name] = self.list_of_all_tags()
        df = df.value_counts().rename_axis(col_name).reset_index(name='counts')
        if texts_list:
            df = self.text_list_per_topic(df, col_name)
        return df
    def list_of_all_tags(self):
        final_list = []
        list_of_rows = self.col_series.dropna().tolist()
        for row in list_of_rows:            
            tag_list = re.split("\s*::\s+", row)
            # Check there are no duplicated tags in the input row (imp if we're using all_versions as input)
            tag_list = list(set(tag_list))
            final_list.extend(tag_list)
        if "" in final_list:
            print("Space found")
            final_list = [i for i in final_list if i != " "]
        return final_list
    def text_list_per_topic(self, df, col_name):
        df_dict = df.to_dict("records")
        for row in df_dict:
            tag_match = self.meta_df[self.meta_df["tags"].str.contains(row[col_name])]["book"].to_list()
            row["uris"] = tag_match
        return pd.DataFrame(df_dict)

    # For future - set up class so obj can be filtered like it is a dataframe

class uriTopicMetadata():
    sns.set_style("darkgrid")

    def __init__(self, meta_csv, topic_csv, topic_filter = [], graphing_par = "uri-ms"):
        meta_df = pd.read_csv(meta_csv, sep="\t")
        self.meta_df_full = meta_df.copy()
        self.meta_df = meta_df[meta_df["status"] == "pri"]
        self.meta_df["ms-total"] = self.meta_df["tok_length"]/300
        self.topic_uri_df = pd.read_csv(topic_csv)[["Topic", "uri", "ms"]].drop_duplicates()
        
        if len(topic_filter) > 0:            
            self.filter_topics(topic_filter)

        self.topic_uri_df["uri-ms"] = self.topic_uri_df["uri"] + self.topic_uri_df["ms"].astype("str")
        self.topic_uri_df = self.merge_meta(self.topic_uri_df)
        self.graphing_par = graphing_par
        self.tags = None

    def merge_meta(self, left, on = "uri"):
        if on == "uri":
            left["uri"] = left["uri"].str.split("\.").str[:3].str.join(".")
            df = left.merge(self.meta_df, left_on="uri", right_on="version_uri", how="left")
            df.drop(columns = ["version_uri"])
        elif on == "book":
            left["book"] = left["uri"].str.split("\.").str[:2].str.join(".")
            df = left.merge(self.meta_df, on ="book", how="left")
        return df

    # Function to return all metadata entries for a set book (all versions)
    def filter_meta_on_book(self, filter_df):
        book_list = filter_df["book"].drop_duplicates().to_list()
        return self.meta_df_full[self.meta_df_full["book"].isin(book_list)]

    # Function for filtering topics
    def filter_topics(self, topic_list):
        self.topic_uri_df = self.topic_uri_df[self.topic_uri_df["Topic"].isin(topic_list)]
    
    # Count unique in field
    def count_unique(self, field):
        unique_count = len(self.topic_uri_df[field].drop_duplicates())
        print(unique_count)
        return unique_count
    
    # Ms or other counts per book
    def count_per_field(self, field="uri", on_ms = True, merge_meta = False):
        book_list = self.topic_uri_df[field].drop_duplicates().to_list()
        counts_out = []
        if on_ms:
            column_head = "ms-count"
            for book in book_list:                
                unique_ms_count = len(self.topic_uri_df[self.topic_uri_df[field] == book]["ms"].drop_duplicates())
                counts_out.append({field: book, column_head: unique_ms_count})
        else:
             column_head = "phrase-count"
             for book in book_list:
                count = len(self.topic_uri_df[self.topic_uri_df[field] == book])
                counts_out.append({field: book, column_head: count})
        df_out = pd.DataFrame(counts_out)
        if merge_meta:
            if field == "uri":
                df_out = self.merge_meta(df_out)
            else:
                print("Incompatible column name {} for meta merging - skipping merge...".format(field))
        
        return df_out.sort_values(by = column_head, ascending=False)
    
    # Get tags associated with the topic list and return dataframe counting the tags - use either the pri or the longest tag field for the calculation
    def get_and_count_tags(self, use_pri = True):
        if use_pri:
            return getTags(topic_meta.topic_uri_df[["uri", "tags"]].drop_duplicates(), input_df=True).count_tags(texts_list = True)
        else:
            book_metadata = self.filter_meta_on_book(self.topic_uri_df).drop_duplicates()
            all_tags = getTags(book_metadata, input_df=True, use_all_versions=True)
            return all_tags.count_tags(texts_list=True)

    # Fetch data according to list of values
    def fetch_data_by_list(self, val_list, field="date", csv_out = None):
        if field == "tags":
            data_filtered = pd.DataFrame()
            to_filter = self.topic_uri_df.dropna()
            for tag in val_list:                
                tag_match = to_filter[to_filter["tags"].str.contains(tag)][["Topic", "uri", "ms"]]
                data_filtered = pd.concat([data_filtered, tag_match])
            data_filtered = data_filtered.drop_duplicates()
        else:
            data_filtered = self.topic_uri_df[self.topic_uri_df[field].isin(val_list)][["Topic", "uri", "ms"]]
        data_filtered = data_filtered.sort_values(by=["uri", "Topic", "ms"])
        if csv_out:
            data_filtered.to_csv(csv_out, index=False)
        return data_filtered
    # Graph by death date
    def hist_on_field(self, field="date", image_path = None, ax_loc = None, hue_on_tags = False):
        if ax_loc == None:
            plt.clf()

        if self.graphing_par is not None:
            graph_df = self.topic_uri_df[[self.graphing_par, field, "tags"]].drop_duplicates()
        else:
            graph_df = self.topic_uri_df.drop_duplicates()
        
        if hue_on_tags:
            print(self.tags)
        if hue_on_tags and self.tags is not None:
            tags_df = pd.DataFrame()
            df_for_tags = graph_df.dropna()
            for tag in self.tags:
                temp_df = df_for_tags[df_for_tags["tags"].str.contains(tag)]
                temp_df["hue"] = "tags"
                tags_df = pd.concat([tags_df, temp_df])
            tags_df = tags_df.drop_duplicates()
            graph_df["hue"] = "all"
            graph_df = pd.concat([graph_df, tags_df])
            graph_df = graph_df.reset_index()
            hue = "hue"

        else:
            hue = None


        g = sns.histplot(graph_df, ax = ax_loc, x = field, hue=hue, binwidth=25, binrange=(0,1000))
        
        # Dynamically labeling based on chosen options
        if ax_loc is None:
            if field == "date":
                g.set(xlabel = 'Death date of author (AH)')
            if self.graphing_par is not None:
                ylabel = 'Frequency of {}'.format(self.graphing_par)
                
            else:
                ylabel = "Frequency of phrases"
            g.set(ylabel= ylabel)
        if image_path:
            figure = g.figure       
            figure.savefig(image_path, dpi=300, bbox_inches='tight')
        return g
    
    def hist_perc(self, field, image_path = None, ax_loc = None):
        if ax_loc == None:
            plt.clf()
        # Load in data
        if self.graphing_par == "uri":
            meta_data = self.meta_df[[field, "version_uri"]].drop_duplicates()
        elif self.graphing_par == "uri-ms":
            meta_data = self.meta_df[[field, "ms-total"]].drop_duplicates()
        else:
            meta_data = self.meta_df[[field, self.graphing_par]].drop_duplicates()
        main_data = self.topic_uri_df[[field, self.graphing_par]].drop_duplicates()

        # Walk through data, calculate percentages and create df for graph
        graph_dicts = []

        for i in range(0, 1000, 25):
            listed_values = list(range(i,i+25))
            meta = meta_data[meta_data[field].isin(listed_values)]
            main_count = len(main_data[main_data[field].isin(listed_values)])
            if self.graphing_par == "uri-ms":
                meta_count = meta["ms-total"].sum()
            else:
                meta_count = len(meta)
            graph_dicts.append({"year": i+12.5, "perc": main_count/meta_count*100})
        
        graph_df = pd.DataFrame(graph_dicts)

        g = sns.histplot(data=graph_df, x="year", ax=ax_loc, weights="perc", binwidth=25, binrange=(0,1000))

        std = graph_df["perc"].std()
        mean = graph_df["perc"].mean()
        g.axhline(y=mean-std, color='black', linestyle='--', linewidth=0.5)
        g.axhline(y=mean, color='black', linestyle='-', linewidth=0.5)
        g.axhline(y=mean+std, color='black', linestyle='--', linewidth=0.5)
        

        if image_path:
            figure = g.figure       
            figure.savefig(image_path, dpi=300, bbox_inches='tight')
        return g

    def calculate_poisson(self, image_path=None, interval = 25, on = "tokens", field="date", fetch_sig_pos = True, fetch_sig_neg = False):
        plt.clf()
        poisson_dicts = []
        meta_data = self.meta_df
        main_data = self.topic_uri_df
        for i in range(0, 1000, 25):
            print(i)
            listed_values = list(range(i,i+24))
            if on == "tokens":
                meta = meta_data[meta_data[field].isin(listed_values)]["tok_length"].sum()
                main_count = len(main_data[main_data[field].isin(listed_values)])
            if on == "uri":
                meta = len(meta_data[meta_data[field].isin(listed_values)]["version_uri"].drop_duplicates())
                main_count = len(main_data[main_data[field].isin(listed_values)]["uri"].drop_duplicates())
            poisson_dicts.append({"year": i, "freq": main_count, on : meta, "rate": main_count/meta})
        
        print("Calculating mean")
        total_obs = len(poisson_dicts)
        poisson_df = pd.DataFrame(poisson_dicts)
        mu = poisson_df["freq"].sum()/poisson_df[on].sum()
        print(mu)
        for block in poisson_dicts:
            
            # block["poisson"] = poisson_prob
            block["exp_freq"] = block[on]*mu
            poisson_prob = 1 - poisson.cdf(k=block["freq"], mu=block["exp_freq"])
            block["poisson_prob"] = poisson_prob
        
        poisson_df = pd.DataFrame(poisson_dicts)

        g = sns.histplot(poisson_df, x="year", weights="freq", binwidth=25, binrange=(0,1000))
        for item in poisson_dicts:
            year = item["year"]
            year_end = year + 24
            if item["poisson_prob"] > 0.95:
                colour = "green"
                if fetch_sig_neg:
                    val_list = list(range(year, year_end))
                    csv_out = "{}-negative-outlier-{}-{}.csv".format(on, year, year_end)
                    self.fetch_data_by_list(val_list, csv_out=csv_out)
            elif item["poisson_prob"] < 0.05:
                colour = "red"
                if fetch_sig_pos:
                    val_list = list(range(year, year_end))
                    csv_out = "{}-positive-outlier-{}-{}.csv".format(on, year, year_end)
                    self.fetch_data_by_list(val_list, csv_out=csv_out)
            else:
                colour = "black"
            g.hlines(y=item["exp_freq"], xmin=year, xmax=year_end, color=colour, linewidth=0.5)

        if field == "date":
                g.set(xlabel = 'Death date of author (AH)')
        if on == "tok_length":
            g.set(ylabel = 'Frequency of phrases')
        if on == "uri":
            g.set(ylabel = 'Frequency of uri')
        

        if image_path:
            figure = g.figure       
            figure.savefig(image_path, dpi=300, bbox_inches='tight')
        return poisson_df, g

    # Produce comparitive subplots of genre tags to a filed
    def comp_hist_on_field_tags(self, tags, field="date", image_path = None, limit_x_values = True, comp_title_word = "", hue_on_tags = False):
        
        """If 'limit_x_values' is set to true, the metadata is filtered to only contain values below the maximum value
        in the specified field. This means that the x-axes are comparable between the two graphs. For example, if the 
        dataset only contains dates before 1000 AH, setting this parameter to true would mean that the x axis in the 
        compared tags graph only contained values before 1000 AH"""

        plt.clf()
        if self.graphing_par == "uri-ms":
            weights = "ms-total"
        else:
            weights = None
        ## Check if fields present for creating averages - if fields not in metadata - do not add that to facetgrid
        # if self.graphing_par in self.meta_df.columns or self.graphing_par == "uri":
        figure, axes = plt.subplots(4,1, sharex=True)
        self.hist_perc(field, ax_loc=axes[3])
        perc_graph = True
        # else:
        #     figure, axes = plt.subplots(3,1, sharex=True)
        #     perc_graph = False

        # Ensure tags are a list and set the parameter
        if type(tags) is not list:
            tags = [tags]
        self.tags = tags
        graph_df = pd.DataFrame()

        # Plot the field histogram on one axis
        self.hist_on_field(field=field, hue_on_tags = hue_on_tags, ax_loc=axes[0])

        # Loop through tags and filter main metadata on them


        if limit_x_values:
            # Filter metadata so that it only contains up to the maximum of the field in the dataset
            max_value = self.topic_uri_df[field].max()
            meta_for_anal = self.meta_df[self.meta_df[field] < max_value]
        else:
            meta_for_anal = self.meta_df
        
        if self.graphing_par == None or self.graphing_par == "uri-ms" or self.graphing_par == "uri":
            main_col = "version_uri"            
        else:
            main_col = self.graphing_par
        # Need to drop dups according to graphing parameter - or else not comparable.
        meta_cols = meta_for_anal[[main_col, field, "ms-total", "tags"]].dropna().drop_duplicates()
        for tag in self.tags:
            filtered_meta = meta_cols[meta_cols['tags'].str.contains(tag)]
            graph_df = pd.concat([graph_df, filtered_meta])
        graph_df = graph_df.drop_duplicates()

        # Plot histogram onto other axis logging the count of death dates by those tags
        sns.histplot(graph_df, ax= axes[1], x="date", weights= weights, binwidth=25, binrange=(0,1000))
        sns.histplot(meta_cols, ax= axes[2], x="date", weights= weights, binwidth=25, binrange=(0,1000))

        # Dynamically label based on chosen fields
        if self.graphing_par is None:
            title_word = "phrases"    
        else:
            title_word = self.graphing_par
        if perc_graph:
            for i in [0,1,2,3]:
                axes[i].set(xlabel="")
            axes[3].set_ylabel("Percentage of {}\nin corpus".format(self.graphing_par), fontsize=5)
            
        else:
            for i in [0,1,2]:
                axes[i].set(xlabel="")
            
        axes[0].set_ylabel(ylabel="{}\nin topics".format(title_word), fontsize=5)
        axes[1].set_ylabel(ylabel="{}\nwith {} tags".format(self.graphing_par, comp_title_word), fontsize=5)
        axes[2].set_ylabel(ylabel="{}\nin corpus".format(self.graphing_par), fontsize=5)
        
        
        
        """Code for Setting title when comparing horizontally rather than vertically"""
        # axes[0].set_title("{}\nin topics".format(title_word))

        # axes[1].set_title("{}\nwith {} tags".format(main_col, comp_title_word))
        # axes[2].set_title("{}\nin corpus".format(main_col, comp_title_word))
        
        if field == "date":
            common_x_label = "Death date of author (AH)"
        else:
            common_x_label = field
        figure.text(0.5, 0.04, common_x_label, ha='center')


        if image_path:
            figure.savefig(image_path, dpi=300, bbox_inches='tight')
        return figure, axes

def produce_graphs(graph_png_path, topic_filter = [], comp_meta_tags = None, graphing_par = "ms-uri", topic_meta_obj = None, topic_data = None, metadata = None, comp_title_word = ""):
    if not topic_meta_obj:
        topic_meta = uriTopicMetadata(topic_data, metadata, topic_filter, graphing_par = graphing_par)
    else:
        topic_meta = topic_meta_obj
        topic_meta.graphing_par = graphing_par

    if comp_meta_tags:
        fig = topic_meta.comp_hist_on_field_tags(comp_meta_tags, image_path = graph_png_path, comp_title_word = comp_title_word, hue_on_tags = True)
    else:
        fig = topic_meta.hist_on_field(image_path = graph_png_path)
    
    return fig

if __name__ == "__main__":
    
    graphing_sets = [
        {"graph_path": "Yusuf-Hadith-comp-ms-uri+hadith-uris.png", "comp_meta_tags": ["_HADITH", "GAL@hadith"], "graphing_par": "uri-ms", "title-word": "Hadith"}, 
        {"graph_path": "Yusuf-Hadith-comp-uri+hadith-uris.png", "comp_meta_tags": ["_HADITH", "GAL@hadith"], "graphing_par": "uri", "title-word": "Hadith"},
        {"graph_path": "Yusuf-Hadith-comp-uri+hadith-authors.png", "comp_meta_tags": ["_HADITH", "GAL@hadith"], "graphing_par": "author_from_uri", "title-word": "Hadith"},
         {"graph_path": "Yusuf-Hadith-sentence-counts.png", "comp_meta_tags": None, "graphing_par": None, "title-word": ""}]
    
    data_path = "E:/topicModelling/data/outputs/searchModelling/results-camelbert-seed10-run2-outliers4.csv"

    meta_path = "E:/Corpus Stats/2022/OpenITI_metadata_2022-1-6_merged.csv"

    topic_focus = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/tasks/output/Yusuf-famine-hadith.csv"

    topic_list = pd.read_csv(topic_focus)["Topic"].tolist()
    if "Total" in topic_list:
        topic_list.remove("Total")
    topic_list = [int(t) for t in topic_list]
    print(topic_list)

    topic_meta = uriTopicMetadata(meta_path, data_path, topic_filter = topic_list)

  
    # Test poisson
    # poisson_df, g = topic_meta.calculate_poisson(image_path="tok-count-poisson.png")
    # poisson_df.to_csv("tok_count_poisson.csv")
    # poisson_df, g = topic_meta.calculate_poisson(image_path="uri-poisson.png", on = "uri")
    # poisson_df.to_csv("uri_poisson.csv")
    # topic_meta.fetch_data_by_list(val_list=["GAL@history", "GAL@historiography"], csv_out="gal-history-historiography.csv", field="tags")
    # topic_meta.fetch_data_by_list(val_list=["GAL@history-world", "GAL@history-universal"], csv_out="gal-history-world-universal.csv", field="tags")

    # for graph in graphing_sets:
    #     print(graph)
    #     produce_graphs(graph["graph_path"], topic_meta_obj = topic_meta, comp_meta_tags = graph["comp_meta_tags"], graphing_par = graph["graphing_par"], comp_title_word=graph["title-word"])

    # Produce summary sets
    topic_tags = topic_meta.get_and_count_tags(use_pri=False)
    topic_tags.to_csv("Yusuf-hadith-top-tags.csv", encoding = 'utf-8-sig')

    # topic_meta.count_per_field(field="author_from_uri").to_csv("Yusuf-hadith-ms-count-author.csv")
    # topic_meta.count_per_field(field="author_from_uri", on_ms=False).to_csv("Yusuf-hadith-count-author.csv")

    # topic_meta.count_per_field().to_csv("Yusuf-hadith-ms-count-book.csv")
    # topic_meta.count_per_field(on_ms=False).to_csv("Yusuf-hadith-count-book.csv")




        
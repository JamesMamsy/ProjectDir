import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder

class FlightDelayApriori:
    def __init__(self,data):
        self.df = None
        self.df = data
        self.load_data()

    def load_data(self):
        self.prepare_data()
        #self.pickle("*.csv", "output.pkl")
        #self.df = pd.read_pickle('output.pkl')
        #self.prepare_data()


    def prepare_data(self):
        delay_types = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
        self.df['No_Delay'] = self.df[delay_types].isnull().all(axis=1).astype(int)
        self.df['FL_DATE'] = pd.to_datetime(self.df['FL_DATE'])
        self.df['Month'] = self.df['FL_DATE'].dt.strftime('%m')

    def filter_data(self, user_month, user_carrier, user_dept_apt):
        filtered_df = self.df[(pd.to_datetime(self.df['FL_DATE']).dt.month == int(user_month)) &
                              (self.df['OP_UNIQUE_CARRIER'] == user_carrier) &
                              (self.df['ORIGIN'] == user_dept_apt)]
        return self.analyze_delays(filtered_df)

    def analyze_delays(self, filtered_df):
        delay_types = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
        bins = [-1, 30, 60, 120, 240, float('inf')]
        labels = ['0-30 Minutes', '31-60 Minutes', '61-120 Minutes', '121-240 Minutes', '240+ Minutes']
        delay_bucket_cols = [f'{dt}_Bucket' for dt in delay_types]

        for delay_type in delay_types:
            bucket_col_name = delay_type + '_Bucket'
            filtered_df[bucket_col_name] = pd.cut(
                filtered_df[delay_type].replace(0, np.nan), bins=bins, labels=labels, include_lowest=True
            )

        filtered_df['Delay_Status'] = filtered_df.apply(
            lambda row: 'No Delay' if row[delay_bucket_cols].isnull().all() else 'Delay',
            axis=1
        )
        delay_bucket_cols: list[str] = \
            ['CARRIER_DELAY_Bucket', 'WEATHER_DELAY_Bucket', 'NAS_DELAY_Bucket',
             'SECURITY_DELAY_Bucket', 'LATE_AIRCRAFT_DELAY_Bucket']

        # Replace NaN values with "No Delay" in the bucket columns
        for delay_bucket_col in delay_bucket_cols:
            # Get the current categories
            current_categories = filtered_df[delay_bucket_col].cat.categories.tolist()

            # If "No Delay" is not already a category, add it
            if "No Delay" not in current_categories:
                new_categories = current_categories + ["No Delay"]
                filtered_df[delay_bucket_col].cat.set_categories(new_categories, in_place = True)

            # Now you can fill NaN values with "No Delay"
            filtered_df[delay_bucket_col].fillna('No Delay', inplace=True)

        other_relevant_cols = ['Month', 'Delay_Status']
        all_relevant_cols = delay_bucket_cols + other_relevant_cols

        trim_df = filtered_df[all_relevant_cols]
        return trim_df

    def calculate_probabilities(self, filtered_df):
        delay_proportion = (filtered_df['Delay_Status'] == 'Delay').sum() / len(filtered_df) * 100
        delay_only_df = filtered_df[filtered_df['Delay_Status'] == 'Delay']
        transformed_data = self.transform_data(delay_only_df)
        apriori_df = self.apply_transaction_encoder(transformed_data)
        frequent_itemsets = self.apriori_from_scratch(apriori_df, min_support=0.01)
        highest_support_itemset = frequent_itemsets.iloc[0]['itemsets']
        highest_support_value = list(highest_support_itemset)[0]

        return delay_proportion, highest_support_value

    def transform_data(self, delay_only_df):
        delay_types = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
        transformed_data = []
        for _, row in delay_only_df.iterrows():
            features = set()
            for delay_type in delay_types:
                bucket_col = delay_type + '_Bucket'
                delay_value = row[bucket_col]
                if delay_value != 'No Delay':
                    features.add(f"{delay_type}: {delay_value}")
            transformed_data.append(list(features))
        return transformed_data

    def apply_transaction_encoder(self, transformed_data):
        te = TransactionEncoder()
        te_ary = te.fit(transformed_data).transform(transformed_data)
        return pd.DataFrame(te_ary, columns=te.columns_)

    def apriori_from_scratch(self, df, min_support):
        def generate_candidates(prev_frequent, k):
            candidates = set()
            for itemset1 in prev_frequent:
                for itemset2 in prev_frequent:
                    union_set = itemset1.union(itemset2)
                    if len(union_set) == k:
                        candidates.add(union_set)
            return candidates

        def calculate_support(df, candidates):
            supports = {}
            for candidate in candidates:
                candidate_list = list(candidate)
                supports[candidate] = df[candidate_list].all(axis=1).mean()
            return supports

        frequent_itemsets = []
        single_items = [frozenset([col]) for col in df.columns if df[col].mean() >= min_support]
        k = 1
        current_frequent = single_items

        while current_frequent:
            # Add current frequent itemsets to the global list
            frequent_itemsets.extend(current_frequent)

            # Generate new candidates from current frequent itemsets
            candidates = generate_candidates(current_frequent, k + 1)

            # Calculate support for new candidates
            candidate_supports = calculate_support(df, candidates)

            # Filter candidates by min support
            current_frequent = [itemset for itemset, support in candidate_supports.items() if support >= min_support]

            k += 1

        # Create DataFrame of frequent itemsets with support values
        itemset_list = []
        for itemset in frequent_itemsets:
            itemset_list.append({
                'itemsets': itemset,
                'support': df[list(itemset)].all(axis=1).mean()
            })

        return pd.DataFrame(itemset_list).sort_values(by='support', ascending=False)


def generate_candidates(prev_frequent, k):
    """Generate candidate itemsets of size k from the previous frequent itemsets."""
    candidates = set()
    for itemset1 in prev_frequent:
        for itemset2 in prev_frequent:
            union_set = itemset1.union(itemset2)
            if len(union_set) == k:
                candidates.add(union_set)
    return candidates


def calculate_support(df, candidates):
    """Calculate the support for candidate itemsets in the dataframe."""
    supports = {}
    for candidate in candidates:
        candidate_list = list(candidate)
        supports[candidate] = df[candidate_list].all(axis=1).mean()
    return supports


def apriori_from_scratch(df, min_support):
    """Perform the apriori algorithm on a pandas DataFrame of booleans."""
    frequent_itemsets = []
    single_items = [frozenset([col]) for col in df.columns if df[col].mean() >= min_support]
    k = 1
    current_frequent = single_items

    while current_frequent:
        # Add current frequent itemsets to the global list
        frequent_itemsets.extend(current_frequent)

        # Generate new candidates from current frequent itemsets
        candidates = generate_candidates(current_frequent, k + 1)

        # Calculate support for new candidates
        candidate_supports = calculate_support(df, candidates)

        # Filter candidates by min support
        current_frequent = [itemset for itemset, support in candidate_supports.items() if support >= min_support]

        k += 1

    # Create DataFrame of frequent itemsets with support values
    itemset_list = []
    for itemset in frequent_itemsets:
        itemset_list.append({
            'itemsets': itemset,
            'support': df[list(itemset)].all(axis=1).mean()
        })

    return pd.DataFrame(itemset_list).sort_values(by='support', ascending=False)

import pandas as pd
import numpy as np
from data import pickler as pk
from mlxtend.preprocessing import TransactionEncoder

pk.pickle("*.csv", "output.pkl",
          "C:/Users/eliza/OneDrive/Documents/Education/OU - Student Files/Fall 2023/Data Mining/Project Files/Data Files")

# Load the pickled DataFrame
df = pd.read_pickle('output.pkl')

# Define delay types
delay_types = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

# Add 'No_Delay' column, which is 1 if all delay columns are NaN, otherwise 0
df['No_Delay'] = df[delay_types].isnull().all(axis=1).astype(int)

# Convert 'FL_DATE' to datetime if it's not already
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# Create a new column with the month in two digits
df['Month'] = df['FL_DATE'].dt.strftime('%m')

# Get user input for month/day and carrier
user_month = input("Enter the month of the flight (MM): ")
user_carrier = input("Enter the carrier code: ")
user_dept_apt = input("Enter the 3 digit Departure Airport Code: ")

# Filter the DataFrame for the given month, carrier, and departure airport
filtered_df = df[(pd.to_datetime(df['FL_DATE']).dt.month == int(user_month)) &
                 (df['OP_UNIQUE_CARRIER'] == user_carrier) &
                 (df['ORIGIN'] == user_dept_apt)]

# Display the filtered DataFrame
print(filtered_df)

# Define the bins for delay times
bins = [-1, 30, 60, 120, 240, float('inf')]
labels = ['0-30 Minutes', '31-60 Minutes', '61-120 Minutes', '121-240 Minutes', '240+ Minutes']

# Apply the binning to each delay column
for delay_type in delay_types:
    # Create a new column for the binned data
    bucket_col_name = delay_type + '_Bucket'
    # Replace zero delays with NaN, then bin the data
    filtered_df[bucket_col_name] = pd.cut(
        filtered_df[delay_type].replace(0, np.nan), bins=bins, labels=labels, include_lowest=True
    )

# List of bucket columns
delay_bucket_cols: list[str] = \
    ['CARRIER_DELAY_Bucket', 'WEATHER_DELAY_Bucket', 'NAS_DELAY_Bucket',
     'SECURITY_DELAY_Bucket', 'LATE_AIRCRAFT_DELAY_Bucket']

# Create a new column based on the condition
filtered_df['Delay_Status'] = filtered_df.apply(
    lambda row: 'No Delay' if row[delay_bucket_cols].isnull().all() else 'Delay',
    axis=1
)

# Replace NaN values with "No Delay" in the bucket columns
for delay_bucket_col in delay_bucket_cols:
    # Get the current categories
    current_categories = filtered_df[delay_bucket_col].cat.categories.tolist()

    # If "No Delay" is not already a category, add it
    if "No Delay" not in current_categories:
        new_categories = current_categories + ["No Delay"]
        filtered_df[delay_bucket_col].cat.set_categories(new_categories, inplace=True)

    # Now you can fill NaN values with "No Delay"
    filtered_df[delay_bucket_col].fillna('No Delay', inplace=True)

print(filtered_df.head())

other_relevant_cols = ['Month', 'Delay_Status']
all_relevant_cols = delay_bucket_cols + other_relevant_cols

trim_df = filtered_df[all_relevant_cols]

print(trim_df.head())

# calculate probability of delay
# Count the number of rows with 'Delay_Status' as 'Delay'
delay_count = (filtered_df['Delay_Status'] == 'Delay').sum()

# Get the total number of rows in the DataFrame
total_rows = len(filtered_df)

# Calculate the proportion
delay_proportion = delay_count / total_rows * 100

print(f"Proportion of delays: {delay_proportion:.2f}%")

# Filter the DataFrame to include only rows with 'Delay' in 'Delay_Status'
delay_only_df = filtered_df[filtered_df['Delay_Status'] == 'Delay']

# Create a transformed dataset
transformed_data = []
for _, row in delay_only_df.iterrows():
    features = set()

    # Add concatenated delay type and bucket
    for delay_type in delay_types:
        bucket_col = delay_type + '_Bucket'
        delay_value = row[bucket_col]

        # Add delay feature only if it's not 'No Delay'
        if delay_value != 'No Delay':
            features.add(f"{delay_type}: {delay_value}")

    transformed_data.append(list(features))

# Apply TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transformed_data).transform(transformed_data)
apriori_df = pd.DataFrame(te_ary, columns=te.columns_)

# After creating apriori_df, check for missing bucket/delay type combinations

# Define all expected bucket labels
bucket_labels = ['0-30 Minutes', '31-60 Minutes', '61-120 Minutes', '121-240 Minutes', '240+ Minutes']

# Define all expected delay types
delay_types = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

# Generate all expected combinations
expected_combinations = [f"{delay_type}: {label}" for delay_type in delay_types for label in bucket_labels]

# Check if each expected combination is in apriori_df columns, if not, add it with False values
for combo in expected_combinations:
    if combo not in apriori_df.columns:
        apriori_df[combo] = False

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

frequent_itemsets = apriori_from_scratch(apriori_df, min_support=0.01)
print(frequent_itemsets)

#apriori_df.to_csv("apriori_df.csv", index=False)
#frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)

# Sort by support in descending order
frequent_itemsets_sorted = frequent_itemsets.sort_values('support', ascending=False)

# Get the highest support itemset
highest_support_itemset = frequent_itemsets_sorted.iloc[0]

# Parse out the value(s) from the itemset
highest_support_value = list(highest_support_itemset['itemsets'])[0]

# Print the highest support value and its support
print(f"Probability of Delay: {delay_proportion:.2f}%")
print("Most Likely Delay:", highest_support_value)
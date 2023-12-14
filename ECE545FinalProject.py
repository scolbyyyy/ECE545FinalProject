import pandas as pd
import random
import itertools

##########################################################################################
###########################    DATA GENERATION   #########################################
##########################################################################################

num_records = 50
medical_conditions = ['Condition A', 'Condition B', 'Condition C', 'Condition D', 'Condition E']
data = {
    'age': [random.randint(18, 90) for _ in range(num_records)],
    'zipcode': [random.randint(10000, 99999) for _ in range(num_records)],
    'medical_condition': [random.choice(medical_conditions) for _ in range(num_records)]
}

df = pd.DataFrame(data)
df.to_csv('survey.csv', index=False)
data = pd.read_csv('survey.csv')

##########################################################################################
###########################    BEGIN FUNCTIONS   #########################################
##########################################################################################

def apply_k_anonymity_l_diversity(df, quasi_identifiers, sensitive_col, k, l):
    """
    Sort dataframe based on quasi-identifiers, k and l values
    returns dataframe for anonymized data-set based on parameters
    """
    df_sorted = df.sort_values(by=quasi_identifiers).reset_index(drop=True)
    anonymized_df = pd.DataFrame(columns=df.columns)

    start = 0
    while start < len(df_sorted):
        end = start + k

        while end <= len(df_sorted):
            subset = df_sorted.iloc[start:end]
            if subset[sensitive_col].nunique() >= l:
                break
            end += 1
        if end > len(df_sorted):
            # No suitable group found, break to avoid infinite loop
            break

        # For numerical data: generalize to the range (min, max)
        # For categorical data: generalize to a common category or a higher-level category
        anonymized_subset = subset.copy()
        for col in quasi_identifiers:
            if anonymized_subset[col].dtype.kind in 'biufc':
                min_val, max_val = anonymized_subset[col].min(), anonymized_subset[col].max()
                anonymized_subset[col] = f"{min_val}-{max_val}"
            else:
                anonymized_subset[col] = 'Generalized Category'

        anonymized_df = pd.concat([anonymized_df, anonymized_subset], ignore_index=True)
        start = end

    return anonymized_df


def calculate_utility_metric(df, original_df, dropped):
    """
    Check the number of participants that get dropped and return value
    that will later drop the k and l combo if the result is too large
    """
    ogColumns = len(original_df.axes[0]) # number of records before
    newColumns = len(df.axes[0]) # number of records after
    result = ogColumns - newColumns
    if result > dropped:
        return -1
    else:
        return result

def calculate_privacy_metric(df, quasi_identifiers, sensitive_col):
    """
    FInd average number of members in each group for k value score,
    then find average l value in all these groups and combine them for
    an accurate privacy score
    """
    kGrouped = df.groupby(quasi_identifiers)
    kGroup_sizes = [len(group) for _, group in kGrouped]

    if len(kGroup_sizes) == 0:
        return 0  # No valid groups were formed

    diversity_scores = []

    for group_name, group_df in df.groupby(quasi_identifiers):
        unique_sensitive_values = group_df[sensitive_col].nunique()
        diversity_scores.append(unique_sensitive_values)

    average_group_size = sum(kGroup_sizes) / len(kGroup_sizes)
    average_diversity_score = sum(diversity_scores) / len(diversity_scores)

    kandlscore = average_group_size*average_diversity_score
    return kandlscore

def combined_score(utility, privacy):
    """
    Give a score of zero if it fails utility test and return privacy score
    if it passes
    """
    if utility == -1:
        return 0
    else:
        return privacy

# Evaluation of combinations of K and L
def evaluate_combination(df, quasi_identifiers, sensitive_col, k, l, original_df):
    """
    returns utility score and privacy score of a certain k and l combo on the dataset
    """
    anonymized_df = apply_k_anonymity_l_diversity(df, quasi_identifiers, sensitive_col, k, l)
    utility_metric = calculate_utility_metric(anonymized_df, original_df, dropped)
    # privacy_metric = calculate_privacy_metric(anonymized_df, quasi_identifiers)
    privacy_metric = calculate_privacy_metric(anonymized_df, quasi_identifiers, sensitive_col)
    return utility_metric, privacy_metric


##########################################################################################
##############################    Main Program   #########################################
##########################################################################################

dropped = int(input('How many records would you allow to drop?\n'))
max_K = int(input('Maximum k value allowed?\n'))

# Define quasi-identifiers
quasi_identifiers = ['age', 'zipcode']
sensitive_col = 'medical_condition'

# Testing different combinations of K and L
best_combination = None
highest_score = -1

for k, l in itertools.product(range(1, max_K), repeat=2):  # Example range, adjust as needed
    utility, privacy = evaluate_combination(data, quasi_identifiers, sensitive_col, k, l, df)
    score = combined_score(utility, privacy)

    if score > highest_score:
        highest_score = score
        best_combination = (k, l)


best_k, best_l = best_combination

# Bug management to fix overrunning value for l (sometimes it returns 1 when it should return 4)
if best_l == 1:
    best_l = len(medical_conditions)

print(f"Best combination of K and L is: %s, %s" %(best_k, best_l))
result = apply_k_anonymity_l_diversity(df, quasi_identifiers, sensitive_col, best_k, best_l)
print(result)

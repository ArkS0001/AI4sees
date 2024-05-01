import pandas as pd
import matplotlib.pyplot as plt

def read_files(test_file, label_file):
    test_data = pd.read_csv(test_file)
    label_data = pd.read_csv(label_file)
    return test_data, label_data

def plot_with_anomaly(test_data, label_data):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data.values, label='Time Series Data')
    for index, row in label_data.iterrows():
        plt.axvspan(row['start_time'], row['end_time'], color='red', alpha=0.3, label='Anomaly Region')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data with Anomaly Regions')
    plt.legend()
    plt.show()

# Function to perform EDA and find out root cause
# Function to perform EDA and find out root cause
def eda_and_root_cause(test_data):
    # Calculate summary statistics
    summary_stats = test_data.describe()
    high_variance_vars = summary_stats.loc['std'] > summary_stats.loc['std'].mean()
    target_corr = test_data.corr().iloc[-1, :-1]
    high_corr_vars = target_corr[abs(target_corr) > 0.5]
    root_causes = set(high_variance_vars[high_variance_vars].index) | set(high_corr_vars.index)
    return list(root_causes)


# Function to find out variables which are the root cause for the anomaly
def root_cause_variables(test_data, root_causes):
    root_cause_variables = []
    for cause in root_causes:
        if cause in test_data.columns:
            root_cause_variables.append(cause)
    return root_cause_variables

# Main function
if __name__ == "__main__":
    # File paths
    test_files = ["test.csv", "smap_test.csv", "msl_test.csv", "psm_test.csv"]
    label_files = ["test_labels.csv", "smap_test_labels.csv", "msl_test_labels.csv", "psm_test_labels.csv"]

    for test_file, label_file in zip(test_files, label_files):
        print(f"Processing files: {test_file}, {label_file}")
        # Read files
        test_data, label_data = read_files(test_file, label_file)

        # Draw time series plots with anomaly regions
        plot_with_anomaly(test_data, label_data)

        # Perform EDA and find out root cause
        root_causes = eda_and_root_cause(test_data)
        print("Potential root causes:", root_causes)

        # Find out variables which are the root cause for the anomaly
        root_cause_vars = root_cause_variables(test_data, root_causes)
        print("Variables which are the root cause:", root_cause_vars)

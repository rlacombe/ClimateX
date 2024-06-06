from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get the overall accuracy for the results
def get_overall_accuracy(results_df, model_classification_col_name):
    true_labels = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk'), 'confidence']
    predicted_labels = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk'), model_classification_col_name]
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# Get the bias for the results (distance of predicted mean from ground truth mean)
def get_overall_bias(results_df, model_classification_col_name):
    predictions = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk')] 
    label_scores = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk'), "score"] 

    # Compute a certainty score
    value_map = {'low': 0, 'medium': 1, 'high': 2, 'very high': 3}

    predictions['predicted_score'] = predictions[model_classification_col_name].apply(
        lambda x: value_map[x])

    bias = predictions.predicted_score.mean() - label_scores.mean()
    return bias


# Get the slope for the regression.
# You can optionally plot the regression with the plot flag.
# Optionally print average scores (predicted, ground truth, predicted per category) with verbose flag
def get_slope(results_df, model_classification_col_name, plot=False, verbose=False):
    # Filter samples
    valid_results = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk')] 

    # Compute a certainty score
    value_map = {'low': 0, 'medium': 1, 'high': 2, 'very high': 3}

    valid_results['predicted_score'] = valid_results[model_classification_col_name].apply(
        lambda x: value_map[x])

    # Break down scores
    scores_all = {
        "low": valid_results.loc[valid_results['score'] == 0, 'predicted_score'].mean(),
        "medium": valid_results.loc[valid_results['score'] == 1, 'predicted_score'].mean(),
        "high": valid_results.loc[valid_results['score'] == 2, 'predicted_score'].mean(),
        "very high": valid_results.loc[valid_results['score'] == 3, 'predicted_score'].mean()
    }

    if verbose:
        # print("=== All AR6 reports===")
        print(f"Average ground truth score: {valid_results['score'].mean()}")
        print(f"Average predicted score: {valid_results['predicted_score'].mean()}")
        print(f"Average scores per category: {scores_all}\n")

    # Extract labels and values from the data dictionary
    labels = list(scores_all.keys())
    values_all = list(scores_all.values())

    # Define the custom labels for the x-axis
    x_labels = ['0 (Low)', '1 (Medium)', '2 (High)', '3 (Very high)']

    # Create an instance of the LinearRegression model
    model = LinearRegression()

    # Fit the model using the X and Y columns of your dataframe
    model.fit(valid_results[['score']], valid_results['predicted_score'])

    # Extract the slope (coefficient) of the regression line
    slope = model.coef_[0]

    # print("The slope of the regression line is:", slope)

    if plot:
        # Plotting the data points and the regression line
        plt.figure(figsize=(10, 6))

        # Scatter plot of the data points
        # sns.scatterplot(x='score', y='predicted_score', data=valid_results, label='Data Points')

        # Plot the regression line
        sns.lineplot(x=valid_results['score'], y=model.predict(valid_results[['score']]), color='red', label='Regression Line')
        sns.lineplot(x=labels, y=values_all, color="black", label="average prediction")

        plt.title('Predicted certainty level per class')
        plt.xlabel('Label')
        plt.ylabel('Prediction')
        plt.xticks(labels, x_labels)
        plt.ylim(0, 3)
        plt.legend()
        plt.show()
    return slope


# Get the accuracy, slope, and bias
def print_accuracy_slope_bias_metrics(results_df, model_classification_col_names, plot=False, verbose=False):
    accuracy = [get_overall_accuracy(results_df, col_name) for col_name in model_classification_col_names]
    bias = [get_overall_bias(results_df, col_name) for col_name in model_classification_col_names]
    print("accuracies", accuracy)
    print("biases:", bias)
    slope = [get_slope(results_df, col_name, plot=plot, verbose=verbose) for col_name in model_classification_col_names]
    print("slopes", slope)

    mean_accuracy = 100*np.mean(accuracy)
    mean_slope = np.mean(slope)
    mean_bias = np.mean(bias)

    std_accuracy = 100*np.std(accuracy)
    std_slope = np.std(slope)
    std_bias = np.std(bias)

    print(f"""
---------------------------------------------------
Metric, Std Dev:
Accuracy: {mean_accuracy:.1f} ±{std_accuracy:.1f}
Slope: {mean_slope:.3f} ±{std_slope:.3f}
Bias: {mean_bias:.3f} ±{std_bias:.3f}
"""
    )


def print_metrics(results_df, model_classification_col_name):
    true_labels = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk'), 'confidence']
    predicted_labels = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk'), model_classification_col_name]

    # Compute macro F1 score
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    print("Macro F1 score:", f1)

    # Compute weighted F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print("Weighted F1 score:", f1)

    # Compute precision for each class
    precision = precision_score(true_labels, predicted_labels, average=None)

    # Compute recall for each class
    recall = recall_score(true_labels, predicted_labels, average=None)

    # Compute F1 score for each class
    f1 = f1_score(true_labels, predicted_labels, average=None)

    # Create a dataframe to store precision and recall for each class
    class_metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1': f1})

    # Add labels to the class metrics dataframe
    class_metrics_df['Class'] = true_labels.unique().astype(str)

    # Sort the dataframe by class index or name
    class_metrics_df = class_metrics_df.sort_values('Class', key=lambda x: pd.Categorical(x, categories=["low", "medium", "high", "very high"]))

    # Print class metrics dataframe
    #print(class_metrics_df)

    # Compute accuracy for the whole system
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Compute accuracy by class

    print("Accuracy (total):", accuracy)

    report = classification_report(true_labels, predicted_labels, digits=4)
    print(report)

    # Count classes
    category_counts = true_labels.value_counts()
    print(category_counts)

    return accuracy


def plot_confidence_assessment(results_df, model_classification_col_name):
    # Filter samples
    valid_results = results_df.loc[(results_df[model_classification_col_name] != 'N/A') & (results_df[model_classification_col_name] != 'idk')] 

    # Compute a certainty score
    value_map = {'low': 0, 'medium': 1, 'high': 2, 'very high': 3}

    valid_results['predicted_score'] = valid_results[model_classification_col_name].apply(
        lambda x: value_map[x])

    # Break down scores
    scores_all = {
        "low": valid_results.loc[valid_results['score'] == 0, 'predicted_score'].mean(),
        "medium": valid_results.loc[valid_results['score'] == 1, 'predicted_score'].mean(),
        "high": valid_results.loc[valid_results['score'] == 2, 'predicted_score'].mean(),
        "very high": valid_results.loc[valid_results['score'] == 3, 'predicted_score'].mean()
    }

    scores_wg1 = {
        "low": valid_results.loc[(valid_results['score'] == 0) & (valid_results['report'] == 'AR6_WGI'), 'predicted_score'].mean(),
        "medium": valid_results.loc[(valid_results['score'] == 1) & (valid_results['report'] == 'AR6_WGI'), 'predicted_score'].mean(),
        "high": valid_results.loc[(valid_results['score'] == 2) & (valid_results['report'] == 'AR6_WGI'), 'predicted_score'].mean(),
        "very high": valid_results.loc[(valid_results['score'] == 3) & (valid_results['report'] == 'AR6_WGI'), 'predicted_score'].mean()
    }

    scores_wg23 = {
        "low": valid_results.loc[(valid_results['score'] == 0) & (valid_results['report'] != 'AR6_WGI'), 'predicted_score'].mean(),
        "medium": valid_results.loc[(valid_results['score'] == 1) & (valid_results['report'] != 'AR6_WGI'), 'predicted_score'].mean(),
        "high": valid_results.loc[(valid_results['score'] == 2) & (valid_results['report'] != 'AR6_WGI'), 'predicted_score'].mean(),
        "very high": valid_results.loc[(valid_results['score'] == 3) & (valid_results['report'] != 'AR6_WGI'), 'predicted_score'].mean()
    }

    print("=== All AR6 reports===")
    print(f"Average ground truth score: {results_df['score'].mean()}")
    print(f"Average predicted score: {valid_results['predicted_score'].mean()}")
    print(f"Average scores per category: {scores_all}\n")

    print("=== AR6 WGI report ===")
    print(f"Average ground truth score: {results_df.loc[results_df['report'] == 'AR6_WGI', 'score'].mean()}")
    print(f"Average predicted score: {valid_results.loc[valid_results['report'] == 'AR6_WGI', 'predicted_score'].mean()}")
    print(f"Average scores per category: {scores_wg1}\n")

    print("=== AR6 WGII/III reports ===")
    print(f"Average ground truth score: {results_df.loc[results_df['report'] != 'AR6_WGI', 'score'].mean()}")
    print(f"Average predicted score: {valid_results.loc[valid_results['report'] != 'AR6_WGI', 'predicted_score'].mean()}")
    print(f"Average scores per category: {scores_wg23}\n")

    # Extract labels and values from the data dictionary
    labels = list(scores_all.keys())
    values_all = list(scores_all.values())
    values_wg1 = list(scores_wg1.values())
    values_wg23 = list(scores_wg23.values())

    # Define the custom labels for the x-axis
    x_labels = ['0 (Low)', '1 (Medium)', '2 (High)', '3 (Very high)']

    # Create the line plot with labeled curve
    sns.lineplot(x=labels, y=values_all, label='Average prediction')
    sns.lineplot(x=labels, y=values_wg1, linestyle='--', color="steelblue", label='WG1 report')
    sns.lineplot(x=labels, y=values_wg23, linestyle='dotted', color="steelblue", label='WG2 & WG3 report')


    # Add the ground truth line (y = x)
    x = np.arange(len(labels))
    plt.plot(x, x, linestyle='--', color='red', label='Ground truth')

    # Customize the x-axis tick labels
    plt.xticks(labels, x_labels)

    # Set the y-axis limits
    plt.ylim(0, 3)

    # Set the title and labels
    plt.title("Average predicted certainty level per class")
    plt.xlabel("Label")
    plt.ylabel("Value")

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()

    return 
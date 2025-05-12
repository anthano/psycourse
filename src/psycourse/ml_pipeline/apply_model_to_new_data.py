import pandas as pd


def apply_model_to_new_data(model, new_data):
    """
    Apply a trained model to new psycourse data with no labels and probability scores.

    Args:
    - model: The trained svm model.
    - new_data: The new psycourse data that has not been labelled.

    Returns:
    - predictions: The labels made by the model on the new data.
    """
    # Apply the model to the new data
    labels = model.predict(new_data)

    # Get the predicted probabilities
    probabilities = model.predict_proba(new_data)

    # Class labels from the model
    class_labels = model.classes_

    # Create DataFrame explicitly preserving the original index
    prob_df = pd.DataFrame(
        probabilities,
        columns=[f"prob_class_{cls}" for cls in class_labels],
        index=new_data.index,
    )

    # Add the predicted labels to the DataFrame
    prob_df["predicted_label"] = labels
    print(prob_df)

    return prob_df

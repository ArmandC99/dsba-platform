import pandas as pd
from typing import List
from .model_registry import load_model
from sklearn.metrics import accuracy_score, f1_score 

def compare_models_simple(model_ids: List[str],
                           X_eval: pd.DataFrame,
                           y_eval: pd.Series,
                           metrics: List[str]) -> pd.DataFrame:
    
    results = []
    
    supported_metrics = ['accuracy', 'f1']
    valid_metrics = [m.lower() for m in metrics if m.lower() in supported_metrics]

    print(f"Comparing models: {model_ids}")
    print(f"Using metrics: {valid_metrics}")

    for model_id in model_ids:
        scores = {'model_id': model_id}
        print(f"Evaluating model: {model_id}")
        try:
            # Load the model from the registry
            model = load_model(model_id)

            # Make predictions on the evaluation set
            y_pred = model.predict(X_eval)

            # Calculate requested metrics
            for metric_name in valid_metrics:
                score = float('nan') # Default score if calculation fails
                if metric_name == 'accuracy':
                    score = accuracy_score(y_eval, y_pred)
                elif metric_name == 'f1':
                    # Assuming binary classification for F1, like in compare_models_minimal
                    score = f1_score(y_eval, y_pred, average='binary', zero_division=0)
                # Add other metrics here if needed
                # elif metric_name == 'mae':
                #     score = mean_absolute_error(y_eval, y_pred)
                # elif metric_name == 'r2':
                #     score = r2_score(y_eval, y_pred)

                scores[metric_name] = score
            print(f"  Scores: {scores}")
            results.append(scores)

        except Exception as e:
            print(f"  Failed to evaluate model {model_id}: {e}")
            # Optionally add placeholder scores or skip the model
            for metric_name in valid_metrics:
                 if metric_name not in scores: # Ensure metrics columns exist even on failure
                     scores[metric_name] = float('nan')
            results.append(scores) # Append scores even if partial or NaN on error

    # Convert results to DataFrame
    comparison_df = pd.DataFrame(results)

    # Ensure columns exist even if no models succeeded or no metrics were valid
    final_cols = ['model_id'] + valid_metrics
    if comparison_df.empty:
         comparison_df = pd.DataFrame(columns=final_cols)
    else:
         # Reorder columns and ensure all valid metrics columns are present
         for metric in valid_metrics:
              if metric not in comparison_df.columns:
                   comparison_df[metric] = float('nan')
         comparison_df = comparison_df[final_cols]


    print("Comparison finished.")
    return comparison_df
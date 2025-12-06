"""
Ensemble predictions from 4 streams for Fall Detection
Based on BlockGCN ensemble methodology
"""

import os
import numpy as np
import pickle
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_scores(work_dir, run_folder):
    """Load prediction scores from a trained model"""
    score_path = os.path.join(work_dir, run_folder, 'score.pkl')
    
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Score file not found: {score_path}")
    
    with open(score_path, 'rb') as f:
        score_dict = pickle.load(f)
    
    return score_dict


def ensemble_predictions(scores_list, weights=None):
    """
    Ensemble predictions from multiple streams
    
    Args:
        scores_list: List of score arrays from different streams
        weights: Weights for each stream (default: equal weights)
    
    Returns:
        Ensemble predictions and scores
    """
    n_streams = len(scores_list)
    
    if weights is None:
        weights = [1.0 / n_streams] * n_streams
    
    # Weighted average of scores
    ensemble_scores = sum(w * scores for w, scores in zip(weights, scores_list))
    
    # Final predictions
    predictions = np.argmax(ensemble_scores, axis=1)
    
    return predictions, ensemble_scores


def evaluate(predictions, labels):
    """Calculate evaluation metrics"""
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Ensemble 4-stream Fall Detection')
    parser.add_argument('--work-dir', default='./work_dir/fall_detection', 
                        help='Work directory containing all stream results')
    parser.add_argument('--joint-run', required=True, help='Joint stream run folder')
    parser.add_argument('--bone-run', required=True, help='Bone stream run folder')
    parser.add_argument('--vel-run', required=True, help='Velocity stream run folder')
    parser.add_argument('--bone-vel-run', required=True, help='Bone-Velocity stream run folder')
    parser.add_argument('--weights', default=None, help='Ensemble weights (comma-separated), e.g., "0.25,0.25,0.25,0.25"')
    
    args = parser.parse_args()
    
    print("="*60)
    print("4-Stream Ensemble for Fall Detection")
    print("="*60)
    print("")
    
    # Parse weights
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        assert len(weights) == 4, "Must provide 4 weights"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    else:
        weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
    
    print(f"Ensemble weights: Joint={weights[0]}, Bone={weights[1]}, Vel={weights[2]}, Bone-Vel={weights[3]}")
    print("")
    
    # Load scores from each stream
    print("Loading predictions from 4 streams...")
    
    try:
        joint_scores = load_scores(os.path.join(args.work_dir, 'joint'), args.joint_run)
        bone_scores = load_scores(os.path.join(args.work_dir, 'bone'), args.bone_run)
        vel_scores = load_scores(os.path.join(args.work_dir, 'velocity'), args.vel_run)
        bone_vel_scores = load_scores(os.path.join(args.work_dir, 'bone_velocity'), args.bone_vel_run)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure all 4 streams have been trained and evaluated!")
        return
    
    # Get test predictions and labels
    # Assuming score_dict format: [predictions, labels, scores]
    joint_pred = joint_scores[0]
    bone_pred = bone_scores[0]
    vel_pred = vel_scores[0]
    bone_vel_pred = bone_vel_scores[0]
    
    joint_score = joint_scores[2]
    bone_score = bone_scores[2]
    vel_score = vel_scores[2]
    bone_vel_score = bone_vel_scores[2]
    
    labels = joint_scores[1]  # Ground truth labels
    
    print(f"✅ Joint stream loaded: {joint_pred.shape[0]} samples")
    print(f"✅ Bone stream loaded: {bone_pred.shape[0]} samples")
    print(f"✅ Velocity stream loaded: {vel_pred.shape[0]} samples")
    print(f"✅ Bone-Velocity stream loaded: {bone_vel_pred.shape[0]} samples")
    print("")
    
    # Individual stream performance
    print("-" * 60)
    print("Individual Stream Performance:")
    print("-" * 60)
    
    joint_metrics = evaluate(joint_pred, labels)
    bone_metrics = evaluate(bone_pred, labels)
    vel_metrics = evaluate(vel_pred, labels)
    bone_vel_metrics = evaluate(bone_vel_pred, labels)
    
    print(f"Joint:         Acc={joint_metrics['accuracy']*100:.2f}%, F1={joint_metrics['f1']*100:.2f}%")
    print(f"Bone:          Acc={bone_metrics['accuracy']*100:.2f}%, F1={bone_metrics['f1']*100:.2f}%")
    print(f"Velocity:      Acc={vel_metrics['accuracy']*100:.2f}%, F1={vel_metrics['f1']*100:.2f}%")
    print(f"Bone-Velocity: Acc={bone_vel_metrics['accuracy']*100:.2f}%, F1={bone_vel_metrics['f1']*100:.2f}%")
    print("")
    
    # Ensemble
    print("-" * 60)
    print("Ensemble Performance:")
    print("-" * 60)
    
    scores_list = [joint_score, bone_score, vel_score, bone_vel_score]
    ensemble_pred, ensemble_scores = ensemble_predictions(scores_list, weights)
    
    ensemble_metrics = evaluate(ensemble_pred, labels)
    
    print(f"Ensemble:      Acc={ensemble_metrics['accuracy']*100:.2f}%, F1={ensemble_metrics['f1']*100:.2f}%")
    print(f"               Precision={ensemble_metrics['precision']*100:.2f}%, Recall={ensemble_metrics['recall']*100:.2f}%")
    print("")
    
    # Confusion Matrix
    print("Confusion Matrix (Ensemble):")
    print(ensemble_metrics['confusion_matrix'])
    print("")
    
    # Improvement
    best_single = max(joint_metrics['accuracy'], bone_metrics['accuracy'], 
                     vel_metrics['accuracy'], bone_vel_metrics['accuracy'])
    improvement = (ensemble_metrics['accuracy'] - best_single) * 100
    
    print("="*60)
    print(f"✅ Ensemble Improvement: +{improvement:.2f}% over best single stream")
    print("="*60)
    

if __name__ == '__main__':
    main()

"""
Fast experimentation with cached combined classifier scores.
Load pre-computed scores and quickly test different thresholds and sequential filtering.
"""
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


def load_cache(cache_file):
    """Load cached inference scores."""
    print(f"Loading cached scores from: {cache_file}")
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    print(f"✓ Loaded cache")
    print(f"  Task: {cache_data['metadata']['task']}")
    print(f"  Model: {cache_data['metadata']['model_type']}")
    print(f"  Samples: {cache_data['metadata']['total_samples']}")
    print(f"  Timesteps: {cache_data['metadata']['timesteps']}")
    return cache_data


def apply_threshold(cache_data, success_threshold=0.5, aggregation='any', selected_cameras=None):
    """
    Apply threshold to cached scores.
    
    Args:
        cache_data: Loaded cache from cache_combined_scores.py
        success_threshold: Threshold for success prediction (0-1)
        aggregation: 'any' (if any timestep filters, filter it) or 'majority' or 'all'
        selected_cameras: List of cameras to use (e.g., ['corner', 'corner2']), None = all cameras
    
    Returns:
        Dictionary with results per timestep and aggregate
    """
    timesteps = cache_data['metadata']['timesteps']
    episode_mapping = cache_data['episode_mapping']
    all_cameras = cache_data['metadata']['cameras']
    
    # Use selected cameras or all cameras
    cameras_to_use = selected_cameras if selected_cameras is not None else all_cameras
    
    # Filter episode mapping to only selected cameras
    if selected_cameras is not None:
        valid_indices = [i for i, ep in enumerate(episode_mapping) if ep['camera'] in selected_cameras]
    else:
        valid_indices = list(range(len(episode_mapping)))
    
    results = {
        'config': {
            'success_threshold': success_threshold,
            'aggregation': aggregation,
            'selected_cameras': cameras_to_use
        },
        'per_timestep': {},
        'aggregate': None
    }
    
    # Process each timestep
    all_decisions = []  # Store decisions for each timestep
    
    for timestep in timesteps:
        scores = cache_data['timestep_scores'][str(timestep)]
        
        # Filter to only selected cameras
        probs = np.array([scores['probs'][i] for i in valid_indices])
        labels = np.array([scores['labels'][i] for i in valid_indices])
        
        # Apply threshold
        decisions = (probs >= success_threshold).astype(int)
        all_decisions.append(decisions)
        
        # Compute metrics
        accuracy = accuracy_score(labels, decisions)
        f1 = f1_score(labels, decisions, zero_division=0)
        precision = precision_score(labels, decisions, zero_division=0)
        recall = recall_score(labels, decisions, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(labels, probs)
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(labels, decisions)
        tn, fp, fn, tp = cm.ravel()
        
        results['per_timestep'][timestep] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
    
    # Aggregate across timesteps
    all_decisions = np.array(all_decisions)  # Shape: (n_timesteps, n_samples)
    
    if aggregation == 'any':
        # If any timestep says failure (0), mark as failure
        aggregate_decisions = np.min(all_decisions, axis=0)
    elif aggregation == 'majority':
        # Majority vote
        aggregate_decisions = (np.mean(all_decisions, axis=0) >= 0.5).astype(int)
    elif aggregation == 'all':
        # Only filter if all timesteps say failure
        aggregate_decisions = np.max(all_decisions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Get labels (same across all timesteps)
    first_timestep_str = str(timesteps[0])
    labels = np.array([cache_data['timestep_scores'][first_timestep_str]['labels'][i] for i in valid_indices])
    
    # Compute aggregate metrics
    agg_accuracy = accuracy_score(labels, aggregate_decisions)
    agg_f1 = f1_score(labels, aggregate_decisions, zero_division=0)
    agg_precision = precision_score(labels, aggregate_decisions, zero_division=0)
    agg_recall = recall_score(labels, aggregate_decisions, zero_division=0)
    
    agg_cm = confusion_matrix(labels, aggregate_decisions)
    agg_tn, agg_fp, agg_fn, agg_tp = agg_cm.ravel()
    
    results['aggregate'] = {
        'accuracy': float(agg_accuracy),
        'precision': float(agg_precision),
        'recall': float(agg_recall),
        'f1_score': float(agg_f1),
        'confusion_matrix': {'tn': int(agg_tn), 'fp': int(agg_fp), 'fn': int(agg_fn), 'tp': int(agg_tp)}
    }
    
    return results


def print_results(results, cache_data):
    """Pretty print results."""
    config = results['config']
    agg = results['aggregate']
    cm = agg['confusion_matrix']
    
    # Calculate n_samples from actual results (in case cameras were filtered)
    n_samples = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
    total_actual_failures = cm['tn'] + cm['fp']
    total_actual_successes = cm['fn'] + cm['tp']
    
    print("\n" + "="*80)
    print("COMBINED CLASSIFIER RESULTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Success threshold:  {config['success_threshold']:.2f}")
    print(f"  Aggregation:        {config['aggregation']}")
    print(f"  Selected cameras:   {config.get('selected_cameras', 'all')}")
    
    print(f"\n{'AGGREGATE PERFORMANCE':^80}")
    print("-"*80)
    print(f"Accuracy:  {agg['accuracy']:.4f}")
    print(f"Precision: {agg['precision']:.4f}")
    print(f"Recall:    {agg['recall']:.4f}")
    print(f"F1 Score:  {agg['f1_score']:.4f}")
    
    print(f"\n{'CONFUSION MATRIX':^80}")
    print("-"*80)
    print(f"True Negatives (TN):  {cm['tn']:4} (failures correctly caught)")
    print(f"False Positives (FP): {cm['fp']:4} (failures wrongly passed)")
    print(f"False Negatives (FN): {cm['fn']:4} (successes wrongly filtered)")
    print(f"True Positives (TP):  {cm['tp']:4} (successes correctly kept)")
    
    print(f"\n{'FILTERING PERFORMANCE':^80}")
    print("-"*80)
    print(f"Actual failures:      {total_actual_failures}")
    print(f"  Caught:             {cm['tn']} ({100*cm['tn']/total_actual_failures if total_actual_failures > 0 else 0:.1f}%)")
    print(f"  Missed:             {cm['fp']} ({100*cm['fp']/total_actual_failures if total_actual_failures > 0 else 0:.1f}%)")
    
    print(f"\nActual successes:     {total_actual_successes}")
    print(f"  Preserved:          {cm['tp']} ({100*cm['tp']/total_actual_successes if total_actual_successes > 0 else 0:.1f}%)")
    print(f"  Lost:               {cm['fn']} ({100*cm['fn']/total_actual_successes if total_actual_successes > 0 else 0:.1f}%)")
    
    kept_total = cm['tp'] + cm['fp']
    if kept_total > 0:
        kept_success_rate = 100 * cm['tp'] / kept_total
        original_success_rate = 100 * total_actual_successes / n_samples
        print(f"\n{'QUALITY IMPROVEMENT':^80}")
        print("-"*80)
        print(f"Original success rate: {original_success_rate:.1f}%")
        print(f"Kept samples success:  {kept_success_rate:.1f}%")
        print(f"Improvement:           {kept_success_rate - original_success_rate:+.1f}%")
    
    print("="*80)


def threshold_grid_search(cache_data, thresholds, aggregation='any'):
    """
    Run grid search over thresholds.
    """
    print("\n" + "="*80)
    print("THRESHOLD GRID SEARCH")
    print("="*80)
    print(f"Testing {len(thresholds)} thresholds")
    
    best_config = None
    best_f1 = -1
    best_recall = -1
    all_results = []
    
    for threshold in thresholds:
        results = apply_threshold(
            cache_data,
            success_threshold=threshold,
            aggregation=aggregation
        )
        
        f1 = results['aggregate']['f1_score']
        recall = results['aggregate']['recall']
        
        all_results.append({
            'config': results['config'],
            'f1': f1,
            'recall': recall,
            'accuracy': results['aggregate']['accuracy'],
            'precision': results['aggregate']['precision']
        })
        
        # Track best by F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_config = results
        
        # Track best by recall
        if recall > best_recall:
            best_recall = recall
    
    print(f"\nBest F1 Score: {best_f1:.4f}")
    print(f"  Threshold: {best_config['config']['success_threshold']:.2f}")
    
    print(f"\nBest Recall: {best_recall:.4f}")
    
    # Print all configurations
    print(f"\n{'ALL CONFIGURATIONS BY F1':^80}")
    print("-"*80)
    print(f"{'Rank':<6} {'Threshold':<12} {'F1':<10} {'Recall':<10} {'Precision':<10} {'Accuracy':<10}")
    print("-"*80)
    
    sorted_results = sorted(all_results, key=lambda x: x['f1'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        cfg = r['config']
        print(f"{i:<6} {cfg['success_threshold']:<12.2f} {r['f1']:<10.4f} {r['recall']:<10.4f} "
              f"{r['precision']:<10.4f} {r['accuracy']:<10.4f}")
    
    print("="*80)
    
    return best_config, all_results


def simulate_sequential_filtering(cache_data, success_threshold=0.5, selected_cameras=None):
    """
    Simulate sequential filtering as if generating from high timestep to low.
    Start at highest timestep (e.g., 85) and progressively filter down to lowest (e.g., 15).
    
    Args:
        cache_data: Loaded cache from cache_combined_scores.py
        success_threshold: Threshold for success prediction (0-1)
        selected_cameras: List of cameras to use (e.g., ['corner', 'corner2']), None = all cameras
    """
    timesteps = sorted(cache_data['metadata']['timesteps'], reverse=True)  # High to low
    all_cameras = cache_data['metadata']['cameras']
    episode_mapping = cache_data['episode_mapping']
    
    # Use selected cameras or all cameras
    cameras = selected_cameras if selected_cameras is not None else all_cameras
    
    # Filter episode mapping to only selected cameras
    if selected_cameras is not None:
        valid_indices = [i for i, ep in enumerate(episode_mapping) if ep['camera'] in selected_cameras]
        filtered_episode_mapping = [episode_mapping[i] for i in valid_indices]
    else:
        valid_indices = list(range(len(episode_mapping)))
        filtered_episode_mapping = episode_mapping
    
    print("\n" + "="*80)
    print("SEQUENTIAL FILTERING SIMULATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Success threshold:  {success_threshold:.2f}")
    print(f"  Selected cameras:   {cameras}")
    print(f"  Filtering order:    {timesteps} (high to low)")
    
    # Count initial dataset (only selected cameras)
    total_failures = sum(1 for ep in filtered_episode_mapping if ep['label'] == 0)
    total_successes = sum(1 for ep in filtered_episode_mapping if ep['label'] == 1)
    original_success_rate = 100 * total_successes / len(filtered_episode_mapping) if len(filtered_episode_mapping) > 0 else 0
    
    print(f"\n{'INITIAL DATASET':^80}")
    print("-"*80)
    print(f"Total samples:     {len(filtered_episode_mapping)}")
    print(f"Failures (0):      {total_failures} ({100*total_failures/len(filtered_episode_mapping) if len(filtered_episode_mapping) > 0 else 0:.1f}%)")
    print(f"Successes (1):     {total_successes} ({original_success_rate:.1f}%)")
    
    # Track remaining indices (indices into filtered_episode_mapping, not original)
    remaining_local_indices = set(range(len(filtered_episode_mapping)))
    
    cumulative_stats = []
    
    # Sequential filtering
    for stage, timestep in enumerate(timesteps, 1):
        scores = cache_data['timestep_scores'][str(timestep)]
        
        # Filter scores to only selected cameras using valid_indices
        probs = np.array([scores['probs'][i] for i in valid_indices])
        
        # Apply threshold
        decisions = (probs >= success_threshold).astype(int)
        
        # Find what gets filtered at this stage (from remaining pool)
        filtered_this_stage = set()
        for local_idx in remaining_local_indices:
            if decisions[local_idx] == 0:  # Predicted as failure
                filtered_this_stage.add(local_idx)
        
        # Remove filtered from remaining
        remaining_local_indices = remaining_local_indices - filtered_this_stage
        
        # Count statistics
        filtered_failures = sum(1 for local_idx in filtered_this_stage if filtered_episode_mapping[local_idx]['label'] == 0)
        filtered_successes = sum(1 for local_idx in filtered_this_stage if filtered_episode_mapping[local_idx]['label'] == 1)
        
        remaining_failures = sum(1 for local_idx in remaining_local_indices if filtered_episode_mapping[local_idx]['label'] == 0)
        remaining_successes = sum(1 for local_idx in remaining_local_indices if filtered_episode_mapping[local_idx]['label'] == 1)
        
        # Print stage results
        print(f"\n{'='*80}")
        print(f"Stage {stage}: Timestep {timestep}")
        print(f"{'='*80}")
        
        total_filtered = filtered_failures + filtered_successes
        total_remaining = remaining_failures + remaining_successes
        
        print(f"\n{'Action':<20} {'Failures':<15} {'Successes':<15} {'Total':<15}")
        print("-"*65)
        print(f"{'Filtered this stage':<20} {filtered_failures:<15} {filtered_successes:<15} {total_filtered:<15}")
        print(f"{'Remaining':<20} {remaining_failures:<15} {remaining_successes:<15} {total_remaining:<15}")
        
        if total_remaining > 0:
            remaining_success_rate = 100 * remaining_successes / total_remaining
            print(f"\nRemaining success rate: {remaining_success_rate:.1f}%")
        
        # Sample filtered episodes
        if len(filtered_this_stage) > 0:
            print(f"\n{'SAMPLE FILTERED EPISODES (first 15)':^65}")
            print("-"*65)
            
            # Group by camera
            by_camera = {}
            for cam in cameras:
                by_camera[cam] = {'failures': [], 'successes': []}
            
            for local_idx in sorted(filtered_this_stage)[:15]:
                ep = filtered_episode_mapping[local_idx]
                label_type = 'failures' if ep['label'] == 0 else 'successes'
                by_camera[ep['camera']][label_type].append(ep['episode'])
            
            for camera in cameras:
                if by_camera[camera]['failures'] or by_camera[camera]['successes']:
                    print(f"\n{camera}:")
                    if by_camera[camera]['failures']:
                        print(f"  Failures:  {by_camera[camera]['failures']}")
                    if by_camera[camera]['successes']:
                        print(f"  Successes: {by_camera[camera]['successes']}")
        
        # Show remaining episodes after this stage
        if len(remaining_local_indices) > 0:
            print(f"\n{'SAMPLE REMAINING EPISODES (first 20)':^65}")
            print("-"*65)
            
            # Group by camera
            remaining_by_camera = {}
            for cam in cameras:
                remaining_by_camera[cam] = {'failures': [], 'successes': []}
            
            for local_idx in sorted(remaining_local_indices)[:20]:
                ep = filtered_episode_mapping[local_idx]
                label_type = 'failures' if ep['label'] == 0 else 'successes'
                remaining_by_camera[ep['camera']][label_type].append(ep['episode'])
            
            for camera in cameras:
                if remaining_by_camera[camera]['failures'] or remaining_by_camera[camera]['successes']:
                    print(f"\n{camera}:")
                    if remaining_by_camera[camera]['successes']:
                        print(f"  Successes: {remaining_by_camera[camera]['successes']}")
                    if remaining_by_camera[camera]['failures']:
                        print(f"  Failures:  {remaining_by_camera[camera]['failures']}")
        
        cumulative_stats.append({
            'stage': stage,
            'timestep': timestep,
            'filtered_failures': filtered_failures,
            'filtered_successes': filtered_successes,
            'remaining_failures': remaining_failures,
            'remaining_successes': remaining_successes
        })
    
    # Summary table
    print("\n\n" + "="*80)
    print("CUMULATIVE FILTERING SUMMARY")
    print("="*80)
    print(f"\n{'Stage':<8} {'Timestep':<12} {'Remaining':<20} {'Success Rate':<15} {'Cumulative Filtered'}")
    print(f"{'':8} {'':12} {'Fail':<8} {'Success':<8} {'':15} {'Fail':<8} {'Success':<8}")
    print("-"*80)
    
    cumulative_filtered_failures = 0
    cumulative_filtered_successes = 0
    
    for stat in cumulative_stats:
        cumulative_filtered_failures += stat['filtered_failures']
        cumulative_filtered_successes += stat['filtered_successes']
        
        rf = stat['remaining_failures']
        rs = stat['remaining_successes']
        total_remaining = rf + rs
        success_rate = 100 * rs / total_remaining if total_remaining > 0 else 0
        
        print(f"{stat['stage']:<8} {stat['timestep']:<12} {rf:<8} {rs:<8} {success_rate:>6.1f}%         "
              f"{cumulative_filtered_failures:<8} {cumulative_filtered_successes:<8}")
    
    # Final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    final_remaining_failures = cumulative_stats[-1]['remaining_failures']
    final_remaining_successes = cumulative_stats[-1]['remaining_successes']
    final_total_remaining = final_remaining_failures + final_remaining_successes
    
    final_success_rate = 100 * final_remaining_successes / final_total_remaining if final_total_remaining > 0 else 0
    
    print(f"\nOriginal dataset:")
    print(f"  Total samples:  {len(filtered_episode_mapping)}")
    print(f"  Success rate:   {original_success_rate:.1f}%")
    
    print(f"\nAfter sequential filtering:")
    print(f"  Remaining:      {final_total_remaining} ({100*final_total_remaining/len(filtered_episode_mapping) if len(filtered_episode_mapping) > 0 else 0:.1f}% of original)")
    print(f"  Success rate:   {final_success_rate:.1f}%")
    print(f"  Improvement:    {final_success_rate - original_success_rate:+.1f}%")
    
    print(f"\nTotal filtered out:")
    print(f"  Failures:       {cumulative_filtered_failures}/{total_failures} ({100*cumulative_filtered_failures/total_failures if total_failures > 0 else 0:.1f}%)")
    print(f"  Successes:      {cumulative_filtered_successes}/{total_successes} ({100*cumulative_filtered_successes/total_successes if total_successes > 0 else 0:.1f}%)")
    
    # Show final remaining episodes
    print("\n" + "="*80)
    print("FINAL REMAINING EPISODES")
    print("="*80)
    
    by_camera = {}
    for cam in cameras:
        by_camera[cam] = {'failures': [], 'successes': []}
    
    for local_idx in sorted(remaining_local_indices):
        ep = filtered_episode_mapping[local_idx]
        label_type = 'failures' if ep['label'] == 0 else 'successes'
        by_camera[ep['camera']][label_type].append(ep['episode'])
    
    for camera in cameras:
        print(f"\n{camera}: {len(by_camera[camera]['failures']) + len(by_camera[camera]['successes'])} episodes")
        if by_camera[camera]['successes']:
            print(f"  Successes ({len(by_camera[camera]['successes'])}): {by_camera[camera]['successes'][:30]}")
            if len(by_camera[camera]['successes']) > 30:
                print(f"    ... and {len(by_camera[camera]['successes']) - 30} more")
        if by_camera[camera]['failures']:
            print(f"  Failures ({len(by_camera[camera]['failures'])}):  {by_camera[camera]['failures'][:30]}")
            if len(by_camera[camera]['failures']) > 30:
                print(f"    ... and {len(by_camera[camera]['failures']) - 30} more")
    
    print("="*80)
    
    return cumulative_stats, remaining_local_indices


if __name__ == "__main__":
    # Load cached scores
    # Update this to match your cache file
    model_type = 'convlstm'  # 'simple' or 'convlstm'
    task = 'basketball'  # or 'buttonpress'
    cache_file = f'combined_scores_cache_{task}_{model_type}.json'
    
    cache_data = load_cache(cache_file)
    
    print("\n" + "="*80)
    print("FAST EXPERIMENTATION MODE - COMBINED CLASSIFIER")
    print("="*80)
    
    # Example 1: Single threshold
    print("\n[Example 1] Testing single threshold...")
    results = apply_threshold(
        cache_data,
        success_threshold=0.5,
        aggregation='any'
    )
    print_results(results, cache_data)
    
    # Example 2: Grid search over thresholds
    print("\n[Example 2] Grid search over thresholds...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_config, all_results = threshold_grid_search(
        cache_data,
        thresholds=thresholds,
        aggregation='any'
    )
    
    print("\n[Best Configuration Details]")
    print_results(best_config, cache_data)
    
    # Example 3: Sequential filtering simulation
    print("\n[Example 3] Sequential filtering simulation...")
    simulate_sequential_filtering(
        cache_data,
        success_threshold=0.5
    )
    
    # Example 4: Try a different threshold
    print("\n[Example 4] Sequential filtering with lower threshold (0.3)...")
    simulate_sequential_filtering(
        cache_data,
        success_threshold=0.3
    )
    
    print("\n" + "="*80)
    print("Done! Modify this script to test your own configurations.")
    print("To use specific cameras, add: selected_cameras=['corner', 'corner2']")
    print("="*80)

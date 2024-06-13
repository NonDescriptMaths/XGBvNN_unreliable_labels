import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'run': ['final_fixed'],
        'benchmark': [
            # 'labelled_exp', 
            'missing_labels'
            ],
        'model': [
            'neural_net',
            'xgboost',
            ],
        'learning_rate': [
            # 0.001,              
            0.01
            ],
        'query_alpha': [0,0.5,1],
        'full_train_every': [-1,1,10],
        # 'update_ratio': [0.1,0.5],
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'exp.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)
